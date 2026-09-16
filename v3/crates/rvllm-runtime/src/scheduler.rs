//! Scheduler per spec 07.
//!
//! Emits one `BatchPlan` per step of exactly one variant (`Prefill`,
//! `Decode`, or `Idle`). No mixed prefill+decode in the same step —
//! that was one of the metadata-coupling sources in v2.

use std::time::{Duration, Instant};

use rvllm_core::{ReqId, TokenId};

use crate::paged_kv::KvChainHandle;
use crate::sched_state::{ReqState, Request};

/// Bucket list for decode. Must match graph-capture buckets.
pub const DECODE_BUCKETS: &[u32] = &[1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256];

/// Smallest decode bucket that holds `actual` sequences.
pub fn bucket_for(actual: u32) -> Option<u32> {
    DECODE_BUCKETS.iter().copied().find(|&b| b >= actual)
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SchedulerConfig {
    pub max_prefill_tokens: u32,
    pub max_decode_sequences: u32,
    pub decode_target: Duration,
    /// Waiting requests gain one effective priority point per interval. This
    /// bounds starvation without erasing explicit caller priority.
    pub priority_aging_interval: Duration,
    /// A ready prefill request that has made no progress for this long gets a
    /// reduced chunk even while decode deadlines remain overdue.
    pub prefill_starvation_limit: Duration,
    /// Reduced aggregate prefill budget used for the starvation escape hatch
    /// under sustained decode pressure.
    pub decode_pressure_prefill_tokens: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_prefill_tokens: if cfg!(target_os = "ios") { 64 } else { 128 },
            max_decode_sequences: 256,
            decode_target: Duration::from_millis(50),
            priority_aging_interval: Duration::from_millis(250),
            prefill_starvation_limit: Duration::from_millis(250),
            decode_pressure_prefill_tokens: if cfg!(target_os = "ios") { 16 } else { 32 },
        }
    }
}

impl SchedulerConfig {
    /// Derive the decode deadline from a calibrated batch-one token latency.
    /// The 1.25x factor is the Apple interactive-latency promotion contract.
    #[must_use]
    pub fn with_calibrated_single_token_latency(mut self, latency: Duration) -> Self {
        let calibrated = latency.mul_f64(1.25);
        self.decode_target = calibrated.max(Duration::from_micros(1));
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SchedulerError {
    EmptyPrompt { req_id: ReqId },
    DuplicateRequest { req_id: ReqId },
    UnknownRequest { req_id: ReqId },
    InvalidPrefillCommit { req_id: ReqId, tokens: u32 },
}

/// Scheduler output for one step.
#[derive(Debug)]
pub enum BatchPlan {
    Idle,
    Prefill {
        req_ids: Vec<ReqId>,
        prompt_tokens_flat: Vec<TokenId>,
        cu_seqlens_q: Vec<u32>,
        query_start_positions: Vec<u32>,
        context_lens: Vec<u32>,
        kv_chains: Vec<Option<KvChainHandle>>,
    },
    Decode {
        req_ids: Vec<ReqId>,
        bucket: u32,
        last_tokens: Vec<TokenId>,
        positions: Vec<u32>,
        context_lens: Vec<u32>,
        kv_chains: Vec<Option<KvChainHandle>>,
    },
}

impl BatchPlan {
    #[must_use]
    pub fn req_ids(&self) -> &[ReqId] {
        match self {
            Self::Idle => &[],
            Self::Prefill { req_ids, .. } | Self::Decode { req_ids, .. } => req_ids,
        }
    }
}

pub struct Scheduler {
    requests: Vec<Request>,
    config: SchedulerConfig,
}

impl Scheduler {
    pub fn new() -> Self {
        Self::with_config(SchedulerConfig::default())
    }

    pub fn with_config(config: SchedulerConfig) -> Self {
        assert!(config.max_prefill_tokens > 0);
        assert!(config.max_decode_sequences > 0);
        assert!(!config.decode_target.is_zero());
        assert!(!config.priority_aging_interval.is_zero());
        assert!(!config.prefill_starvation_limit.is_zero());
        assert!(config.decode_pressure_prefill_tokens > 0);
        Self {
            requests: Vec::with_capacity(256),
            config,
        }
    }

    pub fn enqueue(&mut self, req: Request) {
        self.try_enqueue(req).expect("invalid scheduler request");
    }

    pub fn try_enqueue(&mut self, req: Request) -> Result<(), SchedulerError> {
        if req.prompt_tokens.is_empty() {
            return Err(SchedulerError::EmptyPrompt { req_id: req.id });
        }
        if self
            .requests
            .iter()
            .any(|existing| existing.id == req.id && existing.is_alive())
        {
            return Err(SchedulerError::DuplicateRequest { req_id: req.id });
        }
        self.requests.push(req);
        Ok(())
    }

    pub fn num_alive(&self) -> usize {
        self.requests.iter().filter(|r| r.is_alive()).count()
    }

    /// Pick a plan without committing request progress. State advances only
    /// after the backend step successfully completes.
    pub fn schedule(&mut self) -> BatchPlan {
        let now = Instant::now();
        let has_prefill = self.requests.iter().any(Request::is_schedulable_prefill);
        let decode_due = self.requests.iter().any(|request| {
            request.is_schedulable_decode()
                && request
                    .decode_deadline
                    .map_or(true, |deadline| deadline <= now)
        });
        let prefill_starved = self.requests.iter().any(|request| {
            request.is_schedulable_prefill()
                && now.saturating_duration_since(request.last_progress_time)
                    >= self.config.prefill_starvation_limit
        });

        if has_prefill && (!decode_due || prefill_starved) {
            let mut indices: Vec<_> = self
                .requests
                .iter()
                .enumerate()
                .filter(|(_, request)| request.is_schedulable_prefill())
                .map(|(index, _)| index)
                .collect();
            indices.sort_by_key(|&index| {
                (
                    std::cmp::Reverse(aged_priority(
                        &self.requests[index],
                        now,
                        self.config.priority_aging_interval,
                    )),
                    index,
                )
            });

            let mut req_ids = Vec::new();
            let mut prompt_tokens_flat = Vec::new();
            let mut cu_seqlens_q = Vec::new();
            let mut query_start_positions = Vec::new();
            let mut context_lens = Vec::new();
            let mut kv_chains = Vec::new();
            cu_seqlens_q.push(0);
            let mut budget = if decode_due {
                self.config
                    .max_prefill_tokens
                    .min(self.config.decode_pressure_prefill_tokens)
            } else {
                self.config.max_prefill_tokens
            };
            for index in indices {
                if budget == 0 {
                    break;
                }
                let request = &self.requests[index];
                let query_tokens = request.prefill_remaining().min(budget);
                if query_tokens == 0 {
                    continue;
                }
                let start = request.prefill_cursor as usize;
                let end = start + query_tokens as usize;
                req_ids.push(request.id);
                prompt_tokens_flat.extend_from_slice(&request.prompt_tokens[start..end]);
                cu_seqlens_q.push(prompt_tokens_flat.len() as u32);
                query_start_positions.push(start as u32);
                context_lens.push(end as u32);
                kv_chains.push(request.kv_chain);
                budget -= query_tokens;
            }
            return BatchPlan::Prefill {
                req_ids,
                prompt_tokens_flat,
                cu_seqlens_q,
                query_start_positions,
                context_lens,
                kv_chains,
            };
        }

        let mut active: Vec<&Request> = self
            .requests
            .iter()
            .filter(|r| r.is_schedulable_decode())
            .collect();
        if active.is_empty() {
            // No decoder can block prefill, so any ready prompt is schedulable.
            if has_prefill {
                for request in self
                    .requests
                    .iter_mut()
                    .filter(|r| r.is_schedulable_prefill())
                {
                    request.decode_deadline = None;
                }
                return self.schedule();
            }
            return BatchPlan::Idle;
        }
        active.sort_by_key(|request| {
            (
                request.decode_deadline,
                std::cmp::Reverse(aged_priority(
                    request,
                    now,
                    self.config.priority_aging_interval,
                )),
                request.id,
            )
        });
        active.truncate(self.config.max_decode_sequences as usize);
        let actual = active.len() as u32;
        let bucket = bucket_for(actual).expect("max_decode_sequences must fit a decode bucket");
        let mut req_ids = Vec::with_capacity(active.len());
        let mut last_tokens = Vec::with_capacity(active.len());
        let mut positions = Vec::with_capacity(active.len());
        let mut context_lens = Vec::with_capacity(active.len());
        let mut kv_chains = Vec::with_capacity(active.len());
        for r in &active {
            req_ids.push(r.id);
            last_tokens.push(
                *r.output_tokens
                    .last()
                    .unwrap_or(&r.prompt_tokens[r.prompt_tokens.len() - 1]),
            );
            positions.push(r.context_len() - 1);
            context_lens.push(r.context_len());
            kv_chains.push(r.kv_chain);
        }
        BatchPlan::Decode {
            req_ids,
            bucket,
            last_tokens,
            positions,
            context_lens,
            kv_chains,
        }
    }

    pub fn bind_kv_chain(
        &mut self,
        req_id: ReqId,
        chain: KvChainHandle,
    ) -> Result<(), SchedulerError> {
        let request = self
            .requests
            .iter_mut()
            .rev()
            .find(|request| request.id == req_id && request.is_alive())
            .ok_or(SchedulerError::UnknownRequest { req_id })?;
        request.bind_kv_chain(chain);
        Ok(())
    }

    /// Temporarily remove one live request from scheduling while preserving
    /// all request state, deadlines, aging timestamps, and KV ownership.
    pub fn set_backpressured(
        &mut self,
        req_id: ReqId,
        backpressured: bool,
    ) -> Result<(), SchedulerError> {
        let request = self
            .requests
            .iter_mut()
            .rev()
            .find(|request| request.id == req_id && request.is_alive())
            .ok_or(SchedulerError::UnknownRequest { req_id })?;
        request.backpressured = backpressured;
        Ok(())
    }

    pub fn unbind_kv_chain(
        &mut self,
        req_id: ReqId,
        expected: KvChainHandle,
    ) -> Result<(), SchedulerError> {
        let request = self
            .requests
            .iter_mut()
            .rev()
            .find(|request| request.id == req_id && request.is_alive())
            .ok_or(SchedulerError::UnknownRequest { req_id })?;
        if request.kv_chain == Some(expected) {
            request.kv_chain = None;
        }
        Ok(())
    }

    #[must_use]
    pub fn kv_chain_for(&self, req_id: ReqId) -> Option<KvChainHandle> {
        self.requests
            .iter()
            .rev()
            .find(|request| request.id == req_id && request.is_alive())
            .and_then(|request| request.kv_chain)
    }

    #[must_use]
    pub fn request_is_alive(&self, req_id: ReqId) -> bool {
        self.requests
            .iter()
            .rev()
            .any(|request| request.id == req_id && request.is_alive())
    }

    #[must_use]
    pub fn request_is_accelerator_in_flight(&self, req_id: ReqId) -> bool {
        self.requests
            .iter()
            .rev()
            .find(|request| request.id == req_id)
            .is_some_and(|request| request.accelerator_in_flight)
    }

    pub fn mark_accelerator_in_flight(&mut self, req_ids: &[ReqId], in_flight: bool) {
        for &req_id in req_ids {
            if let Some(request) = self
                .requests
                .iter_mut()
                .rev()
                .find(|request| request.id == req_id)
            {
                request.accelerator_in_flight = in_flight;
            }
        }
    }

    pub fn commit_prefill(&mut self, completed: &[(ReqId, u32)]) -> Result<(), SchedulerError> {
        for &(id, tokens) in completed {
            let request = self
                .requests
                .iter_mut()
                .rev()
                .find(|request| request.id == id && request.is_alive())
                .ok_or(SchedulerError::UnknownRequest { req_id: id })?;
            request
                .commit_prefill(tokens)
                .map_err(|_| SchedulerError::InvalidPrefillCommit { req_id: id, tokens })?;
            if request.is_decoding() {
                request.decode_deadline = None;
            }
        }
        Ok(())
    }

    /// Commit per-seq outputs from a completed decode step.
    pub fn commit_decode(&mut self, req_tokens: &[(ReqId, TokenId)]) {
        for &(id, tok) in req_tokens {
            if let Some(r) = self
                .requests
                .iter_mut()
                .rev()
                .find(|r| r.id == id && r.is_alive())
            {
                r.push_output(tok);
                if r.is_decoding() {
                    r.decode_deadline = Some(Instant::now() + self.config.decode_target);
                } else {
                    r.decode_deadline = None;
                }
            }
        }
    }

    pub fn cancel_request(&mut self, id: ReqId) -> bool {
        if let Some(request) = self
            .requests
            .iter_mut()
            .rev()
            .find(|request| request.id == id && request.is_alive())
        {
            request.state = ReqState::Aborted;
            request.decode_deadline = None;
            true
        } else {
            false
        }
    }

    /// Finish one request before its max-token budget is exhausted.
    pub fn finish_request(&mut self, id: ReqId) -> bool {
        if let Some(r) = self
            .requests
            .iter_mut()
            .rev()
            .find(|r| r.id == id && r.is_alive())
        {
            r.state = ReqState::Finished;
            true
        } else {
            false
        }
    }
}

fn aged_priority(request: &Request, now: Instant, interval: Duration) -> u64 {
    let waited = now.saturating_duration_since(request.arrival_time);
    let interval_ns = interval.as_nanos().max(1);
    let age_points = waited.as_nanos() / interval_ns;
    u64::from(request.priority).saturating_add(u64::try_from(age_points).unwrap_or(u64::MAX))
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_rounds_up() {
        assert_eq!(bucket_for(1), Some(1));
        assert_eq!(bucket_for(3), Some(4));
        assert_eq!(bucket_for(100), Some(128));
        assert_eq!(bucket_for(256), Some(256));
        assert_eq!(bucket_for(257), None);
    }

    #[test]
    fn calibrated_decode_deadline_is_one_and_a_quarter_batch_one_latency() {
        let config = SchedulerConfig::default()
            .with_calibrated_single_token_latency(Duration::from_millis(20));
        assert_eq!(config.decode_target, Duration::from_millis(25));

        let minimum =
            SchedulerConfig::default().with_calibrated_single_token_latency(Duration::ZERO);
        assert_eq!(minimum.decode_target, Duration::from_micros(1));
    }

    #[test]
    fn priority_aging_eventually_schedules_an_old_cache_miss_first() {
        let mut scheduler = Scheduler::with_config(SchedulerConfig {
            max_prefill_tokens: 1,
            priority_aging_interval: Duration::from_millis(10),
            ..SchedulerConfig::default()
        });
        scheduler.enqueue(
            Request::new(ReqId(1), vec![TokenId(1)], 1)
                .with_priority(0)
                .with_arrival_time(Instant::now() - Duration::from_secs(1)),
        );
        scheduler.enqueue(Request::new(ReqId(2), vec![TokenId(2)], 1).with_priority(10));

        let BatchPlan::Prefill { req_ids, .. } = scheduler.schedule() else {
            panic!("expected prefill");
        };
        assert_eq!(req_ids, vec![ReqId(1)]);
    }

    #[test]
    fn sustained_decode_pressure_allows_only_a_reduced_starvation_prefill_chunk() {
        let mut scheduler = Scheduler::with_config(SchedulerConfig {
            max_prefill_tokens: 128,
            decode_pressure_prefill_tokens: 8,
            prefill_starvation_limit: Duration::from_millis(10),
            ..SchedulerConfig::default()
        });
        let mut decoder = Request::new(ReqId(1), vec![TokenId(1)], 4);
        decoder.state = ReqState::Decoding;
        decoder.decode_deadline = Some(Instant::now() - Duration::from_millis(1));
        scheduler.enqueue(decoder);
        scheduler.enqueue(
            Request::new(ReqId(2), vec![TokenId(2); 20], 1)
                .with_arrival_time(Instant::now() - Duration::from_secs(1)),
        );

        let BatchPlan::Prefill {
            req_ids,
            prompt_tokens_flat,
            ..
        } = scheduler.schedule()
        else {
            panic!("expected starvation escape prefill");
        };
        assert_eq!(req_ids, vec![ReqId(2)]);
        assert_eq!(prompt_tokens_flat.len(), 8);

        scheduler.commit_prefill(&[(ReqId(2), 8)]).unwrap();
        assert!(matches!(scheduler.schedule(), BatchPlan::Decode { .. }));
    }

    #[test]
    fn schedule_emits_prefill_then_decode() {
        let mut s = Scheduler::new();
        s.enqueue(Request::new(ReqId(1), vec![TokenId(10), TokenId(11)], 4));
        s.enqueue(Request::new(ReqId(2), vec![TokenId(20), TokenId(21)], 4));
        match s.schedule() {
            BatchPlan::Prefill { req_ids, .. } => assert_eq!(req_ids.len(), 2),
            other => panic!("expected Prefill, got {other:?}"),
        }
        s.commit_prefill(&[(ReqId(1), 2), (ReqId(2), 2)]).unwrap();
        // After commit of first prefill round, next schedule is Decode.
        match s.schedule() {
            BatchPlan::Decode {
                req_ids, bucket, ..
            } => {
                assert_eq!(req_ids.len(), 2);
                assert_eq!(bucket, 2);
            }
            other => panic!("expected Decode, got {other:?}"),
        }
    }

    #[test]
    fn backpressured_request_keeps_state_but_is_not_scheduled() {
        let mut scheduler = Scheduler::new();
        scheduler.enqueue(Request::new(ReqId(1), vec![TokenId(10)], 3));
        scheduler.enqueue(Request::new(ReqId(2), vec![TokenId(20)], 3));
        let BatchPlan::Prefill { req_ids, .. } = scheduler.schedule() else {
            panic!("expected joint prefill")
        };
        assert_eq!(req_ids, vec![ReqId(1), ReqId(2)]);
        scheduler
            .commit_prefill(&[(ReqId(1), 1), (ReqId(2), 1)])
            .unwrap();

        scheduler.set_backpressured(ReqId(1), true).unwrap();
        let BatchPlan::Decode {
            req_ids, positions, ..
        } = scheduler.schedule()
        else {
            panic!("expected fast request decode")
        };
        assert_eq!(req_ids, vec![ReqId(2)]);
        assert_eq!(positions, vec![0]);
        scheduler.commit_decode(&[(ReqId(2), TokenId(21))]);

        scheduler.set_backpressured(ReqId(1), false).unwrap();
        let BatchPlan::Decode {
            req_ids, positions, ..
        } = scheduler.schedule()
        else {
            panic!("expected resumed decode batch")
        };
        assert!(req_ids.contains(&ReqId(1)));
        let resumed_index = req_ids.iter().position(|id| *id == ReqId(1)).unwrap();
        assert_eq!(positions[resumed_index], 0);
        let fast_index = req_ids.iter().position(|id| *id == ReqId(2)).unwrap();
        assert_eq!(positions[fast_index], 1);
    }

    #[test]
    fn decode_plan_advances_last_token_position_and_context_len_after_commit() {
        let mut s = Scheduler::new();
        s.enqueue(Request::new(
            ReqId(1),
            vec![TokenId(0), TokenId(1), TokenId(2)],
            3,
        ));

        match s.schedule() {
            BatchPlan::Prefill {
                req_ids,
                prompt_tokens_flat,
                cu_seqlens_q,
                query_start_positions,
                context_lens,
                ..
            } => {
                assert_eq!(req_ids, vec![ReqId(1)]);
                assert_eq!(prompt_tokens_flat, vec![TokenId(0), TokenId(1), TokenId(2)]);
                assert_eq!(cu_seqlens_q, vec![0, 3]);
                assert_eq!(query_start_positions, vec![0]);
                assert_eq!(context_lens, vec![3]);
            }
            other => panic!("expected Prefill, got {other:?}"),
        }
        s.commit_prefill(&[(ReqId(1), 3)]).unwrap();

        match s.schedule() {
            BatchPlan::Decode {
                req_ids,
                last_tokens,
                positions,
                context_lens,
                ..
            } => {
                assert_eq!(req_ids, vec![ReqId(1)]);
                assert_eq!(last_tokens, vec![TokenId(2)]);
                assert_eq!(positions, vec![2]);
                assert_eq!(context_lens, vec![3]);
            }
            other => panic!("expected Decode, got {other:?}"),
        }

        s.commit_decode(&[(ReqId(1), TokenId(3))]);
        match s.schedule() {
            BatchPlan::Decode {
                req_ids,
                last_tokens,
                positions,
                context_lens,
                ..
            } => {
                assert_eq!(req_ids, vec![ReqId(1)]);
                assert_eq!(last_tokens, vec![TokenId(3)]);
                assert_eq!(positions, vec![3]);
                assert_eq!(context_lens, vec![4]);
            }
            other => panic!("expected Decode, got {other:?}"),
        }

        s.commit_decode(&[(ReqId(1), TokenId(4))]);
        match s.schedule() {
            BatchPlan::Decode {
                req_ids,
                last_tokens,
                positions,
                context_lens,
                ..
            } => {
                assert_eq!(req_ids, vec![ReqId(1)]);
                assert_eq!(last_tokens, vec![TokenId(4)]);
                assert_eq!(positions, vec![4]);
                assert_eq!(context_lens, vec![5]);
            }
            other => panic!("expected Decode, got {other:?}"),
        }

        s.commit_decode(&[(ReqId(1), TokenId(5))]);
        assert!(matches!(s.schedule(), BatchPlan::Idle));
    }

    #[test]
    fn decode_plan_bucket_four_preserves_independent_context_lens() {
        let mut s = Scheduler::new();
        s.enqueue(Request::new(ReqId(1), vec![TokenId(2)], 1));
        s.enqueue(Request::new(ReqId(2), vec![TokenId(0), TokenId(3)], 1));
        s.enqueue(Request::new(
            ReqId(3),
            vec![TokenId(0), TokenId(1), TokenId(4)],
            1,
        ));
        s.enqueue(Request::new(ReqId(4), vec![TokenId(2)], 1));

        assert!(matches!(s.schedule(), BatchPlan::Prefill { .. }));
        s.commit_prefill(&[(ReqId(1), 1), (ReqId(2), 2), (ReqId(3), 3), (ReqId(4), 1)])
            .unwrap();
        match s.schedule() {
            BatchPlan::Decode {
                req_ids,
                bucket,
                last_tokens,
                positions,
                context_lens,
                ..
            } => {
                assert_eq!(bucket, 4);
                assert_eq!(req_ids, vec![ReqId(1), ReqId(2), ReqId(3), ReqId(4)]);
                assert_eq!(
                    last_tokens,
                    vec![TokenId(2), TokenId(3), TokenId(4), TokenId(2)]
                );
                assert_eq!(positions, vec![0, 1, 2, 0]);
                assert_eq!(context_lens, vec![1, 2, 3, 1]);
            }
            other => panic!("expected Decode, got {other:?}"),
        }

        s.commit_decode(&[
            (ReqId(1), TokenId(3)),
            (ReqId(2), TokenId(4)),
            (ReqId(3), TokenId(5)),
            (ReqId(4), TokenId(3)),
        ]);
        assert!(matches!(s.schedule(), BatchPlan::Idle));
    }

    #[test]
    fn finish_request_removes_request_from_decode() {
        let mut s = Scheduler::new();
        s.enqueue(Request::new(ReqId(1), vec![TokenId(10), TokenId(11)], 4));
        assert!(matches!(s.schedule(), BatchPlan::Prefill { .. }));
        assert!(s.finish_request(ReqId(1)));
        assert_eq!(s.num_alive(), 0);
        assert!(matches!(s.schedule(), BatchPlan::Idle));
        assert!(!s.finish_request(ReqId(99)));
    }

    #[test]
    fn chunked_prefill_uses_absolute_query_starts_and_commits_explicitly() {
        let mut s = Scheduler::with_config(SchedulerConfig {
            max_prefill_tokens: 32,
            ..SchedulerConfig::default()
        });
        s.enqueue(Request::new(ReqId(1), vec![TokenId(7); 65], 1));

        for (expected_start, expected_len) in [(0, 32), (32, 32), (64, 1)] {
            let BatchPlan::Prefill {
                req_ids,
                prompt_tokens_flat,
                query_start_positions,
                context_lens,
                ..
            } = s.schedule()
            else {
                panic!("expected chunked prefill");
            };
            assert_eq!(req_ids, vec![ReqId(1)]);
            assert_eq!(prompt_tokens_flat.len(), expected_len as usize);
            assert_eq!(query_start_positions, vec![expected_start]);
            assert_eq!(context_lens, vec![expected_start + expected_len]);
            s.commit_prefill(&[(ReqId(1), expected_len)]).unwrap();
        }
        assert!(matches!(s.schedule(), BatchPlan::Decode { .. }));
    }

    #[test]
    fn cache_restore_and_cancellation_are_schedulable_without_false_progress() {
        let mut request = Request::new(ReqId(9), vec![TokenId(1); 65], 1);
        request.begin_restore();
        let mut s = Scheduler::new();
        s.enqueue(request);
        assert!(matches!(s.schedule(), BatchPlan::Idle));
        let request = s.requests.iter_mut().find(|r| r.id == ReqId(9)).unwrap();
        request.finish_restore(64).unwrap();
        assert!(matches!(s.schedule(), BatchPlan::Prefill { .. }));
        assert!(s.cancel_request(ReqId(9)));
        assert!(matches!(s.schedule(), BatchPlan::Idle));
    }

    #[test]
    fn request_id_reuse_targets_the_newest_live_generation() {
        let mut s = Scheduler::new();
        s.enqueue(Request::new(ReqId(4), vec![TokenId(1)], 1));
        assert!(s.cancel_request(ReqId(4)));
        s.enqueue(Request::new(ReqId(4), vec![TokenId(9)], 1));

        let BatchPlan::Prefill {
            req_ids,
            prompt_tokens_flat,
            ..
        } = s.schedule()
        else {
            panic!("expected replacement request prefill")
        };
        assert_eq!(req_ids, vec![ReqId(4)]);
        assert_eq!(prompt_tokens_flat, vec![TokenId(9)]);
        s.commit_prefill(&[(ReqId(4), 1)]).unwrap();

        let BatchPlan::Decode { last_tokens, .. } = s.schedule() else {
            panic!("expected replacement request decode")
        };
        assert_eq!(last_tokens, vec![TokenId(9)]);
        s.commit_decode(&[(ReqId(4), TokenId(10))]);
        assert!(matches!(s.schedule(), BatchPlan::Idle));
    }
}
