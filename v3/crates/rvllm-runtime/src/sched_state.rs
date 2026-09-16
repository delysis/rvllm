//! Request state machine per spec 07.
//!
//! Transitions are explicit: `Queued → Prefilling → Decoding → Finished`.
//! `Aborted` reachable from any state.

use std::time::Instant;

use rvllm_core::{ReqId, TokenId};

use crate::paged_kv::KvChainHandle;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ReqState {
    Queued,
    Restoring,
    Prefilling,
    Decoding,
    Finished,
    Aborted,
}

#[derive(Debug)]
pub struct Request {
    pub id: ReqId,
    pub state: ReqState,
    pub prompt_tokens: Vec<TokenId>,
    pub output_tokens: Vec<TokenId>,
    pub max_output_tokens: u32,
    /// Number of prompt tokens whose KV state has been committed.
    pub prefill_cursor: u32,
    /// Stable request-owned KV state. Never inferred from batch order.
    pub kv_chain: Option<KvChainHandle>,
    /// Larger values win within the same deadline class.
    pub priority: u32,
    /// Stable admission time used for priority aging. It is never reset by a
    /// cache restore, prefill chunk, or decode transition.
    pub arrival_time: Instant,
    /// Last successful restore, prefill, or decode commit. This lets the
    /// scheduler bound starvation under a permanently overdue decode cohort.
    pub last_progress_time: Instant,
    /// Deadline for the next decode token once this request is decoding.
    pub decode_deadline: Option<Instant>,
    /// Host response backpressure temporarily removes this request from
    /// accelerator eligibility without changing its lifecycle or KV owner.
    pub backpressured: bool,
    /// True while a submitted accelerator step owns this request's mutable
    /// scheduler/KV transition. In-flight requests are excluded from another
    /// batch until completion or failure is collected.
    pub accelerator_in_flight: bool,
}

impl Request {
    pub fn new(id: ReqId, prompt_tokens: Vec<TokenId>, max_output_tokens: u32) -> Self {
        let now = Instant::now();
        Self {
            id,
            state: ReqState::Queued,
            prompt_tokens,
            output_tokens: Vec::new(),
            max_output_tokens,
            prefill_cursor: 0,
            kv_chain: None,
            priority: 128,
            arrival_time: now,
            last_progress_time: now,
            decode_deadline: None,
            backpressured: false,
            accelerator_in_flight: false,
        }
    }

    #[must_use]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    #[must_use]
    pub fn with_arrival_time(mut self, arrival_time: Instant) -> Self {
        self.arrival_time = arrival_time;
        self.last_progress_time = arrival_time;
        self
    }

    pub fn bind_kv_chain(&mut self, chain: KvChainHandle) {
        self.kv_chain = Some(chain);
    }

    pub fn begin_restore(&mut self) {
        self.state = ReqState::Restoring;
    }

    pub fn finish_restore(&mut self, restored_tokens: u32) -> Result<(), &'static str> {
        if restored_tokens > self.prompt_tokens.len() as u32 {
            return Err("restored prefix exceeds prompt length");
        }
        self.prefill_cursor = restored_tokens;
        self.last_progress_time = Instant::now();
        self.state = if restored_tokens == self.prompt_tokens.len() as u32 {
            ReqState::Decoding
        } else if restored_tokens == 0 {
            ReqState::Queued
        } else {
            ReqState::Prefilling
        };
        Ok(())
    }

    #[must_use]
    pub fn prefill_remaining(&self) -> u32 {
        (self.prompt_tokens.len() as u32).saturating_sub(self.prefill_cursor)
    }

    pub fn commit_prefill(&mut self, tokens: u32) -> Result<(), &'static str> {
        if tokens == 0 || tokens > self.prefill_remaining() {
            return Err("invalid committed prefill token count");
        }
        self.prefill_cursor += tokens;
        self.last_progress_time = Instant::now();
        self.state = if self.prefill_remaining() == 0 {
            ReqState::Decoding
        } else {
            ReqState::Prefilling
        };
        Ok(())
    }

    pub fn is_alive(&self) -> bool {
        !matches!(self.state, ReqState::Finished | ReqState::Aborted)
    }

    pub fn is_decoding(&self) -> bool {
        matches!(self.state, ReqState::Decoding)
    }

    #[must_use]
    pub fn is_schedulable_decode(&self) -> bool {
        self.is_decoding() && !self.backpressured && !self.accelerator_in_flight
    }

    pub fn is_prefill_ready(&self) -> bool {
        matches!(self.state, ReqState::Queued | ReqState::Prefilling)
            && self.prefill_remaining() > 0
    }

    #[must_use]
    pub fn is_schedulable_prefill(&self) -> bool {
        self.is_prefill_ready() && !self.backpressured && !self.accelerator_in_flight
    }

    pub fn context_len(&self) -> u32 {
        (self.prompt_tokens.len() + self.output_tokens.len()) as u32
    }

    pub fn push_output(&mut self, tok: TokenId) {
        self.output_tokens.push(tok);
        self.last_progress_time = Instant::now();
        if self.output_tokens.len() as u32 >= self.max_output_tokens {
            self.state = ReqState::Finished;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_finishes_at_max_tokens() {
        let mut r = Request::new(ReqId(1), vec![TokenId(0); 4], 3);
        r.state = ReqState::Decoding;
        r.push_output(TokenId(1));
        r.push_output(TokenId(2));
        assert!(r.is_decoding());
        r.push_output(TokenId(3));
        assert_eq!(r.state, ReqState::Finished);
        assert!(!r.is_alive());
    }

    #[test]
    fn chunked_prefill_and_restore_advance_only_when_committed() {
        let mut r = Request::new(ReqId(7), vec![TokenId(0); 65], 2);
        assert_eq!(r.prefill_cursor, 0);
        r.commit_prefill(32).unwrap();
        assert_eq!(r.state, ReqState::Prefilling);
        assert_eq!(r.prefill_cursor, 32);
        r.begin_restore();
        r.finish_restore(64).unwrap();
        assert_eq!(r.state, ReqState::Prefilling);
        r.commit_prefill(1).unwrap();
        assert_eq!(r.state, ReqState::Decoding);
    }
}
