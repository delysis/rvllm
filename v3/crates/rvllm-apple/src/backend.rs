use std::collections::HashSet;

use rvllm_core::{AppleCtx, AppleError, ReqId, Result, RvllmError, TokenId};
use serde::{Deserialize, Serialize};

use crate::handoff::HandoffCapsule;
use crate::plan::{AppleRuntimePlan, RolloutBucket};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum AppleLaunchKind {
    Prefill,
    Rollout,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct AppleLaunchTicket {
    pub step_id: u64,
    pub kind: AppleLaunchKind,
    pub bucket: Option<RolloutBucket>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct StepToken {
    pub req_id: ReqId,
    pub token_id: TokenId,
    pub finished: bool,
}

pub trait AppleBackend {
    fn prepare(&mut self, plan: &AppleRuntimePlan) -> Result<()>;
    fn launch_prefill(&mut self, handoff: &HandoffCapsule) -> Result<AppleLaunchTicket>;
    fn launch_rollout(
        &mut self,
        handoff: &HandoffCapsule,
        bucket: RolloutBucket,
    ) -> Result<AppleLaunchTicket>;
    fn collect(&mut self, ticket: AppleLaunchTicket) -> Result<Vec<StepToken>>;
}

#[derive(Debug, Default)]
pub struct StubAppleBackend {
    prepared: bool,
    next_step_id: u64,
    pending: HashSet<AppleLaunchTicket>,
}

impl StubAppleBackend {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn ctx(op: &'static str) -> AppleCtx {
        AppleCtx {
            backend: "stub-apple",
            op,
            device: "host-test",
        }
    }

    fn ensure_prepared(&self, op: &'static str) -> Result<()> {
        if self.prepared {
            Ok(())
        } else {
            Err(RvllmError::apple(
                AppleError::NotPrepared {
                    backend: "stub-apple",
                },
                Self::ctx(op),
            ))
        }
    }

    fn next_ticket(
        &mut self,
        kind: AppleLaunchKind,
        bucket: Option<RolloutBucket>,
    ) -> AppleLaunchTicket {
        let ticket = AppleLaunchTicket {
            step_id: self.next_step_id,
            kind,
            bucket,
        };
        self.next_step_id += 1;
        self.pending.insert(ticket);
        ticket
    }

    fn ensure_pending(&mut self, ticket: AppleLaunchTicket) -> Result<()> {
        if self.pending.remove(&ticket) {
            Ok(())
        } else {
            Err(RvllmError::apple(
                AppleError::LaunchNotPending {
                    step_id: ticket.step_id,
                },
                Self::ctx("collect"),
            ))
        }
    }
}

impl AppleBackend for StubAppleBackend {
    fn prepare(&mut self, plan: &AppleRuntimePlan) -> Result<()> {
        plan.validate()?;
        self.prepared = true;
        self.pending.clear();
        Ok(())
    }

    fn launch_prefill(&mut self, handoff: &HandoffCapsule) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_prefill")?;
        handoff.validate()?;
        Ok(self.next_ticket(AppleLaunchKind::Prefill, None))
    }

    fn launch_rollout(
        &mut self,
        handoff: &HandoffCapsule,
        bucket: RolloutBucket,
    ) -> Result<AppleLaunchTicket> {
        self.ensure_prepared("launch_rollout")?;
        handoff.validate()?;
        Ok(self.next_ticket(AppleLaunchKind::Rollout, Some(bucket)))
    }

    fn collect(&mut self, ticket: AppleLaunchTicket) -> Result<Vec<StepToken>> {
        self.ensure_prepared("collect")?;
        self.ensure_pending(ticket)?;
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::AppleAcceleratorTarget;
    use crate::handoff::HandoffKind;
    use crate::plan::{AppleBackendMode, AppleMatmulConfig, RolloutBucket};

    fn plan() -> AppleRuntimePlan {
        AppleRuntimePlan {
            target: AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: AppleBackendMode::MetalPrefillMetalDecode,
            matmul: AppleMatmulConfig::fp16(),
            rollout_bucket: None,
            rollout_tokens: 1,
            private_ane_opt_in: false,
        }
    }

    fn handoff() -> HandoffCapsule {
        HandoffCapsule::new(
            HandoffKind::MetalPrefillToMetalDecode,
            vec![ReqId(1)],
            vec![TokenId(10)],
            vec![0, 1],
            vec![0],
            vec![1],
        )
    }

    #[test]
    fn launch_requires_prepare_with_typed_error() {
        let mut backend = StubAppleBackend::new();
        let err = match backend.launch_prefill(&handoff()) {
            Ok(ticket) => panic!("launch unexpectedly succeeded with ticket {ticket:?}"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            RvllmError::Apple {
                err: AppleError::NotPrepared {
                    backend: "stub-apple"
                },
                ..
            }
        ));

        let err = match backend.launch_rollout(&handoff(), RolloutBucket { seqs: 1, tokens: 1 }) {
            Ok(ticket) => panic!("launch unexpectedly succeeded with ticket {ticket:?}"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            RvllmError::Apple {
                err: AppleError::NotPrepared {
                    backend: "stub-apple"
                },
                ..
            }
        ));
    }

    #[test]
    fn backend_launch_collect_lifecycle_requires_real_ticket() {
        let mut backend = StubAppleBackend::new();
        assert!(backend.prepare(&plan()).is_ok());
        let forged_ticket = AppleLaunchTicket {
            step_id: 99,
            kind: AppleLaunchKind::Rollout,
            bucket: Some(RolloutBucket { seqs: 1, tokens: 1 }),
        };
        let err = match backend.collect(forged_ticket) {
            Ok(tokens) => panic!("collect unexpectedly succeeded with tokens {tokens:?}"),
            Err(err) => err,
        };
        assert!(matches!(
            err,
            RvllmError::Apple {
                err: AppleError::LaunchNotPending { step_id: 99 },
                ..
            }
        ));

        let ticket = match backend.launch_rollout(&handoff(), RolloutBucket { seqs: 1, tokens: 1 })
        {
            Ok(v) => v,
            Err(e) => panic!("unexpected launch error: {e}"),
        };
        assert_eq!(ticket.kind, AppleLaunchKind::Rollout);
        assert!(backend.collect(ticket).is_ok());
    }
}
