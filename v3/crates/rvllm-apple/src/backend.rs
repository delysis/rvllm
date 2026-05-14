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
                AppleError::NotPrepared { backend: "stub-apple" },
                Self::ctx(op),
            ))
        }
    }

    fn next_ticket(&mut self, kind: AppleLaunchKind, bucket: Option<RolloutBucket>) -> AppleLaunchTicket {
        let ticket = AppleLaunchTicket {
            step_id: self.next_step_id,
            kind,
            bucket,
        };
        self.next_step_id += 1;
        ticket
    }
}

impl AppleBackend for StubAppleBackend {
    fn prepare(&mut self, plan: &AppleRuntimePlan) -> Result<()> {
        plan.validate()?;
        self.prepared = true;
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
        let _ = ticket;
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::AppleAcceleratorTarget;
    use crate::handoff::HandoffKind;
    use crate::plan::{AppleBackendMode, RolloutBucket};

    fn plan() -> AppleRuntimePlan {
        AppleRuntimePlan {
            target: AppleAcceleratorTarget::from_device_name("Apple M4 Max", 1),
            mode: AppleBackendMode::MetalPrefillMetalDecode,
            rollout_bucket: None,
            rollout_tokens: 1,
            private_ane_opt_in: false,
        }
    }

    #[test]
    fn backend_requires_prepare_before_launch() {
        let mut backend = StubAppleBackend::new();
        let handoff = HandoffCapsule::new(
            HandoffKind::MetalPrefillToMetalDecode,
            vec![ReqId(1)],
            vec![TokenId(10)],
            vec![0, 1],
            vec![0],
            vec![1],
        );
        assert!(backend.launch_prefill(&handoff).is_err());
        assert!(backend.prepare(&plan()).is_ok());
        let ticket = match backend.launch_rollout(&handoff, RolloutBucket { seqs: 1, tokens: 1 }) {
            Ok(v) => v,
            Err(e) => panic!("unexpected launch error: {e}"),
        };
        assert_eq!(ticket.kind, AppleLaunchKind::Rollout);
    }
}
