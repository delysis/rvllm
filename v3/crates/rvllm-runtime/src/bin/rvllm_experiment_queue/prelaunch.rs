//! Keep observing launch eligibility while potentially large pins are hashed.
#![forbid(unsafe_code)]

use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;

/// The caller supplies the existing stable gate, not just a controls equality
/// check. A false observation defers launch. Verification/probe errors halt it.
/// Always join the scoped verifier before returning, including after a lost
/// gate or an observer error; no hashing may spill into the accelerator phase.
/// There is no retry, timeout-based thread termination, or device call here.
pub(super) fn verify<V, O>(verify_pins: V, mut observe: O) -> Result<bool, String>
where
    V: FnOnce() -> Result<(), String> + Send,
    O: FnMut() -> Result<bool, String>,
{
    if !observe()? {
        return Ok(false);
    }
    std::thread::scope(|scope| {
        let (completed, completion) = mpsc::sync_channel(1);
        let verifier = scope.spawn(move || {
            let result = verify_pins();
            let _ = completed.send(());
            result
        });
        let observation = loop {
            match completion.recv_timeout(Duration::from_secs(1)) {
                Ok(()) | Err(RecvTimeoutError::Disconnected) => break Ok(true),
                Err(RecvTimeoutError::Timeout) => match observe() {
                    Ok(true) => {}
                    other => break other,
                },
            }
        };
        verifier
            .join()
            .map_err(|_| "prelaunch pin verifier panicked; no trial started".to_owned())??;
        if !observation? {
            return Ok(false);
        }
        // Even a fast hash can span a control change or a scheduling gap.
        // The final check happens after joining, never before it.
        observe()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    #[test]
    fn initial_refusal_does_not_start_verification() {
        assert!(!verify(|| panic!("must not hash"), || Ok(false)).unwrap());
    }

    #[test]
    fn initial_observer_error_does_not_start_verification() {
        assert_eq!(
            verify(|| panic!("must not hash"), || Err("probe failed".into())),
            Err("probe failed".into())
        );
    }

    #[test]
    fn successful_verification_requires_a_final_observation() {
        let mut observations = 0;
        let accepted = verify(
            || Ok(()),
            || {
                observations += 1;
                Ok(observations == 1)
            },
        )
        .unwrap();
        assert!(!accepted);
        assert!(observations >= 2);
    }

    #[test]
    fn pending_verification_is_observed_without_waiting_for_completion() {
        let (release, waiting) = mpsc::sync_channel(1);
        let mut release = Some(release);
        let mut observations = 0;
        let accepted = verify(
            move || {
                waiting
                    .recv_timeout(Duration::from_secs(10))
                    .map_err(|e| e.to_string())
            },
            || {
                observations += 1;
                if observations >= 2 {
                    if let Some(release) = release.take() {
                        release.send(()).map_err(|e| e.to_string())?;
                    }
                }
                Ok(true)
            },
        )
        .unwrap();
        assert!(accepted);
        assert!(observations >= 3); // Initial, while pending, after joining.
    }

    #[test]
    fn loss_of_readiness_joins_the_verifier_before_returning() {
        let (release, waiting) = mpsc::sync_channel(1);
        let finished = Arc::new(AtomicBool::new(false));
        let worker_finished = Arc::clone(&finished);
        let mut observations = 0;
        let accepted = verify(
            move || {
                waiting
                    .recv_timeout(Duration::from_secs(10))
                    .map_err(|e| e.to_string())?;
                worker_finished.store(true, Ordering::SeqCst);
                Ok(())
            },
            || {
                observations += 1;
                if observations == 1 {
                    return Ok(true);
                }
                release.send(()).map_err(|e| e.to_string())?;
                Ok(false)
            },
        )
        .unwrap();
        assert!(!accepted);
        assert!(finished.load(Ordering::SeqCst));
        assert_eq!(observations, 2);
    }

    #[test]
    fn observer_error_joins_the_verifier_before_returning() {
        let (release, waiting) = mpsc::sync_channel(1);
        let finished = Arc::new(AtomicBool::new(false));
        let worker_finished = Arc::clone(&finished);
        let mut observations = 0;
        let result = verify(
            move || {
                waiting
                    .recv_timeout(Duration::from_secs(10))
                    .map_err(|e| e.to_string())?;
                worker_finished.store(true, Ordering::SeqCst);
                Ok(())
            },
            || {
                observations += 1;
                if observations == 1 {
                    return Ok(true);
                }
                release.send(()).map_err(|e| e.to_string())?;
                Err("probe failed".into())
            },
        );
        assert_eq!(result, Err("probe failed".into()));
        assert!(finished.load(Ordering::SeqCst));
        assert_eq!(observations, 2);
    }

    #[test]
    fn pin_failure_is_not_downgraded_to_a_retry() {
        assert_eq!(
            verify(|| Err("changed pin".into()), || Ok(true)),
            Err("changed pin".into())
        );
    }

    #[test]
    fn verifier_panic_is_reported_without_detaching_the_thread() {
        assert_eq!(
            verify(|| panic!("host-only verifier fixture"), || Ok(true)),
            Err("prelaunch pin verifier panicked; no trial started".into())
        );
    }
}
