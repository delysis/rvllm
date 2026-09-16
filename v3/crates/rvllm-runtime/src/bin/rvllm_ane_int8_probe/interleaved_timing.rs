//! Counterbalanced short slots; power and process counters cover whole blocks.
#![forbid(unsafe_code)]

use super::batch_timing;
use rvllm_runtime::apple_measurement::PowerMonitor;
use serde::Serialize;
use serde_json::{json, Value};
use std::time::{Duration, Instant};

const BLOCKS: usize = 3;
const CYCLES: usize = 32;
const REPETITIONS: usize = 4;
const DRIFT_LIMIT: f64 = 0.05;

#[derive(Clone, Debug, Serialize)]
struct Slot {
    cycle: usize,
    slot: usize,
    batch: bool,
    repetitions: usize,
    useful_outputs: usize,
    start_ms: f64,
    end_ms: f64,
    wall_ms: f64,
    output_bit_identical_to_serial: bool,
}

fn order(block: usize, cycle: usize) -> [bool; 4] {
    if (block + cycle) % 2 == 0 {
        [false, true, true, false]
    } else {
        [true, false, false, true]
    }
}

fn drift(values: [f64; 2]) -> f64 {
    values[0].max(values[1]) / values[0].min(values[1]) - 1.0
}

fn summarize_block(block: usize, slots: &[Slot], wall_ms: f64) -> Result<Value, String> {
    if block >= BLOCKS || slots.len() != CYCLES * 4 || !wall_ms.is_finite() || wall_ms <= 0.0 {
        return Err("invalid interleaved block extent".into());
    }
    let mut occurrences = [[0.0; 2]; 2];
    let mut halves = [[0.0; 2]; 2];
    let mut previous_end = 0.0;
    for (cycle, records) in slots.chunks_exact(4).enumerate() {
        let mut seen = [0; 2];
        for (index, (record, expected_batch)) in records.iter().zip(order(block, cycle)).enumerate()
        {
            if record.cycle != cycle
                || record.slot != index
                || record.batch != expected_batch
                || record.repetitions != REPETITIONS
                || record.useful_outputs != 2 * REPETITIONS
                || !record.output_bit_identical_to_serial
                || !record.start_ms.is_finite()
                || !record.end_ms.is_finite()
                || !record.wall_ms.is_finite()
                || record.start_ms < previous_end
                || record.end_ms > wall_ms
                || record.wall_ms <= 0.0
                || ((record.end_ms - record.start_ms) - record.wall_ms).abs() > 0.000001
            {
                return Err("invalid interleaved order, work, interval or output".into());
            }
            previous_end = record.end_ms;
            let variant = usize::from(record.batch);
            occurrences[variant][seen[variant]] += record.wall_ms;
            seen[variant] += 1;
            halves[variant][usize::from(cycle >= CYCLES / 2)] += record.wall_ms;
        }
    }
    let occurrence_drift = occurrences.map(drift);
    let half_drift = halves.map(drift);
    let passed = occurrence_drift
        .iter()
        .chain(&half_drift)
        .all(|&d| d <= DRIFT_LIMIT);
    let totals = occurrences.map(|t| t[0] + t[1]);
    let outputs_per_variant = CYCLES * 2 * REPETITIONS * 2;
    Ok(json!({"block":block,"repeat_drift_passed":passed,
        "serial_occurrence_drift_fraction":occurrence_drift[0],
        "batch_occurrence_drift_fraction":occurrence_drift[1],
        "serial_half_drift_fraction":half_drift[0],"batch_half_drift_fraction":half_drift[1],
        "serial_total_ms":totals[0],"batch_total_ms":totals[1],
        "baseline_over_candidate_wall":totals[0]/totals[1],
        "useful_outputs_per_variant":outputs_per_variant,
        "serial_ms_per_useful_output":totals[0]/outputs_per_variant as f64,
        "batch_ms_per_useful_output":totals[1]/outputs_per_variant as f64}))
}

/// `evaluate` times only projection work, then checks its output before returning.
/// Errors stop the run. There is no skip, retry or replacement-sample path.
pub(super) fn run(
    monitor: &PowerMonitor,
    exploratory_fair: bool,
    mut evaluate: impl FnMut(bool, usize) -> Result<(Instant, Duration), String>,
) -> Result<Value, String> {
    let mut blocks = Vec::with_capacity(BLOCKS);
    let mut common = None;
    let mut controls_passed = true;
    let mut drift_passed = true;
    let mut ratios = Vec::with_capacity(BLOCKS);
    for block in 0..BLOCKS {
        // Reserve all per-slot storage before the enclosing measurement starts.
        let mut slots = Vec::with_capacity(CYCLES * 4);
        let phase = monitor.begin();
        let origin = Instant::now();
        for cycle in 0..CYCLES {
            for (slot, batch) in order(block, cycle).into_iter().enumerate() {
                let (start, elapsed) = evaluate(batch, REPETITIONS)?;
                let start_ms = start.duration_since(origin).as_secs_f64() * 1000.0;
                let wall_ms = elapsed.as_secs_f64() * 1000.0;
                slots.push(Slot {
                    cycle,
                    slot,
                    batch,
                    repetitions: REPETITIONS,
                    useful_outputs: 2 * REPETITIONS,
                    start_ms,
                    end_ms: start_ms + wall_ms,
                    wall_ms,
                    output_bit_identical_to_serial: true,
                });
            }
        }
        let measurement = phase.finish(CYCLES * 4 * REPETITIONS * 2);
        let summary = summarize_block(
            block,
            &slots,
            measurement["wall_ms"]
                .as_f64()
                .ok_or("missing block wall time")?,
        )?;
        let controls = batch_timing::stratum(&measurement, exploratory_fair);
        let mut rejection = controls.as_ref().err().cloned();
        if let Ok(controls) = controls {
            if let Some(previous) = &common {
                if previous != &controls {
                    rejection = Some("controls differ between blocks".into());
                }
            } else {
                common = Some(controls);
            }
        }
        controls_passed &= rejection.is_none();
        drift_passed &= summary["repeat_drift_passed"] == true;
        ratios.push(
            summary["baseline_over_candidate_wall"]
                .as_f64()
                .ok_or("missing block ratio")?,
        );
        blocks.push(
            json!({"measurement":measurement,"slots":slots,"summary":summary,
            "timing_controls_eligible":rejection.is_none(),"timing_rejection":rejection}),
        );
        if !controls_passed {
            break;
        }
    }
    ratios.sort_by(f64::total_cmp);
    let complete = blocks.len() == BLOCKS;
    Ok(
        json!({"schema":"rvllm.interleaved_ffn_timing.v1","blocks":blocks,
        "status":if !controls_passed || !complete {"inconclusive_controls"}
            else if !drift_passed {"inconclusive_repeat_drift"} else {"complete_component_comparison"},
        "repeat_drift_limit_fraction":DRIFT_LIMIT,"repeat_drift_passed":drift_passed,
        "comparison_stratum":common,"timing_controls_eligible":controls_passed,
        "median_baseline_over_candidate_wall":if complete {Some(ratios[1])} else {None},
        "all_blocks_favor_batch":complete && ratios.iter().all(|&r| r>1.0),
        "claim":"Short counterbalanced FFN wall-time slots. No device-cycle normalization or full-model speedup. Queue contention eligibility is also required."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(block: usize) -> Vec<Slot> {
        let mut now = 0.0;
        let mut slots = Vec::new();
        for cycle in 0..CYCLES {
            for (slot, batch) in order(block, cycle).into_iter().enumerate() {
                let wall_ms = if batch { 8.0 } else { 16.0 };
                slots.push(Slot {
                    cycle,
                    slot,
                    batch,
                    repetitions: REPETITIONS,
                    useful_outputs: 8,
                    start_ms: now,
                    end_ms: now + wall_ms,
                    wall_ms,
                    output_bit_identical_to_serial: true,
                });
                now += wall_ms + 0.1;
            }
        }
        slots
    }

    #[test]
    fn counterbalanced_work_and_ratio() {
        for block in 0..BLOCKS {
            let summary = summarize_block(block, &fixture(block), 2000.0).unwrap();
            assert_eq!(summary["baseline_over_candidate_wall"], 2.0);
            assert_eq!(summary["repeat_drift_passed"], true);
            assert_eq!(summary["useful_outputs_per_variant"], 512);
            assert_eq!(summary["serial_ms_per_useful_output"], 2.0);
        }
        assert!(summarize_block(BLOCKS, &fixture(0), 2000.0).is_err());
        assert!(summarize_block(0, &fixture(1), 2000.0).is_err());
    }

    #[test]
    fn corrupt_order_counts_intervals_and_outputs_reject() {
        let original = fixture(0);
        for mutation in 0..11 {
            let mut slots = original.clone();
            match mutation {
                0 => slots[1].cycle += 1,
                1 => slots[1].slot += 1,
                2 => slots[1].batch = !slots[1].batch,
                3 => slots[1].repetitions += 1,
                4 => slots[1].useful_outputs += 1,
                5 => slots[1].output_bit_identical_to_serial = false,
                6 => slots[1].start_ms = slots[0].start_ms,
                7 => slots[1].wall_ms = f64::NAN,
                8 => slots[1].end_ms = f64::INFINITY,
                9 => slots[1].wall_ms = 0.0,
                _ => {
                    slots.pop();
                }
            }
            assert!(
                summarize_block(0, &slots, 2000.0).is_err(),
                "mutation {mutation}"
            );
        }
        assert!(summarize_block(0, &original, 100.0).is_err());
    }

    #[test]
    fn both_occurrence_and_temporal_drift_are_required() {
        for half_drift in [false, true] {
            let mut slots = fixture(0);
            let mut now = 0.0;
            for record in &mut slots {
                if if half_drift {
                    record.cycle >= CYCLES / 2
                } else {
                    record.slot >= 2
                } {
                    record.wall_ms *= 1.1;
                }
                record.start_ms = now;
                record.end_ms = now + record.wall_ms;
                now = record.end_ms + 0.1;
            }
            let result = summarize_block(0, &slots, now).unwrap();
            assert_eq!(result["repeat_drift_passed"], false);
            let key = if half_drift {
                "serial_half_drift_fraction"
            } else {
                "serial_occurrence_drift_fraction"
            };
            assert!(result[key].as_f64().unwrap() > DRIFT_LIMIT);
        }
    }
}
