use serde::Serialize;
use serde_json::Value;
use std::cmp::Ordering;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize)]
struct QueueSummary {
    schema: &'static str,
    results_directory: String,
    jobs: Vec<JobSummary>,
}

#[derive(Debug, Serialize)]
struct JobSummary {
    id: String,
    directory: String,
    model: Option<String>,
    prompt_tokens: Option<u64>,
    generation_tokens: Option<u64>,
    requested_trials: Option<u64>,
    status: String,
    sampled_conditions_eligible: Option<bool>,
    sampled_controls_eligible: Option<bool>,
    comparison_stratum: Option<Value>,
    observed_strata: Vec<Value>,
    exit_code: Option<i64>,
    trials: Vec<Trial>,
    reported_averages: Option<ReportedAverages>,
    statistics: Option<Statistics>,
    errors: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct Trial {
    trial: u64,
    prompt_tps: f64,
    generation_tps: f64,
    peak_memory: f64,
    total_time: f64,
}

#[derive(Debug, PartialEq, Serialize)]
struct ReportedAverages {
    prompt_tps: f64,
    generation_tps: f64,
    peak_memory: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_time: Option<f64>,
}

#[derive(Debug, PartialEq, Serialize)]
struct Statistics {
    prompt_tps: Distribution,
    generation_tps: Distribution,
    peak_memory: Distribution,
    total_time: Distribution,
}

#[derive(Debug, PartialEq, Serialize)]
struct Distribution {
    median: f64,
    min: f64,
    max: f64,
}

fn main() {
    let mut args = env::args_os();
    let program = args.next().unwrap_or_default();
    let Some(results_directory) = args.next() else {
        eprintln!(
            "usage: {} RESULTS_DIRECTORY",
            PathBuf::from(program).display()
        );
        std::process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("expected exactly one RESULTS_DIRECTORY");
        std::process::exit(2);
    }

    match summarize(Path::new(&results_directory))
        .and_then(|summary| serde_json::to_string_pretty(&summary).map_err(|e| e.to_string()))
    {
        Ok(json) => println!("{json}"),
        Err(error) => {
            eprintln!("rvllm_mlx_queue_summary: {error}");
            std::process::exit(1);
        }
    }
}

fn summarize(results_directory: &Path) -> Result<QueueSummary, String> {
    let entries = fs::read_dir(results_directory)
        .map_err(|e| format!("read {}: {e}", results_directory.display()))?;
    let mut directories = Vec::new();
    for entry in entries {
        let entry =
            entry.map_err(|e| format!("read {} entry: {e}", results_directory.display()))?;
        let file_type = entry
            .file_type()
            .map_err(|e| format!("inspect {}: {e}", entry.path().display()))?;
        if file_type.is_dir() {
            directories.push(entry.path());
        }
    }
    directories.sort();

    let jobs = directories.iter().map(|path| summarize_job(path)).collect();
    Ok(QueueSummary {
        schema: "rvllm.mlx_queue_summary.v1",
        results_directory: results_directory.display().to_string(),
        jobs,
    })
}

fn summarize_job(directory: &Path) -> JobSummary {
    let directory_name = directory
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("<non-utf8-directory>")
        .to_owned();
    let mut summary = JobSummary {
        id: directory_name,
        directory: directory.display().to_string(),
        model: None,
        prompt_tokens: None,
        generation_tokens: None,
        requested_trials: None,
        status: "incomplete".to_owned(),
        sampled_conditions_eligible: None,
        sampled_controls_eligible: None,
        comparison_stratum: None,
        observed_strata: Vec::new(),
        exit_code: None,
        trials: Vec::new(),
        reported_averages: None,
        statistics: None,
        errors: Vec::new(),
    };

    match read_json(&directory.join("job.json")) {
        Ok(job) => parse_job(&job, &mut summary),
        Err(error) => summary.errors.push(error),
    }
    match read_json(&directory.join("report.json")) {
        Ok(report) => parse_report(&report, &mut summary),
        Err(error) => summary.errors.push(error),
    }
    match fs::read_to_string(directory.join("trial.stdout")) {
        Ok(stdout) => match parse_benchmark_stdout(&stdout) {
            Ok((trials, averages)) => {
                summary.statistics = statistics(&trials);
                summary.trials = trials;
                summary.reported_averages = averages;
            }
            Err(error) => summary.errors.push(format!("parse trial.stdout: {error}")),
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            summary.errors.push("missing trial.stdout".to_owned());
        }
        Err(error) => summary.errors.push(format!("read trial.stdout: {error}")),
    }
    validate_summary(&mut summary);
    summary
}

fn validate_summary(summary: &mut JobSummary) {
    if summary.model.is_none() {
        summary
            .errors
            .push("job.json is missing --model".to_owned());
    }
    for (name, value) in [
        ("--prompt-tokens", summary.prompt_tokens),
        ("--generation-tokens", summary.generation_tokens),
        ("--num-trials", summary.requested_trials),
    ] {
        if value.is_none() {
            summary.errors.push(format!("job.json is missing {name}"));
        }
    }
    if let Some(requested) = summary.requested_trials {
        if summary.trials.len() as u64 != requested {
            summary.errors.push(format!(
                "trial count mismatch: requested {requested}, parsed {}",
                summary.trials.len()
            ));
        }
    }
    if matches!(summary.status.as_str(), "succeeded" | "rejected")
        && summary.reported_averages.is_none()
    {
        summary
            .errors
            .push("terminal result is missing reported Averages".to_owned());
    }
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn parse_job(job: &Value, summary: &mut JobSummary) {
    if let Some(id) = job.get("id").and_then(Value::as_str) {
        summary.id = id.to_owned();
    }
    let Some(args) = job
        .pointer("/command/args")
        .and_then(Value::as_array)
        .map(|args| args.iter().filter_map(Value::as_str).collect::<Vec<_>>())
    else {
        summary
            .errors
            .push("job.json command.args is missing or invalid".to_owned());
        return;
    };
    summary.model = argument(&args, "--model").map(str::to_owned);
    summary.prompt_tokens = numeric_argument(&args, "--prompt-tokens", &mut summary.errors);
    summary.generation_tokens = numeric_argument(&args, "--generation-tokens", &mut summary.errors);
    summary.requested_trials = numeric_argument(&args, "--num-trials", &mut summary.errors);
}

fn argument<'a>(args: &[&'a str], name: &str) -> Option<&'a str> {
    args.iter().enumerate().find_map(|(index, value)| {
        value
            .strip_prefix(name)
            .and_then(|rest| rest.strip_prefix('='))
            .or_else(|| {
                (*value == name)
                    .then(|| args.get(index + 1).copied())
                    .flatten()
            })
    })
}

fn numeric_argument(args: &[&str], name: &str, errors: &mut Vec<String>) -> Option<u64> {
    let value = argument(args, name)?;
    match value.parse() {
        Ok(value) => Some(value),
        Err(error) => {
            errors.push(format!(
                "job.json {name} value {value:?} is invalid: {error}"
            ));
            None
        }
    }
}

fn parse_report(report: &Value, summary: &mut JobSummary) {
    summary.status = report
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("invalid_report")
        .to_owned();
    if summary.status == "invalid_report" {
        summary
            .errors
            .push("report.json status is missing or invalid".to_owned());
    }
    summary.sampled_conditions_eligible = report
        .get("sampled_conditions_eligible")
        .and_then(Value::as_bool);
    summary.sampled_controls_eligible = report
        .pointer("/measurement/sampled_controls_eligible")
        .and_then(Value::as_bool);
    summary.comparison_stratum = report
        .pointer("/measurement/comparison_stratum")
        .filter(|value| !value.is_null())
        .cloned();
    for pointer in [
        "/measurement/comparison_stratum",
        "/measurement/start_host",
        "/measurement/end_host",
    ] {
        push_unique_stratum(&mut summary.observed_strata, report.pointer(pointer));
    }
    if let Some(samples) = report
        .pointer("/measurement/power_samples")
        .and_then(Value::as_array)
    {
        for sample in samples {
            push_unique_stratum(&mut summary.observed_strata, sample.get("controls"));
        }
    }
    summary.exit_code = report.get("exit_code").and_then(Value::as_i64);
}

fn push_unique_stratum(strata: &mut Vec<Value>, value: Option<&Value>) {
    if let Some(value) = value.filter(|value| !value.is_null()) {
        if !strata.contains(value) {
            strata.push(value.clone());
        }
    }
}

fn parse_benchmark_stdout(stdout: &str) -> Result<(Vec<Trial>, Option<ReportedAverages>), String> {
    let mut trials = Vec::new();
    let mut averages = None;
    for line in stdout.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("Trial ") {
            let trial = parse_trial(rest)?;
            if trials
                .iter()
                .any(|existing: &Trial| existing.trial == trial.trial)
            {
                return Err(format!("duplicate Trial {}", trial.trial));
            }
            trials.push(trial);
        } else if let Some(rest) = line.strip_prefix("Averages:") {
            if averages.is_some() {
                return Err("duplicate Averages line".to_owned());
            }
            averages = Some(parse_averages(rest.trim())?);
        }
    }
    if trials.is_empty() {
        return Err("no Trial lines".to_owned());
    }
    Ok((trials, averages))
}

fn parse_trial(rest: &str) -> Result<Trial, String> {
    let (trial, fields) = rest
        .split_once(':')
        .ok_or_else(|| format!("malformed Trial line: {rest}"))?;
    let trial = trial
        .trim()
        .parse::<u64>()
        .map_err(|e| format!("invalid trial number {trial:?}: {e}"))?;
    let values = parse_fields(fields)?;
    Ok(Trial {
        trial,
        prompt_tps: required_field(&values, "prompt_tps")?,
        generation_tps: required_field(&values, "generation_tps")?,
        peak_memory: required_field(&values, "peak_memory")?,
        total_time: required_field(&values, "total_time")?,
    })
}

fn parse_averages(fields: &str) -> Result<ReportedAverages, String> {
    let values = parse_fields(fields)?;
    Ok(ReportedAverages {
        prompt_tps: required_field(&values, "prompt_tps")?,
        generation_tps: required_field(&values, "generation_tps")?,
        peak_memory: required_field(&values, "peak_memory")?,
        total_time: values
            .iter()
            .find_map(|(name, value)| (*name == "total_time").then_some(*value)),
    })
}

fn parse_fields(fields: &str) -> Result<Vec<(&str, f64)>, String> {
    fields
        .split(',')
        .map(|field| {
            let (name, raw) = field
                .trim()
                .split_once('=')
                .ok_or_else(|| format!("malformed metric field {field:?}"))?;
            let value = raw
                .parse::<f64>()
                .map_err(|e| format!("invalid {name} value {raw:?}: {e}"))?;
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be a finite non-negative number"));
            }
            Ok((name, value))
        })
        .collect()
}

fn required_field(values: &[(&str, f64)], name: &str) -> Result<f64, String> {
    let matches = values
        .iter()
        .filter_map(|(candidate, value)| (*candidate == name).then_some(*value))
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [value] => Ok(*value),
        [] => Err(format!("missing {name}")),
        _ => Err(format!("duplicate {name}")),
    }
}

fn statistics(trials: &[Trial]) -> Option<Statistics> {
    (!trials.is_empty()).then(|| Statistics {
        prompt_tps: distribution(trials.iter().map(|trial| trial.prompt_tps)),
        generation_tps: distribution(trials.iter().map(|trial| trial.generation_tps)),
        peak_memory: distribution(trials.iter().map(|trial| trial.peak_memory)),
        total_time: distribution(trials.iter().map(|trial| trial.total_time)),
    })
}

fn distribution(values: impl Iterator<Item = f64>) -> Distribution {
    let mut values = values.collect::<Vec<_>>();
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let middle = values.len() / 2;
    let median = if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    };
    Distribution {
        median,
        min: values[0],
        max: values[values.len() - 1],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OUTPUT: &str = r#"
Running warmup..
Timing with prompt_tokens=256, generation_tokens=64, batch_size=1.
Trial 1:  prompt_tps=94.494, generation_tps=2.583, peak_memory=24.066, total_time=27.503
Trial 2:  prompt_tps=73.956, generation_tps=3.387, peak_memory=24.067, total_time=22.371
Averages: prompt_tps=84.225, generation_tps=2.985, peak_memory=24.067
"#;

    #[test]
    fn parses_mlx_trials_and_reported_averages() {
        let (trials, averages) = parse_benchmark_stdout(OUTPUT).unwrap();
        assert_eq!(trials.len(), 2);
        assert_eq!(trials[0].trial, 1);
        assert_eq!(trials[1].total_time, 22.371);
        assert_eq!(averages.unwrap().generation_tps, 2.985);
    }

    #[test]
    fn calculates_all_distributions_from_trials() {
        let (trials, _) = parse_benchmark_stdout(OUTPUT).unwrap();
        let statistics = statistics(&trials).unwrap();
        assert_eq!(statistics.prompt_tps.min, 73.956);
        assert_eq!(statistics.prompt_tps.median, (73.956 + 94.494) / 2.0);
        assert_eq!(statistics.prompt_tps.max, 94.494);
        assert_eq!(statistics.total_time.min, 22.371);
        assert_eq!(statistics.total_time.max, 27.503);
    }

    #[test]
    fn rejects_missing_and_non_finite_metrics() {
        let missing = "Trial 1: prompt_tps=1, generation_tps=2, peak_memory=3\n";
        assert!(
            parse_benchmark_stdout(missing)
                .unwrap_err()
                .contains("missing total_time")
        );
        let infinite = "Trial 1: prompt_tps=inf, generation_tps=2, peak_memory=3, total_time=4\n";
        assert!(
            parse_benchmark_stdout(infinite)
                .unwrap_err()
                .contains("finite non-negative")
        );
    }

    #[test]
    fn rejects_duplicate_trial_numbers() {
        let duplicate = "Trial 1: prompt_tps=1, generation_tps=2, peak_memory=3, total_time=4\n\
                         Trial 1: prompt_tps=5, generation_tps=6, peak_memory=7, total_time=8\n";
        assert_eq!(
            parse_benchmark_stdout(duplicate).unwrap_err(),
            "duplicate Trial 1"
        );
    }

    #[test]
    fn reads_both_argument_forms() {
        let args = ["--model=/models/gemma", "--prompt-tokens", "256"];
        assert_eq!(argument(&args, "--model"), Some("/models/gemma"));
        assert_eq!(argument(&args, "--prompt-tokens"), Some("256"));
        assert_eq!(argument(&args, "--num-trials"), None);
    }

    #[test]
    fn parses_job_identity_and_work_shape() {
        let job: Value = serde_json::from_str(
            r#"{
                "id":"mlx-q4-1024",
                "command":{"args":["-m","mlx_lm.benchmark","--model","/m/q4",
                    "--prompt-tokens=1024","--generation-tokens","64","--num-trials","7"]}
            }"#,
        )
        .unwrap();
        let mut summary = empty_job_summary();
        parse_job(&job, &mut summary);
        assert_eq!(summary.id, "mlx-q4-1024");
        assert_eq!(summary.model.as_deref(), Some("/m/q4"));
        assert_eq!(summary.prompt_tokens, Some(1024));
        assert_eq!(summary.generation_tokens, Some(64));
        assert_eq!(summary.requested_trials, Some(7));
        assert!(summary.errors.is_empty());
    }

    #[test]
    fn parses_report_eligibility_and_all_observed_strata() {
        let report: Value = serde_json::from_str(
            r#"{
                "status":"rejected", "exit_code":0, "sampled_conditions_eligible":false,
                "measurement":{
                    "sampled_controls_eligible":false,
                    "comparison_stratum":null,
                    "start_host":{"thermal_state":0,"low_power_mode":true},
                    "end_host":{"thermal_state":1,"low_power_mode":true},
                    "power_samples":[
                        {"controls":{"thermal_state":0,"low_power_mode":true}},
                        {"controls":{"thermal_state":1,"low_power_mode":true}}
                    ]
                }
            }"#,
        )
        .unwrap();
        let mut summary = empty_job_summary();
        parse_report(&report, &mut summary);
        assert_eq!(summary.status, "rejected");
        assert_eq!(summary.sampled_conditions_eligible, Some(false));
        assert_eq!(summary.sampled_controls_eligible, Some(false));
        assert_eq!(summary.exit_code, Some(0));
        assert_eq!(summary.observed_strata.len(), 2);
    }

    #[test]
    fn validates_required_identity_trial_count_and_terminal_averages() {
        let mut summary = empty_job_summary();
        summary.status = "rejected".to_owned();
        summary.requested_trials = Some(2);
        summary.trials.push(Trial {
            trial: 1,
            prompt_tps: 1.0,
            generation_tps: 2.0,
            peak_memory: 3.0,
            total_time: 4.0,
        });
        validate_summary(&mut summary);
        assert!(summary.errors.iter().any(|error| error.contains("--model")));
        assert!(
            summary
                .errors
                .iter()
                .any(|error| error.contains("trial count mismatch"))
        );
        assert!(
            summary
                .errors
                .iter()
                .any(|error| error.contains("missing reported Averages"))
        );
    }

    fn empty_job_summary() -> JobSummary {
        JobSummary {
            id: "directory-name".to_owned(),
            directory: "/results/directory-name".to_owned(),
            model: None,
            prompt_tokens: None,
            generation_tokens: None,
            requested_trials: None,
            status: "incomplete".to_owned(),
            sampled_conditions_eligible: None,
            sampled_controls_eligible: None,
            comparison_stratum: None,
            observed_strata: Vec::new(),
            exit_code: None,
            trials: Vec::new(),
            reported_averages: None,
            statistics: None,
            errors: Vec::new(),
        }
    }
}
