use std::fs::File;
use std::io::{BufRead, BufReader};

use arrrg::CommandLine;
use policyai::data::{EvaluationReport, Metrics};

#[derive(Clone, Default, Debug, arrrg_derive::CommandLine)]
struct Args {
    #[arrrg(optional, "Filter by non-zero values in this metrics field")]
    metric: Option<String>,

    #[arrrg(optional, "Filter by error messages containing this substring")]
    error: Option<String>,

    #[arrrg(optional, "Filter by input token count exceeding this value")]
    usage_input_tokens: Option<i32>,

    #[arrrg(optional, "Filter by output token count exceeding this value")]
    usage_output_tokens: Option<i32>,

    #[arrrg(
        optional,
        "Output only entries with fewer than this many policyai_fields_matched"
    )]
    policyai_fields_matched: Option<usize>,

    #[arrrg(
        optional,
        "Output only entries with fewer than this many baseline_fields_matched"
    )]
    baseline_fields_matched: Option<usize>,
}

impl Eq for Args {}

impl PartialEq for Args {
    fn eq(&self, other: &Self) -> bool {
        self.metric == other.metric
            && self.error == other.error
            && self.usage_input_tokens == other.usage_input_tokens
            && self.usage_output_tokens == other.usage_output_tokens
            && self.policyai_fields_matched == other.policyai_fields_matched
            && self.baseline_fields_matched == other.baseline_fields_matched
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (args, free) = Args::from_command_line_relaxed(
        "USAGE: policyai-extract-regressions [OPTIONS] <input_file> [input_file...]",
    );

    if free.is_empty() {
        eprintln!("Expected at least one input file");
        std::process::exit(1);
    }

    for input_file in &free {
        process_file(input_file, &args)?;
    }

    Ok(())
}

fn process_file(input_file: &str, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(input_file)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let report: EvaluationReport = match serde_json::from_str(&line) {
            Ok(report) => report,
            Err(e) => {
                eprintln!("Warning: Failed to parse line in {input_file} as EvaluationReport: {e}");
                continue;
            }
        };

        if matches_filters(&report, args) {
            println!("{line}");
        }
    }

    Ok(())
}

fn matches_filters(report: &EvaluationReport, args: &Args) -> bool {
    // Check metric filter
    if let Some(ref metric_field) = args.metric {
        if !matches_metric_filter(&report.metrics, metric_field) {
            return false;
        }
    }

    // Check error filter
    if let Some(ref error_substring) = args.error {
        if !matches_error_filter(&report.metrics, error_substring) {
            return false;
        }
    }

    // Check input token filter
    if let Some(min_input_tokens) = args.usage_input_tokens {
        if !matches_input_token_filter(&report.metrics, min_input_tokens) {
            return false;
        }
    }

    // Check output token filter
    if let Some(min_output_tokens) = args.usage_output_tokens {
        if !matches_output_token_filter(&report.metrics, min_output_tokens) {
            return false;
        }
    }

    // Check policyai_fields_matched filter
    if !matches_threshold_filter(args.policyai_fields_matched, || {
        report.metrics.policyai_fields_matched
    }) {
        return false;
    }

    // Check baseline_fields_matched filter
    if !matches_threshold_filter(args.baseline_fields_matched, || {
        report.metrics.baseline_fields_matched
    }) {
        return false;
    }

    true
}

fn matches_threshold_filter<F>(threshold: Option<usize>, field_selector: F) -> bool
where
    F: FnOnce() -> usize,
{
    match threshold {
        Some(threshold) => field_selector() < threshold,
        None => true,
    }
}

fn matches_metric_filter(metrics: &Metrics, field_name: &str) -> bool {
    match field_name {
        "policyai_fields_matched" => metrics.policyai_fields_matched > 0,
        "baseline_fields_matched" => metrics.baseline_fields_matched > 0,
        "policyai_fields_with_wrong_value" => metrics.policyai_fields_with_wrong_value > 0,
        "baseline_fields_with_wrong_value" => metrics.baseline_fields_with_wrong_value > 0,
        "policyai_fields_missing" => metrics.policyai_fields_missing > 0,
        "baseline_fields_missing" => metrics.baseline_fields_missing > 0,
        "policyai_extra_fields" => metrics.policyai_extra_fields > 0,
        "baseline_extra_fields" => metrics.baseline_extra_fields > 0,
        "policyai_apply_duration_ms" => metrics.policyai_apply_duration_ms > 0,
        "baseline_apply_duration_ms" => metrics.baseline_apply_duration_ms > 0,
        _ => {
            eprintln!("Warning: Unknown metric field: {field_name}");
            false
        }
    }
}

fn check_error_contains(error_opt: &Option<String>, substring: &str) -> bool {
    error_opt
        .as_ref()
        .is_some_and(|error| error.contains(substring))
}

fn matches_error_filter(metrics: &Metrics, error_substring: &str) -> bool {
    check_error_contains(&metrics.policyai_error, error_substring)
        || check_error_contains(&metrics.baseline_error, error_substring)
}

fn matches_token_filter<F>(metrics: &Metrics, min_tokens: i32, field_selector: F) -> bool
where
    F: Fn(&claudius::Usage) -> i32,
{
    if let Some(ref usage) = metrics.policyai_usage {
        if let Some(ref claudius_usage) = usage.claudius_usage {
            if field_selector(claudius_usage) >= min_tokens {
                return true;
            }
        }
    }

    if let Some(ref usage) = metrics.baseline_usage {
        if let Some(ref claudius_usage) = usage.claudius_usage {
            if field_selector(claudius_usage) >= min_tokens {
                return true;
            }
        }
    }

    false
}

fn matches_input_token_filter(metrics: &Metrics, min_tokens: i32) -> bool {
    matches_token_filter(metrics, min_tokens, |usage| usage.input_tokens)
}

fn matches_output_token_filter(metrics: &Metrics, min_tokens: i32) -> bool {
    matches_token_filter(metrics, min_tokens, |usage| usage.output_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use policyai::data::TestDataPoint;

    fn create_test_evaluation_report(
        policyai_fields_matched: usize,
        baseline_fields_matched: usize,
    ) -> EvaluationReport {
        EvaluationReport {
            input: TestDataPoint {
                text: "test".to_string(),
                policies: vec![],
                expected: None,
                conflicts: None,
            },
            metrics: Metrics {
                policyai_fields_matched,
                baseline_fields_matched,
                ..Default::default()
            },
            output: serde_json::json!({}),
            baseline: None,
        }
    }

    fn create_args_with_policyai_threshold(threshold: usize) -> Args {
        Args {
            policyai_fields_matched: Some(threshold),
            ..Default::default()
        }
    }

    fn create_args_with_baseline_threshold(threshold: usize) -> Args {
        Args {
            baseline_fields_matched: Some(threshold),
            ..Default::default()
        }
    }

    fn create_args_with_both_thresholds(
        policyai_threshold: usize,
        baseline_threshold: usize,
    ) -> Args {
        Args {
            policyai_fields_matched: Some(policyai_threshold),
            baseline_fields_matched: Some(baseline_threshold),
            ..Default::default()
        }
    }

    // Test policyai_fields_matched filtering with threshold 0
    #[test]
    fn policyai_threshold_0_excludes_all_entries() {
        let args = create_args_with_policyai_threshold(0);

        // Test with field value 0 - should be excluded since 0 >= 0
        let report = create_test_evaluation_report(0, 0);
        assert!(!matches_filters(&report, &args));

        // Test with field value 1 - should be excluded since 1 >= 0
        let report = create_test_evaluation_report(1, 0);
        assert!(!matches_filters(&report, &args));

        // Test with field value 5 - should be excluded since 5 >= 0
        let report = create_test_evaluation_report(5, 0);
        assert!(!matches_filters(&report, &args));
    }

    // Test policyai_fields_matched filtering with threshold 1
    #[test]
    fn policyai_threshold_1_keeps_only_field_0() {
        let args = create_args_with_policyai_threshold(1);

        // Test with field value 0 - should pass since 0 < 1
        let report = create_test_evaluation_report(0, 0);
        assert!(matches_filters(&report, &args));

        // Test with field value 1 - should be excluded since 1 >= 1
        let report = create_test_evaluation_report(1, 0);
        assert!(!matches_filters(&report, &args));

        // Test with field value 2 - should be excluded since 2 >= 1
        let report = create_test_evaluation_report(2, 0);
        assert!(!matches_filters(&report, &args));
    }

    // Test policyai_fields_matched filtering with threshold 2
    #[test]
    fn policyai_threshold_2_keeps_field_0_and_1() {
        let args = create_args_with_policyai_threshold(2);

        // Test with field value 0 - should pass since 0 < 2
        let report = create_test_evaluation_report(0, 0);
        assert!(matches_filters(&report, &args));

        // Test with field value 1 - should pass since 1 < 2
        let report = create_test_evaluation_report(1, 0);
        assert!(matches_filters(&report, &args));

        // Test with field value 2 - should be excluded since 2 >= 2
        let report = create_test_evaluation_report(2, 0);
        assert!(!matches_filters(&report, &args));

        // Test with field value 3 - should be excluded since 3 >= 2
        let report = create_test_evaluation_report(3, 0);
        assert!(!matches_filters(&report, &args));
    }

    // Test baseline_fields_matched filtering with threshold 0
    #[test]
    fn baseline_threshold_0_excludes_all_entries() {
        let args = create_args_with_baseline_threshold(0);

        // Test with field value 0 - should be excluded since 0 >= 0
        let report = create_test_evaluation_report(0, 0);
        assert!(!matches_filters(&report, &args));

        // Test with field value 1 - should be excluded since 1 >= 0
        let report = create_test_evaluation_report(0, 1);
        assert!(!matches_filters(&report, &args));

        // Test with field value 5 - should be excluded since 5 >= 0
        let report = create_test_evaluation_report(0, 5);
        assert!(!matches_filters(&report, &args));
    }

    // Test baseline_fields_matched filtering with threshold 1
    #[test]
    fn baseline_threshold_1_keeps_only_field_0() {
        let args = create_args_with_baseline_threshold(1);

        // Test with field value 0 - should pass since 0 < 1
        let report = create_test_evaluation_report(0, 0);
        assert!(matches_filters(&report, &args));

        // Test with field value 1 - should be excluded since 1 >= 1
        let report = create_test_evaluation_report(0, 1);
        assert!(!matches_filters(&report, &args));

        // Test with field value 2 - should be excluded since 2 >= 1
        let report = create_test_evaluation_report(0, 2);
        assert!(!matches_filters(&report, &args));
    }

    // Test baseline_fields_matched filtering with threshold 2
    #[test]
    fn baseline_threshold_2_keeps_field_0_and_1() {
        let args = create_args_with_baseline_threshold(2);

        // Test with field value 0 - should pass since 0 < 2
        let report = create_test_evaluation_report(0, 0);
        assert!(matches_filters(&report, &args));

        // Test with field value 1 - should pass since 1 < 2
        let report = create_test_evaluation_report(0, 1);
        assert!(matches_filters(&report, &args));

        // Test with field value 2 - should be excluded since 2 >= 2
        let report = create_test_evaluation_report(0, 2);
        assert!(!matches_filters(&report, &args));

        // Test with field value 3 - should be excluded since 3 >= 2
        let report = create_test_evaluation_report(0, 3);
        assert!(!matches_filters(&report, &args));
    }

    // Test interaction between multiple filters
    #[test]
    fn both_filters_must_pass_for_entry_to_be_included() {
        let args = create_args_with_both_thresholds(2, 2);

        // Both fields pass their thresholds (0 < 2, 1 < 2)
        let report = create_test_evaluation_report(0, 1);
        assert!(matches_filters(&report, &args));

        // Both fields pass their thresholds (1 < 2, 0 < 2)
        let report = create_test_evaluation_report(1, 0);
        assert!(matches_filters(&report, &args));

        // Both fields are at boundary (1 < 2, 1 < 2)
        let report = create_test_evaluation_report(1, 1);
        assert!(matches_filters(&report, &args));

        // policyai field fails threshold (2 >= 2), baseline field passes (1 < 2)
        let report = create_test_evaluation_report(2, 1);
        assert!(!matches_filters(&report, &args));

        // policyai field passes (1 < 2), baseline field fails threshold (2 >= 2)
        let report = create_test_evaluation_report(1, 2);
        assert!(!matches_filters(&report, &args));

        // Both fields fail their thresholds (2 >= 2, 3 >= 2)
        let report = create_test_evaluation_report(2, 3);
        assert!(!matches_filters(&report, &args));
    }

    // Test with no filters - should always pass
    #[test]
    fn no_filters_always_pass() {
        let args = Args::default();

        // Test various field values - all should pass
        let report = create_test_evaluation_report(0, 0);
        assert!(matches_filters(&report, &args));

        let report = create_test_evaluation_report(5, 10);
        assert!(matches_filters(&report, &args));

        let report = create_test_evaluation_report(100, 200);
        assert!(matches_filters(&report, &args));
    }

    // Test asymmetric thresholds
    #[test]
    fn asymmetric_thresholds_work_independently() {
        let args = create_args_with_both_thresholds(1, 3);

        // policyai=0 passes (0 < 1), baseline=2 passes (2 < 3)
        let report = create_test_evaluation_report(0, 2);
        assert!(matches_filters(&report, &args));

        // policyai=1 fails (1 >= 1), baseline=0 passes (0 < 3)
        let report = create_test_evaluation_report(1, 0);
        assert!(!matches_filters(&report, &args));

        // policyai=0 passes (0 < 1), baseline=3 fails (3 >= 3)
        let report = create_test_evaluation_report(0, 3);
        assert!(!matches_filters(&report, &args));
    }

    // Test boundary conditions with large values
    #[test]
    fn large_threshold_values() {
        let args = create_args_with_both_thresholds(1000, 500);

        // Values below thresholds should pass
        let report = create_test_evaluation_report(999, 499);
        assert!(matches_filters(&report, &args));

        // Values equal to thresholds should be excluded
        let report = create_test_evaluation_report(1000, 500);
        assert!(!matches_filters(&report, &args));

        // Values above thresholds should be excluded
        let report = create_test_evaluation_report(1001, 501);
        assert!(!matches_filters(&report, &args));
    }

    // Test interaction with other filter types to ensure they don't interfere
    #[test]
    fn field_filters_independent_of_other_filters() {
        let mut args = create_args_with_policyai_threshold(2);
        args.metric = Some("unknown_metric".to_string()); // This should make metric filter fail

        // Even though metric filter fails, if policyai_fields_matched passes, overall should fail
        let report = create_test_evaluation_report(1, 0);
        assert!(!matches_filters(&report, &args));

        // Reset to no other filters
        args.metric = None;

        // Now should pass since only field filter applies
        assert!(matches_filters(&report, &args));
    }
}
