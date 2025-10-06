//! Extract regression entries where baseline passes but PolicyAI fails.
//!
//! This tool reads evaluation reports and outputs only those entries where:
//! 1. The baseline output matches the expected output (baseline passes)
//! 2. The PolicyAI output does not match the expected output (PolicyAI fails)
//!
//! This identifies true regressions where the baseline performs better than PolicyAI.

use std::fs::File;
use std::io::{BufRead, BufReader};

use arrrg::CommandLine;
use policyai::data::EvaluationReport;

#[derive(Clone, Default, Debug, Eq, PartialEq, arrrg_derive::CommandLine)]
struct Args {
    #[arrrg(flag, "Include entries where both baseline and PolicyAI fail")]
    include_baseline_failures: bool,

    #[arrrg(flag, "Ignore whitespace differences in string comparisons")]
    ignore_whitespace: bool,

    #[arrrg(flag, "Ignore order in array comparisons")]
    ignore_array_order: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (args, free) = Args::from_command_line_relaxed(
        "USAGE: policyai-extract-regressions [OPTIONS] <input_file> [input_file...]",
    );

    if free.is_empty() {
        eprintln!("ERROR: Expected at least one input file");
        eprintln!("USAGE: policyai-extract-regressions [OPTIONS] <input_file> [input_file...]");
        std::process::exit(1);
    }

    for input_file in &free {
        process_file(input_file, &args)?;
    }

    Ok(())
}

fn process_file(input_file: &str, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::open(input_file)
        .map_err(|e| format!("Failed to open file '{}': {}", input_file, e))?;
    let reader = BufReader::new(file);

    let mut line_number = 0;
    for line_result in reader.lines() {
        line_number += 1;
        let line = line_result.map_err(|e| {
            format!(
                "Failed to read line {} from file '{}': {}",
                line_number, input_file, e
            )
        })?;

        if line.trim().is_empty() {
            continue;
        }

        let report: EvaluationReport = match serde_json::from_str(&line) {
            Ok(report) => report,
            Err(e) => {
                eprintln!(
                    "Warning: Failed to parse line {} in file '{}' as EvaluationReport: {}",
                    line_number, input_file, e
                );
                continue;
            }
        };

        if is_regression(&report, args) {
            println!("{line}");
        }
    }

    Ok(())
}

/// Determine if this report represents a regression.
/// A regression occurs when:
/// 1. Baseline passes (baseline matches expected) AND PolicyAI fails (output doesn't match expected)
/// 2. OR if --include-baseline-failures is set: both fail but we still want to see them
fn is_regression(report: &EvaluationReport, args: &Args) -> bool {
    // Skip if there's no expected value to compare against
    let expected = match &report.input.expected {
        Some(expected) => expected,
        None => return false,
    };

    // Skip if there's no baseline to compare
    let baseline = match &report.baseline {
        Some(baseline) => baseline,
        None => return false,
    };

    let policyai_passes = values_match(&report.output, expected, args);
    let baseline_passes = values_match(baseline, expected, args);

    // Primary case: baseline passes but PolicyAI fails (true regression)
    if baseline_passes && !policyai_passes {
        return true;
    }

    // Secondary case: include baseline failures if flag is set
    if args.include_baseline_failures && !baseline_passes && !policyai_passes {
        return true;
    }

    false
}

/// Compare two JSON values for semantic equality with configurable matching options.
fn values_match(actual: &serde_json::Value, expected: &serde_json::Value, args: &Args) -> bool {
    values_match_recursive(actual, expected, args)
}

fn values_match_recursive(
    actual: &serde_json::Value,
    expected: &serde_json::Value,
    args: &Args,
) -> bool {
    match (actual, expected) {
        // Numbers: apply 0.1% tolerance for floating point comparisons
        (serde_json::Value::Number(a), serde_json::Value::Number(b)) => {
            if let (Some(a_f64), Some(b_f64)) = (a.as_f64(), b.as_f64()) {
                // Calculate 0.1% tolerance based on the expected value
                let tolerance = b_f64.abs() * 0.001; // 0.1% = 0.001
                (a_f64 - b_f64).abs() <= tolerance
            } else {
                // For integers or numbers that can't be converted to f64, use exact equality
                a == b
            }
        }

        // Strings: apply whitespace normalization if specified
        (serde_json::Value::String(a), serde_json::Value::String(b)) => {
            if args.ignore_whitespace {
                normalize_whitespace(a) == normalize_whitespace(b)
            } else {
                a == b
            }
        }

        // Arrays: apply order-independent comparison if specified
        (serde_json::Value::Array(a), serde_json::Value::Array(b)) => {
            if args.ignore_array_order {
                arrays_match_unordered(a, b, args)
            } else {
                arrays_match_ordered(a, b, args)
            }
        }

        // Objects: recursively compare all fields
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
            if a.len() != b.len() {
                return false;
            }
            for (key, a_val) in a {
                match b.get(key) {
                    Some(b_val) => {
                        if !values_match_recursive(a_val, b_val, args) {
                            return false;
                        }
                    }
                    None => return false,
                }
            }
            true
        }

        // For all other types (null, bool), use exact equality
        _ => actual == expected,
    }
}

fn normalize_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn arrays_match_ordered(a: &[serde_json::Value], b: &[serde_json::Value], args: &Args) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .all(|(a_val, b_val)| values_match_recursive(a_val, b_val, args))
}

fn arrays_match_unordered(a: &[serde_json::Value], b: &[serde_json::Value], args: &Args) -> bool {
    if a.len() != b.len() {
        return false;
    }

    // For each element in a, find a matching element in b
    let mut b_used = vec![false; b.len()];
    for a_val in a {
        let mut found_match = false;
        for (b_idx, b_val) in b.iter().enumerate() {
            if !b_used[b_idx] && values_match_recursive(a_val, b_val, args) {
                b_used[b_idx] = true;
                found_match = true;
                break;
            }
        }
        if !found_match {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use policyai::data::{Metrics, TestDataPoint};
    use policyai::Report;

    fn create_test_report(
        expected: Option<serde_json::Value>,
        policyai_output: serde_json::Value,
        baseline_output: Option<serde_json::Value>,
    ) -> EvaluationReport {
        EvaluationReport {
            input: TestDataPoint {
                text: "test".to_string(),
                policies: vec![],
                expected,
                conflicts: None,
            },
            metrics: Metrics::default(),
            // Report is preserved only for inspection and debugging;
            // the output is what's compared.
            report: Report::default(),
            output: policyai_output,
            baseline: baseline_output,
        }
    }

    #[test]
    fn baseline_pass_policyai_fail_is_regression() {
        let expected = serde_json::json!({"field1": "value1"});
        let policyai_output = serde_json::json!({"field1": "wrong_value"});
        let baseline_output = serde_json::json!({"field1": "value1"});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args::default();
        assert!(is_regression(&report, &args));
    }

    #[test]
    fn baseline_fail_policyai_pass_not_regression() {
        let expected = serde_json::json!({"field1": "value1"});
        let policyai_output = serde_json::json!({"field1": "value1"});
        let baseline_output = serde_json::json!({"field1": "wrong_value"});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args::default();
        assert!(!is_regression(&report, &args));
    }

    #[test]
    fn both_pass_not_regression() {
        let expected = serde_json::json!({"field1": "value1"});
        let policyai_output = serde_json::json!({"field1": "value1"});
        let baseline_output = serde_json::json!({"field1": "value1"});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args::default();
        assert!(!is_regression(&report, &args));
    }

    #[test]
    fn both_fail_not_regression_by_default() {
        let expected = serde_json::json!({"field1": "value1"});
        let policyai_output = serde_json::json!({"field1": "wrong1"});
        let baseline_output = serde_json::json!({"field1": "wrong2"});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args::default();
        assert!(!is_regression(&report, &args));
    }

    #[test]
    fn both_fail_is_regression_with_flag() {
        let expected = serde_json::json!({"field1": "value1"});
        let policyai_output = serde_json::json!({"field1": "wrong1"});
        let baseline_output = serde_json::json!({"field1": "wrong2"});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args {
            include_baseline_failures: true,
            ..Default::default()
        };
        assert!(is_regression(&report, &args));
    }

    #[test]
    fn no_expected_not_regression() {
        let policyai_output = serde_json::json!({"field1": "value1"});
        let baseline_output = serde_json::json!({"field1": "value1"});

        let report = create_test_report(None, policyai_output, Some(baseline_output));

        let args = Args::default();
        assert!(!is_regression(&report, &args));
    }

    #[test]
    fn no_baseline_not_regression() {
        let expected = serde_json::json!({"field1": "value1"});
        let policyai_output = serde_json::json!({"field1": "wrong_value"});

        let report = create_test_report(Some(expected), policyai_output, None);

        let args = Args::default();
        assert!(!is_regression(&report, &args));
    }

    #[test]
    fn complex_json_exact_match() {
        let expected = serde_json::json!({
            "user": {
                "name": "John",
                "age": 30,
                "tags": ["important", "urgent"]
            }
        });

        // Baseline matches exactly
        let baseline_output = expected.clone();
        // PolicyAI has different value
        let policyai_output = serde_json::json!({
            "user": {
                "name": "John",
                "age": 25, // different age
                "tags": ["important", "urgent"]
            }
        });

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args::default();
        assert!(is_regression(&report, &args));
    }

    #[test]
    fn empty_objects_match() {
        let expected = serde_json::json!({});
        let policyai_output = serde_json::json!({});
        let baseline_output = serde_json::json!({});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args::default();
        // Both pass, so not a regression
        assert!(!is_regression(&report, &args));
    }

    #[test]
    fn floating_point_exact_match() {
        let expected = serde_json::json!({"score": 0.123456});
        let policyai_output = serde_json::json!({"score": 0.123456});
        let baseline_output = serde_json::json!({"score": 0.123456});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args::default();
        // Both pass exactly, so not a regression
        assert!(!is_regression(&report, &args));
    }

    #[test]
    fn floating_point_slight_difference_is_regression() {
        let expected = serde_json::json!({"score": 100.0});
        let policyai_output = serde_json::json!({"score": 100.2}); // 0.2% difference > 0.1% tolerance
        let baseline_output = serde_json::json!({"score": 100.0});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args::default();
        // Baseline passes, PolicyAI fails due to difference > 0.1%
        assert!(is_regression(&report, &args));
    }

    #[test]
    fn floating_point_tolerance_prevents_regression() {
        let expected = serde_json::json!({"score": 100.0});
        let policyai_output = serde_json::json!({"score": 100.05}); // within 0.1% tolerance
        let baseline_output = serde_json::json!({"score": 100.0});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args::default();
        // Both should pass with 0.1% tolerance, so not a regression
        assert!(!is_regression(&report, &args));
    }

    #[test]
    fn floating_point_tolerance_still_catches_large_differences() {
        let expected = serde_json::json!({"score": 100.0});
        let policyai_output = serde_json::json!({"score": 102.0}); // 2% difference > 0.1% tolerance
        let baseline_output = serde_json::json!({"score": 100.0});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args::default();
        // Baseline passes, PolicyAI fails due to difference > 0.1%
        assert!(is_regression(&report, &args));
    }

    #[test]
    fn whitespace_differences_ignored_when_flag_set() {
        let expected = serde_json::json!({"message": "Hello World"});
        let policyai_output = serde_json::json!({"message": "Hello  World"}); // extra space
        let baseline_output = serde_json::json!({"message": "Hello World"});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args {
            ignore_whitespace: true,
            ..Default::default()
        };
        // Both should pass with whitespace normalization, so not a regression
        assert!(!is_regression(&report, &args));
    }

    #[test]
    fn whitespace_differences_matter_by_default() {
        let expected = serde_json::json!({"message": "Hello World"});
        let policyai_output = serde_json::json!({"message": "Hello  World"}); // extra space
        let baseline_output = serde_json::json!({"message": "Hello World"});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args::default();
        // Baseline passes, PolicyAI fails due to whitespace difference
        assert!(is_regression(&report, &args));
    }

    #[test]
    fn array_order_ignored_when_flag_set() {
        let expected = serde_json::json!({"tags": ["urgent", "important"]});
        let policyai_output = serde_json::json!({"tags": ["important", "urgent"]}); // different order
        let baseline_output = serde_json::json!({"tags": ["urgent", "important"]});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args {
            ignore_array_order: true,
            ..Default::default()
        };
        // Both should pass with order-independent comparison, so not a regression
        assert!(!is_regression(&report, &args));
    }

    #[test]
    fn array_order_matters_by_default() {
        let expected = serde_json::json!({"tags": ["urgent", "important"]});
        let policyai_output = serde_json::json!({"tags": ["important", "urgent"]}); // different order
        let baseline_output = serde_json::json!({"tags": ["urgent", "important"]});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args::default();
        // Baseline passes, PolicyAI fails due to order difference
        assert!(is_regression(&report, &args));
    }

    #[test]
    fn complex_nested_matching_with_all_options() {
        let expected = serde_json::json!({
            "user": {
                "name": "John Doe",
                "score": 95.0,
                "tags": ["premium", "verified"]
            }
        });

        let policyai_output = serde_json::json!({
            "user": {
                "name": "John  Doe", // extra whitespace
                "score": 95.05,      // within 0.1% tolerance (95 * 0.001 = 0.095)
                "tags": ["verified", "premium"] // different order
            }
        });

        let baseline_output = expected.clone();

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args {
            ignore_whitespace: true,
            ignore_array_order: true,
            ..Default::default()
        };
        // Both should pass with all matching options enabled
        assert!(!is_regression(&report, &args));
    }

    #[test]
    fn array_with_different_elements_still_fails() {
        let expected = serde_json::json!({"tags": ["urgent", "important"]});
        let policyai_output = serde_json::json!({"tags": ["urgent", "spam"]}); // different element
        let baseline_output = serde_json::json!({"tags": ["urgent", "important"]});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args {
            ignore_array_order: true,
            ..Default::default()
        };
        // Even with order-independent comparison, different elements should fail
        assert!(is_regression(&report, &args));
    }

    #[test]
    fn mixed_type_comparison_fails() {
        let expected = serde_json::json!({"value": 42});
        let policyai_output = serde_json::json!({"value": "42"}); // string instead of number
        let baseline_output = serde_json::json!({"value": 42});

        let report = create_test_report(Some(expected), policyai_output, Some(baseline_output));

        let args = Args::default();
        // Type mismatch should always fail regardless of matching options
        assert!(is_regression(&report, &args));
    }
}
