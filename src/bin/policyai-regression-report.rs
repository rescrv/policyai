//! Generate regression analysis reports from PolicyAI evaluation data.
//!
//! This binary reads evaluation reports and generates comprehensive regression analysis
//! using confusion matrices and metrics to compare PolicyAI performance against baselines.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};

use arrrg::CommandLine;
use policyai::analysis::{ConfusionMatrix, RegressionAnalysis};
use policyai::data::{EvaluationReport, Metrics};

#[derive(Clone, Default, Debug, Eq, PartialEq, arrrg_derive::CommandLine)]
struct Args {
    #[arrrg(flag, "Print detailed metrics for each field")]
    verbose: bool,
    #[arrrg(optional, "Output format (json, csv, text)")]
    format: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (args, free) = Args::from_command_line_relaxed(
        "USAGE: policyai-regression-report [OPTIONS] [input_file...]",
    );

    let reports = if free.is_empty() {
        read_from_stdin()?
    } else {
        read_from_files(&free)?
    };

    if reports.is_empty() {
        eprintln!("No evaluation reports found in input");
        return Ok(());
    }

    let mut analysis = RegressionAnalysis::new();
    let mut field_matrices = FieldAccuracyMatrices::new();

    for report in &reports {
        analysis.add_report(&report.metrics);
        field_matrices.add_report(&report.metrics);
    }
    let field_quality = FieldQualitySummary::from_analysis_and_matrices(&analysis, &field_matrices);

    match args.format.as_deref().unwrap_or("text") {
        "json" => print_json(&analysis, &field_matrices, &field_quality)?,
        "csv" => print_csv(&analysis, &field_matrices, &field_quality)?,
        "text" => print_text(&analysis, &field_matrices, &field_quality, args.verbose)?,
        _ => print_text(&analysis, &field_matrices, &field_quality, args.verbose)?,
    }

    Ok(())
}

#[derive(Clone, Debug, Default)]
struct FieldAccuracyMatrices {
    control: ConfusionMatrix,
    experimental: ConfusionMatrix,
}

impl FieldAccuracyMatrices {
    fn new() -> Self {
        Self::default()
    }

    fn add_report(&mut self, metrics: &Metrics) {
        add_field_counts(
            &mut self.control,
            metrics.baseline_fields_matched,
            metrics.baseline_fields_with_wrong_value,
            metrics.baseline_fields_missing,
            metrics.baseline_extra_fields,
        );
        add_field_counts(
            &mut self.experimental,
            metrics.policyai_fields_matched,
            metrics.policyai_fields_with_wrong_value,
            metrics.policyai_fields_missing,
            metrics.policyai_extra_fields,
        );
    }
}

fn add_field_counts(
    matrix: &mut ConfusionMatrix,
    matched: usize,
    wrong_value: usize,
    missing: usize,
    extra: usize,
) {
    matrix.true_positive += matched;
    // A wrong value misses the expected field/value and emits a field/value that was not expected.
    matrix.false_positive += wrong_value + extra;
    matrix.false_negative += wrong_value + missing;
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct FieldQualitySummary {
    policyai: FieldQualityTotals,
    baseline: FieldQualityTotals,
}

impl FieldQualitySummary {
    fn from_analysis_and_matrices(
        analysis: &RegressionAnalysis,
        field_matrices: &FieldAccuracyMatrices,
    ) -> Self {
        Self {
            policyai: FieldQualityTotals {
                matched_field_values: field_matrices.experimental.true_positive,
                not_expected_not_output_field_values: field_matrices.experimental.true_negative,
                wrong_values: analysis.policyai_total_wrong_values,
                missing_field_values: field_matrices.experimental.false_negative,
                extra_field_values: field_matrices.experimental.false_positive,
            },
            baseline: FieldQualityTotals {
                matched_field_values: field_matrices.control.true_positive,
                not_expected_not_output_field_values: field_matrices.control.true_negative,
                wrong_values: analysis.baseline_total_wrong_values,
                missing_field_values: field_matrices.control.false_negative,
                extra_field_values: field_matrices.control.false_positive,
            },
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct FieldQualityTotals {
    matched_field_values: usize,
    not_expected_not_output_field_values: usize,
    wrong_values: usize,
    missing_field_values: usize,
    extra_field_values: usize,
}

fn print_json(
    analysis: &RegressionAnalysis,
    field_matrices: &FieldAccuracyMatrices,
    field_quality: &FieldQualitySummary,
) -> Result<(), Box<dyn std::error::Error>> {
    let output = serde_json::json!({
        "summary": {
            "total_reports": analysis.total_reports,
            "policyai": {
                "avg_fields_matched": analysis.policyai_avg_fields_matched(),
                "total_matched_field_values": field_quality.policyai.matched_field_values,
                "total_wrong_values": field_quality.policyai.wrong_values,
                "total_missing_fields": field_quality.policyai.missing_field_values,
                "total_missing_field_values": field_quality.policyai.missing_field_values,
                "total_extra_fields": field_quality.policyai.extra_field_values,
                "total_extra_field_values": field_quality.policyai.extra_field_values,
                "total_not_expected_not_output_field_values": field_quality.policyai.not_expected_not_output_field_values,
                "error_rate": analysis.policyai_error_rate(),
                "avg_duration_ms": analysis.policyai_avg_duration_ms(),
            },
            "baseline": {
                "avg_fields_matched": analysis.baseline_avg_fields_matched(),
                "total_matched_field_values": field_quality.baseline.matched_field_values,
                "total_wrong_values": field_quality.baseline.wrong_values,
                "total_missing_fields": field_quality.baseline.missing_field_values,
                "total_missing_field_values": field_quality.baseline.missing_field_values,
                "total_extra_fields": field_quality.baseline.extra_field_values,
                "total_extra_field_values": field_quality.baseline.extra_field_values,
                "total_not_expected_not_output_field_values": field_quality.baseline.not_expected_not_output_field_values,
                "error_rate": analysis.baseline_error_rate(),
                "avg_duration_ms": analysis.baseline_avg_duration_ms(),
            },
            "comparison": {
                "fields_matched_improvement": analysis.policyai_avg_fields_matched() - analysis.baseline_avg_fields_matched(),
                "speed_ratio": if analysis.policyai_avg_duration_ms() > 0.0 {
                    analysis.baseline_avg_duration_ms() / analysis.policyai_avg_duration_ms()
                } else {
                    0.0
                },
                "error_rate_difference": analysis.policyai_error_rate() - analysis.baseline_error_rate(),
            },
            "field_match_accuracy": {
                "expected_vs_control": field_accuracy_json(&field_matrices.control, "baseline"),
                "expected_vs_experimental": field_accuracy_json(&field_matrices.experimental, "output"),
            }
        }
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn field_accuracy_json(matrix: &ConfusionMatrix, source: &str) -> serde_json::Value {
    serde_json::json!({
        "source": source,
        "confusion_matrix": {
            "true_positive": matrix.true_positive,
            "false_positive": matrix.false_positive,
            "true_negative": matrix.true_negative,
            "false_negative": matrix.false_negative,
        },
        "metrics": {
            "precision": matrix.precision(),
            "recall": matrix.recall(),
            "f1_score": matrix.f1_score(),
            "accuracy": matrix.accuracy(),
        }
    })
}

fn print_csv(
    analysis: &RegressionAnalysis,
    field_matrices: &FieldAccuracyMatrices,
    field_quality: &FieldQualitySummary,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("metric,policyai_total,baseline_total,policyai_avg,baseline_avg,improvement");
    println!(
        "fields_matched,{},{},{:.4},{:.4},{:.4}",
        field_quality.policyai.matched_field_values,
        field_quality.baseline.matched_field_values,
        analysis.policyai_avg_fields_matched(),
        analysis.baseline_avg_fields_matched(),
        analysis.policyai_avg_fields_matched() - analysis.baseline_avg_fields_matched()
    );
    println!(
        "missing_field_values,{},{},,,",
        field_quality.policyai.missing_field_values, field_quality.baseline.missing_field_values
    );
    println!(
        "extra_field_values,{},{},,,",
        field_quality.policyai.extra_field_values, field_quality.baseline.extra_field_values
    );
    println!(
        "not_expected_not_output_field_values,{},{},,,",
        field_quality.policyai.not_expected_not_output_field_values,
        field_quality.baseline.not_expected_not_output_field_values
    );
    println!(
        "wrong_values,{},{},,,",
        field_quality.policyai.wrong_values, field_quality.baseline.wrong_values
    );
    println!(
        "errors,{},{},{:.4},{:.4},{:.4}",
        analysis.policyai_errors,
        analysis.baseline_errors,
        analysis.policyai_error_rate(),
        analysis.baseline_error_rate(),
        analysis.policyai_error_rate() - analysis.baseline_error_rate()
    );
    println!(
        "duration_ms,{},{},{:.2},{:.2},{:.2}",
        analysis.policyai_total_duration_ms,
        analysis.baseline_total_duration_ms,
        analysis.policyai_avg_duration_ms(),
        analysis.baseline_avg_duration_ms(),
        if analysis.policyai_avg_duration_ms() > 0.0 {
            analysis.baseline_avg_duration_ms() / analysis.policyai_avg_duration_ms()
        } else {
            0.0
        }
    );

    print_confusion_matrix_csv(
        "field_match_accuracy_expected_vs_control_matrix",
        &field_matrices.control,
    );
    print_confusion_metrics_csv(
        "field_match_accuracy_expected_vs_control_metrics",
        &field_matrices.control,
    );
    print_confusion_matrix_csv(
        "field_match_accuracy_expected_vs_experimental_matrix",
        &field_matrices.experimental,
    );
    print_confusion_metrics_csv(
        "field_match_accuracy_expected_vs_experimental_metrics",
        &field_matrices.experimental,
    );

    Ok(())
}

fn print_confusion_matrix_csv(name: &str, matrix: &ConfusionMatrix) {
    println!("\n{name},value");
    println!("true_positive,{}", matrix.true_positive);
    println!("false_positive,{}", matrix.false_positive);
    println!("true_negative,{}", matrix.true_negative);
    println!("false_negative,{}", matrix.false_negative);
}

fn print_confusion_metrics_csv(name: &str, matrix: &ConfusionMatrix) {
    println!("\n{name},value");
    println!("precision,{:.4}", matrix.precision());
    println!("recall,{:.4}", matrix.recall());
    println!("f1_score,{:.4}", matrix.f1_score());
    println!("accuracy,{:.4}", matrix.accuracy());
}

fn print_text(
    analysis: &RegressionAnalysis,
    field_matrices: &FieldAccuracyMatrices,
    field_quality: &FieldQualitySummary,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("PolicyAI Regression Analysis Report");
    println!("===================================");
    println!("Total evaluation reports: {}", analysis.total_reports);
    println!();

    println!("Performance Comparison:");
    println!("----------------------");

    println!("Fields Matched:");
    println!(
        "  PolicyAI avg: {:.2}",
        analysis.policyai_avg_fields_matched()
    );
    println!(
        "  Baseline avg: {:.2}",
        analysis.baseline_avg_fields_matched()
    );
    println!(
        "  Improvement:  {:.2}",
        analysis.policyai_avg_fields_matched() - analysis.baseline_avg_fields_matched()
    );
    println!();

    println!("Error Rates:");
    println!(
        "  PolicyAI: {:.1}% ({} errors)",
        analysis.policyai_error_rate() * 100.0,
        analysis.policyai_errors
    );
    println!(
        "  Baseline: {:.1}% ({} errors)",
        analysis.baseline_error_rate() * 100.0,
        analysis.baseline_errors
    );
    println!(
        "  Difference: {:.1} percentage points",
        (analysis.policyai_error_rate() - analysis.baseline_error_rate()) * 100.0
    );
    println!();

    println!("Performance:");
    println!(
        "  PolicyAI avg duration: {:.2} ms",
        analysis.policyai_avg_duration_ms()
    );
    println!(
        "  Baseline avg duration: {:.2} ms",
        analysis.baseline_avg_duration_ms()
    );
    if analysis.policyai_avg_duration_ms() > 0.0 {
        let speed_ratio = analysis.baseline_avg_duration_ms() / analysis.policyai_avg_duration_ms();
        println!("  Speed ratio (baseline/policyai): {:.2}x", speed_ratio);
    }
    println!();

    println!("Field Quality:");
    println!("Matched Field/Values:");
    println!(
        "  PolicyAI total: {}",
        field_quality.policyai.matched_field_values
    );
    println!(
        "  Baseline total: {}",
        field_quality.baseline.matched_field_values
    );
    println!();

    println!("Missing Expected Field/Values:");
    println!(
        "  PolicyAI total: {}",
        field_quality.policyai.missing_field_values
    );
    println!(
        "  Baseline total: {}",
        field_quality.baseline.missing_field_values
    );
    println!();

    println!("Unexpected Field/Values:");
    println!(
        "  PolicyAI total: {}",
        field_quality.policyai.extra_field_values
    );
    println!(
        "  Baseline total: {}",
        field_quality.baseline.extra_field_values
    );
    println!();

    println!("No Expected/No Output Field/Values:");
    println!(
        "  PolicyAI total: {}",
        field_quality.policyai.not_expected_not_output_field_values
    );
    println!(
        "  Baseline total: {}",
        field_quality.baseline.not_expected_not_output_field_values
    );
    println!();

    println!("Value Mismatches:");
    println!("  PolicyAI total: {}", field_quality.policyai.wrong_values);
    println!("  Baseline total: {}", field_quality.baseline.wrong_values);
    println!(
        "  Wrong values are included in both missing expected and unexpected field/value totals."
    );
    println!();

    print_confusion_matrix_text(
        "Field Match Accuracy (Expected Output vs Control/Baseline)",
        "Control (baseline)",
        &field_matrices.control,
    );
    print_confusion_matrix_text(
        "Field Match Accuracy (Expected Output vs Experimental/PolicyAI)",
        "Experimental (PolicyAI)",
        &field_matrices.experimental,
    );

    if verbose {
        println!("Additional Details:");
        println!("------------------");
        println!("Total Duration:");
        println!(
            "  PolicyAI total: {} ms",
            analysis.policyai_total_duration_ms
        );
        println!(
            "  Baseline total: {} ms",
            analysis.baseline_total_duration_ms
        );
        println!();
    }

    Ok(())
}

fn print_confusion_matrix_text(name: &str, output_label: &str, matrix: &ConfusionMatrix) {
    println!("{}:", name);

    let tp = matrix.true_positive;
    let fp = matrix.false_positive;
    let tn = matrix.true_negative;
    let fn_val = matrix.false_negative;
    let total = tp + fp + tn + fn_val;

    let tp_cell = format_count_with_percentage(tp, total);
    let fn_cell = format_count_with_percentage(fn_val, total);
    let fp_cell = format_count_with_percentage(fp, total);
    let tn_cell = format_count_with_percentage(tn, total);
    let col_width = [
        "Output field/value".len(),
        "No output value".len(),
        tp_cell.len(),
        fn_cell.len(),
        fp_cell.len(),
        tn_cell.len(),
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    let row_width = "Expected field/value".len();

    println!("  Confusion Matrix:");
    println!("    {:>row_width$} │ {output_label}", "");
    println!(
        "    {:>row_width$} │ {:>col_width$} {:>col_width$}",
        "", "Output field/value", "No output value",
    );
    println!(
        "    {:─>row_width$}─┼─{:─<col_width$}─{:─<col_width$}",
        "", "", "",
    );
    println!(
        "    {:>row_width$} │ {:>col_width$} {:>col_width$}",
        "Expected field/value", tp_cell, fn_cell,
    );
    println!(
        "    {:>row_width$} │ {:>col_width$} {:>col_width$}",
        "Not expected", fp_cell, tn_cell,
    );
    println!();

    println!("  Metrics:");
    println!(
        "    Precision: {:.4} (when {output_label} outputs a field/value, how often is it expected)",
        matrix.precision()
    );
    println!(
        "    Recall:    {:.4} (of expected field/values, how often does {output_label} output them)",
        matrix.recall()
    );
    println!("    F1 Score:  {:.4}", matrix.f1_score());
    println!(
        "    Accuracy:  {:.4} (observed agreement rate)",
        matrix.accuracy()
    );

    println!();
}

fn format_count_with_percentage(count: usize, total: usize) -> String {
    let percentage = if total == 0 {
        0.0
    } else {
        100.0 * count as f64 / total as f64
    };
    format!("{count} ({percentage:.1}%)")
}

fn read_from_stdin() -> Result<Vec<EvaluationReport>, Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    let reports: Vec<EvaluationReport> = input
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(reports)
}

fn read_from_files(files: &[String]) -> Result<Vec<EvaluationReport>, Box<dyn std::error::Error>> {
    let mut reports = Vec::new();

    for file_path in files {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let report: EvaluationReport = match serde_json::from_str(&line) {
                Ok(report) => report,
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to parse line in {file_path} as EvaluationReport: {e}"
                    );
                    continue;
                }
            };

            reports.push(report);
        }
    }

    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_accuracy_matrices_compare_expected_to_control_and_experimental() {
        let metrics = Metrics {
            policyai_fields_matched: 2,
            policyai_fields_with_wrong_value: 3,
            policyai_fields_missing: 5,
            policyai_extra_fields: 7,
            baseline_fields_matched: 11,
            baseline_fields_with_wrong_value: 13,
            baseline_fields_missing: 17,
            baseline_extra_fields: 19,
            ..Default::default()
        };

        let mut matrices = FieldAccuracyMatrices::new();
        matrices.add_report(&metrics);

        assert_eq!(
            matrices.control,
            ConfusionMatrix {
                true_positive: 11,
                false_positive: 32,
                true_negative: 0,
                false_negative: 30,
            }
        );
        assert_eq!(
            matrices.experimental,
            ConfusionMatrix {
                true_positive: 2,
                false_positive: 10,
                true_negative: 0,
                false_negative: 8,
            }
        );
    }

    #[test]
    fn field_quality_summary_uses_confusion_matrix_counts() {
        let metrics = Metrics {
            policyai_fields_matched: 761,
            policyai_fields_with_wrong_value: 8,
            policyai_fields_missing: 24,
            policyai_extra_fields: 23,
            baseline_fields_matched: 612,
            baseline_fields_with_wrong_value: 2,
            baseline_fields_missing: 56,
            baseline_extra_fields: 0,
            ..Default::default()
        };

        let mut analysis = RegressionAnalysis::new();
        analysis.add_report(&metrics);
        let mut matrices = FieldAccuracyMatrices::new();
        matrices.add_report(&metrics);

        assert_eq!(
            FieldQualitySummary::from_analysis_and_matrices(&analysis, &matrices),
            FieldQualitySummary {
                policyai: FieldQualityTotals {
                    matched_field_values: 761,
                    not_expected_not_output_field_values: 0,
                    wrong_values: 8,
                    missing_field_values: 32,
                    extra_field_values: 31,
                },
                baseline: FieldQualityTotals {
                    matched_field_values: 612,
                    not_expected_not_output_field_values: 0,
                    wrong_values: 2,
                    missing_field_values: 58,
                    extra_field_values: 2,
                },
            }
        );
    }
}
