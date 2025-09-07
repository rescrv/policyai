//! Generate regression analysis reports from PolicyAI evaluation data.
//!
//! This binary reads evaluation reports and generates comprehensive regression analysis
//! using confusion matrices and metrics to compare PolicyAI performance against baselines.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};

use arrrg::CommandLine;
use policyai::analysis::{ConfusionMatrix, FieldMatchAccuracyMatrix, RegressionAnalysis};
use policyai::data::EvaluationReport;

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
    let mut accuracy_matrix = FieldMatchAccuracyMatrix::new();

    for report in &reports {
        analysis.add_report(&report.metrics);

        let expected_field_count = report
            .input
            .expected
            .as_ref()
            .and_then(|v| v.as_object())
            .map(|obj| obj.len())
            .unwrap_or(0);

        accuracy_matrix.add_report(&report.metrics, expected_field_count);
    }

    match args.format.as_deref().unwrap_or("text") {
        "json" => print_json(&analysis, &accuracy_matrix, &reports)?,
        "csv" => print_csv(&analysis, &accuracy_matrix)?,
        "text" => print_text(&analysis, &accuracy_matrix, &reports, args.verbose)?,
        _ => print_text(&analysis, &accuracy_matrix, &reports, args.verbose)?,
    }

    Ok(())
}

fn print_json(
    analysis: &RegressionAnalysis,
    accuracy_matrix: &FieldMatchAccuracyMatrix,
    _reports: &[EvaluationReport],
) -> Result<(), Box<dyn std::error::Error>> {
    let output = serde_json::json!({
        "summary": {
            "total_reports": analysis.total_reports,
            "policyai": {
                "avg_fields_matched": analysis.policyai_avg_fields_matched(),
                "total_wrong_values": analysis.policyai_total_wrong_values,
                "total_missing_fields": analysis.policyai_total_missing_fields,
                "total_extra_fields": analysis.policyai_total_extra_fields,
                "error_rate": analysis.policyai_error_rate(),
                "avg_duration_ms": analysis.policyai_avg_duration_ms(),
            },
            "baseline": {
                "avg_fields_matched": analysis.baseline_avg_fields_matched(),
                "total_wrong_values": analysis.baseline_total_wrong_values,
                "total_missing_fields": analysis.baseline_total_missing_fields,
                "total_extra_fields": analysis.baseline_total_extra_fields,
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
                "confusion_matrix": {
                    "true_positive": accuracy_matrix.confusion_matrix.true_positive,
                    "false_positive": accuracy_matrix.confusion_matrix.false_positive,
                    "true_negative": accuracy_matrix.confusion_matrix.true_negative,
                    "false_negative": accuracy_matrix.confusion_matrix.false_negative,
                },
                "metrics": {
                    "precision": accuracy_matrix.confusion_matrix.precision(),
                    "recall": accuracy_matrix.confusion_matrix.recall(),
                    "f1_score": accuracy_matrix.confusion_matrix.f1_score(),
                    "accuracy": accuracy_matrix.confusion_matrix.accuracy(),
                }
            }
        }
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn print_csv(
    analysis: &RegressionAnalysis,
    accuracy_matrix: &FieldMatchAccuracyMatrix,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("metric,policyai_total,baseline_total,policyai_avg,baseline_avg,improvement");
    println!(
        "fields_matched,{},{},{:.4},{:.4},{:.4}",
        analysis.policyai_total_fields_matched,
        analysis.baseline_total_fields_matched,
        analysis.policyai_avg_fields_matched(),
        analysis.baseline_avg_fields_matched(),
        analysis.policyai_avg_fields_matched() - analysis.baseline_avg_fields_matched()
    );
    println!(
        "wrong_values,{},{},,,",
        analysis.policyai_total_wrong_values, analysis.baseline_total_wrong_values
    );
    println!(
        "missing_fields,{},{},,,",
        analysis.policyai_total_missing_fields, analysis.baseline_total_missing_fields
    );
    println!(
        "extra_fields,{},{},,,",
        analysis.policyai_total_extra_fields, analysis.baseline_total_extra_fields
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

    println!("\nfield_match_accuracy_matrix,value");
    println!(
        "true_positive,{}",
        accuracy_matrix.confusion_matrix.true_positive
    );
    println!(
        "false_positive,{}",
        accuracy_matrix.confusion_matrix.false_positive
    );
    println!(
        "true_negative,{}",
        accuracy_matrix.confusion_matrix.true_negative
    );
    println!(
        "false_negative,{}",
        accuracy_matrix.confusion_matrix.false_negative
    );
    println!(
        "precision,{:.4}",
        accuracy_matrix.confusion_matrix.precision()
    );
    println!("recall,{:.4}", accuracy_matrix.confusion_matrix.recall());
    println!(
        "f1_score,{:.4}",
        accuracy_matrix.confusion_matrix.f1_score()
    );
    println!(
        "accuracy,{:.4}",
        accuracy_matrix.confusion_matrix.accuracy()
    );

    Ok(())
}

fn print_text(
    analysis: &RegressionAnalysis,
    accuracy_matrix: &FieldMatchAccuracyMatrix,
    _reports: &[EvaluationReport],
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
    println!("Wrong Values:");
    println!("  PolicyAI total: {}", analysis.policyai_total_wrong_values);
    println!("  Baseline total: {}", analysis.baseline_total_wrong_values);
    println!();

    println!("Missing Fields:");
    println!(
        "  PolicyAI total: {}",
        analysis.policyai_total_missing_fields
    );
    println!(
        "  Baseline total: {}",
        analysis.baseline_total_missing_fields
    );
    println!();

    println!("Extra Fields:");
    println!("  PolicyAI total: {}", analysis.policyai_total_extra_fields);
    println!("  Baseline total: {}", analysis.baseline_total_extra_fields);
    println!();

    // Display confusion matrix for field matching accuracy
    print_confusion_matrix_text(
        "Field Match Accuracy (PolicyAI vs Baseline)",
        &accuracy_matrix.confusion_matrix,
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

fn print_confusion_matrix_text(name: &str, matrix: &ConfusionMatrix) {
    println!("{}:", name);

    // Print confusion matrix in tabular format
    let tp = matrix.true_positive;
    let fp = matrix.false_positive;
    let tn = matrix.true_negative;
    let fn_val = matrix.false_negative;

    // Calculate column widths for alignment
    let values = [tp, fp, tn, fn_val];
    let max_val = values.iter().max().unwrap();
    let val_width = format!("{}", max_val).len().max(4);

    println!("  Confusion Matrix:");
    println!("                     │ PolicyAI");
    println!(
        "                     │ {:>width$} {:>width$}",
        "Correct",
        "Wrong",
        width = val_width + 8
    );
    println!(
        "    ─────────────────┼{:─<width$}─{:─<width$}──",
        "",
        "",
        width = val_width + 8
    );
    let total = tp + fp + tn + fn_val;
    println!(
        "    Baseline Correct │ {:>width$} {:>width$}",
        format!("{} ({:.1}%)", tp, 100.0 * tp as f64 / total as f64),
        format!("{} ({:.1}%)", fn_val, 100.0 * fn_val as f64 / total as f64),
        width = val_width + 8 // Add space for percentage
    );
    println!(
        "               Wrong │ {:>width$} {:>width$}",
        format!("{} ({:.1}%)", fp, 100.0 * fp as f64 / total as f64),
        format!("{} ({:.1}%)", tn, 100.0 * tn as f64 / total as f64),
        width = val_width + 8
    );
    println!();

    // Print metrics
    println!("  Metrics:");
    println!(
        "    Precision: {:.4} (when PolicyAI says correct, how often is it right)",
        matrix.precision()
    );
    println!(
        "    Recall:    {:.4} (when baseline is correct, how often does PolicyAI get it right)",
        matrix.recall()
    );
    println!("    F1 Score:  {:.4}", matrix.f1_score());
    println!(
        "    Accuracy:  {:.4} (overall agreement rate)",
        matrix.accuracy()
    );

    println!();
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
