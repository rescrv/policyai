//! Generate token usage reports from PolicyAI evaluation data.
//!
//! This binary reads evaluation reports and generates comprehensive token usage analysis
//! including min, max, average, p50, and p99 statistics for PolicyAI and baseline systems.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};

use arrrg::CommandLine;
use policyai::analysis::TokenUsageAnalysis;
use policyai::data::EvaluationReport;

#[derive(Clone, Default, Debug, Eq, PartialEq, arrrg_derive::CommandLine)]
struct Args {
    #[arrrg(flag, "Print detailed metrics for each token type")]
    verbose: bool,
    #[arrrg(optional, "Output format (json, csv, text)")]
    format: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (args, free) = Args::from_command_line_relaxed(
        "USAGE: policyai-token-usage-report [OPTIONS] [input_file...]",
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

    let mut analysis = TokenUsageAnalysis::new();

    for report in &reports {
        analysis.add_report(&report.metrics);
    }

    match args.format.as_deref().unwrap_or("text") {
        "json" => print_json(&analysis)?,
        "csv" => print_csv(&analysis)?,
        "text" => print_text(&analysis, args.verbose)?,
        _ => print_text(&analysis, args.verbose)?,
    }

    Ok(())
}

fn print_json(analysis: &TokenUsageAnalysis) -> Result<(), Box<dyn std::error::Error>> {
    let output = serde_json::json!({
        "summary": {
            "total_reports": analysis.total_reports,
            "policyai": {
                "input_tokens": {
                    "total": analysis.policyai_total_input_tokens(),
                    "avg": analysis.policyai_avg_input_tokens(),
                    "min": analysis.policyai_min_input_tokens(),
                    "max": analysis.policyai_max_input_tokens(),
                    "p50": analysis.policyai_p50_input_tokens(),
                    "p99": analysis.policyai_p99_input_tokens(),
                },
                "output_tokens": {
                    "total": analysis.policyai_total_output_tokens(),
                    "avg": analysis.policyai_avg_output_tokens(),
                    "min": analysis.policyai_min_output_tokens(),
                    "max": analysis.policyai_max_output_tokens(),
                    "p50": analysis.policyai_p50_output_tokens(),
                    "p99": analysis.policyai_p99_output_tokens(),
                },
                "cache_creation_tokens": {
                    "total": analysis.policyai_total_cache_creation_tokens(),
                    "avg": analysis.policyai_avg_cache_creation_tokens(),
                    "p99": analysis.policyai_p99_cache_creation_tokens(),
                },
                "cache_read_tokens": {
                    "total": analysis.policyai_total_cache_read_tokens(),
                    "avg": analysis.policyai_avg_cache_read_tokens(),
                    "p99": analysis.policyai_p99_cache_read_tokens(),
                },
                "wall_clock_ms": {
                    "avg": analysis.policyai_avg_wall_clock_ms(),
                    "p50": analysis.policyai_p50_wall_clock_ms(),
                    "p99": analysis.policyai_p99_wall_clock_ms(),
                },
            },
            "baseline": {
                "input_tokens": {
                    "total": analysis.baseline_total_input_tokens(),
                    "avg": analysis.baseline_avg_input_tokens(),
                    "min": analysis.baseline_min_input_tokens(),
                    "max": analysis.baseline_max_input_tokens(),
                    "p50": analysis.baseline_p50_input_tokens(),
                    "p99": analysis.baseline_p99_input_tokens(),
                },
                "output_tokens": {
                    "total": analysis.baseline_total_output_tokens(),
                    "avg": analysis.baseline_avg_output_tokens(),
                    "min": analysis.baseline_min_output_tokens(),
                    "max": analysis.baseline_max_output_tokens(),
                    "p50": analysis.baseline_p50_output_tokens(),
                    "p99": analysis.baseline_p99_output_tokens(),
                },
                "cache_creation_tokens": {
                    "total": analysis.baseline_total_cache_creation_tokens(),
                    "avg": analysis.baseline_avg_cache_creation_tokens(),
                    "p99": analysis.baseline_p99_cache_creation_tokens(),
                },
                "cache_read_tokens": {
                    "total": analysis.baseline_total_cache_read_tokens(),
                    "avg": analysis.baseline_avg_cache_read_tokens(),
                    "p99": analysis.baseline_p99_cache_read_tokens(),
                },
                "wall_clock_ms": {
                    "avg": analysis.baseline_avg_wall_clock_ms(),
                    "p50": analysis.baseline_p50_wall_clock_ms(),
                    "p99": analysis.baseline_p99_wall_clock_ms(),
                },
            },
        }
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn print_csv(analysis: &TokenUsageAnalysis) -> Result<(), Box<dyn std::error::Error>> {
    println!("metric,statistic,policyai,baseline");
    println!(
        "input_tokens,total,{},{}",
        analysis.policyai_total_input_tokens(),
        analysis.baseline_total_input_tokens()
    );
    println!(
        "input_tokens,avg,{:.2},{:.2}",
        analysis.policyai_avg_input_tokens(),
        analysis.baseline_avg_input_tokens()
    );
    println!(
        "input_tokens,min,{},{}",
        analysis.policyai_min_input_tokens(),
        analysis.baseline_min_input_tokens()
    );
    println!(
        "input_tokens,max,{},{}",
        analysis.policyai_max_input_tokens(),
        analysis.baseline_max_input_tokens()
    );
    println!(
        "input_tokens,p50,{},{}",
        analysis.policyai_p50_input_tokens(),
        analysis.baseline_p50_input_tokens()
    );
    println!(
        "input_tokens,p99,{},{}",
        analysis.policyai_p99_input_tokens(),
        analysis.baseline_p99_input_tokens()
    );

    println!(
        "output_tokens,total,{},{}",
        analysis.policyai_total_output_tokens(),
        analysis.baseline_total_output_tokens()
    );
    println!(
        "output_tokens,avg,{:.2},{:.2}",
        analysis.policyai_avg_output_tokens(),
        analysis.baseline_avg_output_tokens()
    );
    println!(
        "output_tokens,min,{},{}",
        analysis.policyai_min_output_tokens(),
        analysis.baseline_min_output_tokens()
    );
    println!(
        "output_tokens,max,{},{}",
        analysis.policyai_max_output_tokens(),
        analysis.baseline_max_output_tokens()
    );
    println!(
        "output_tokens,p50,{},{}",
        analysis.policyai_p50_output_tokens(),
        analysis.baseline_p50_output_tokens()
    );
    println!(
        "output_tokens,p99,{},{}",
        analysis.policyai_p99_output_tokens(),
        analysis.baseline_p99_output_tokens()
    );

    println!(
        "cache_creation_tokens,total,{},{}",
        analysis.policyai_total_cache_creation_tokens(),
        analysis.baseline_total_cache_creation_tokens()
    );
    println!(
        "cache_creation_tokens,avg,{:.2},{:.2}",
        analysis.policyai_avg_cache_creation_tokens(),
        analysis.baseline_avg_cache_creation_tokens()
    );
    println!(
        "cache_creation_tokens,p99,{},{}",
        analysis.policyai_p99_cache_creation_tokens(),
        analysis.baseline_p99_cache_creation_tokens()
    );

    println!(
        "cache_read_tokens,total,{},{}",
        analysis.policyai_total_cache_read_tokens(),
        analysis.baseline_total_cache_read_tokens()
    );
    println!(
        "cache_read_tokens,avg,{:.2},{:.2}",
        analysis.policyai_avg_cache_read_tokens(),
        analysis.baseline_avg_cache_read_tokens()
    );
    println!(
        "cache_read_tokens,p99,{},{}",
        analysis.policyai_p99_cache_read_tokens(),
        analysis.baseline_p99_cache_read_tokens()
    );

    println!(
        "wall_clock_ms,avg,{:.2},{:.2}",
        analysis.policyai_avg_wall_clock_ms(),
        analysis.baseline_avg_wall_clock_ms()
    );
    println!(
        "wall_clock_ms,p50,{},{}",
        analysis.policyai_p50_wall_clock_ms(),
        analysis.baseline_p50_wall_clock_ms()
    );
    println!(
        "wall_clock_ms,p99,{},{}",
        analysis.policyai_p99_wall_clock_ms(),
        analysis.baseline_p99_wall_clock_ms()
    );

    Ok(())
}

fn print_text(
    analysis: &TokenUsageAnalysis,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("PolicyAI Token Usage Report");
    println!("===========================");
    println!("Total evaluation reports: {}", analysis.total_reports);
    println!();

    println!("Input Tokens:");
    println!("-------------");
    println!("  PolicyAI:");
    println!("    Total: {}", analysis.policyai_total_input_tokens());
    println!("    Avg:   {:.2}", analysis.policyai_avg_input_tokens());
    println!("    Min:   {}", analysis.policyai_min_input_tokens());
    println!("    Max:   {}", analysis.policyai_max_input_tokens());
    println!("    P50:   {}", analysis.policyai_p50_input_tokens());
    println!("    P99:   {}", analysis.policyai_p99_input_tokens());
    println!();
    println!("  Baseline:");
    println!("    Total: {}", analysis.baseline_total_input_tokens());
    println!("    Avg:   {:.2}", analysis.baseline_avg_input_tokens());
    println!("    Min:   {}", analysis.baseline_min_input_tokens());
    println!("    Max:   {}", analysis.baseline_max_input_tokens());
    println!("    P50:   {}", analysis.baseline_p50_input_tokens());
    println!("    P99:   {}", analysis.baseline_p99_input_tokens());
    println!();

    println!("Output Tokens:");
    println!("--------------");
    println!("  PolicyAI:");
    println!("    Total: {}", analysis.policyai_total_output_tokens());
    println!("    Avg:   {:.2}", analysis.policyai_avg_output_tokens());
    println!("    Min:   {}", analysis.policyai_min_output_tokens());
    println!("    Max:   {}", analysis.policyai_max_output_tokens());
    println!("    P50:   {}", analysis.policyai_p50_output_tokens());
    println!("    P99:   {}", analysis.policyai_p99_output_tokens());
    println!();
    println!("  Baseline:");
    println!("    Total: {}", analysis.baseline_total_output_tokens());
    println!("    Avg:   {:.2}", analysis.baseline_avg_output_tokens());
    println!("    Min:   {}", analysis.baseline_min_output_tokens());
    println!("    Max:   {}", analysis.baseline_max_output_tokens());
    println!("    P50:   {}", analysis.baseline_p50_output_tokens());
    println!("    P99:   {}", analysis.baseline_p99_output_tokens());
    println!();

    if verbose
        || analysis.policyai_total_cache_creation_tokens() > 0
        || analysis.baseline_total_cache_creation_tokens() > 0
    {
        println!("Cache Creation Tokens:");
        println!("----------------------");
        println!("  PolicyAI:");
        println!(
            "    Total: {}",
            analysis.policyai_total_cache_creation_tokens()
        );
        println!(
            "    Avg:   {:.2}",
            analysis.policyai_avg_cache_creation_tokens()
        );
        println!(
            "    P99:   {}",
            analysis.policyai_p99_cache_creation_tokens()
        );
        println!();
        println!("  Baseline:");
        println!(
            "    Total: {}",
            analysis.baseline_total_cache_creation_tokens()
        );
        println!(
            "    Avg:   {:.2}",
            analysis.baseline_avg_cache_creation_tokens()
        );
        println!(
            "    P99:   {}",
            analysis.baseline_p99_cache_creation_tokens()
        );
        println!();

        println!("Cache Read Tokens:");
        println!("------------------");
        println!("  PolicyAI:");
        println!("    Total: {}", analysis.policyai_total_cache_read_tokens());
        println!(
            "    Avg:   {:.2}",
            analysis.policyai_avg_cache_read_tokens()
        );
        println!("    P99:   {}", analysis.policyai_p99_cache_read_tokens());
        println!();
        println!("  Baseline:");
        println!("    Total: {}", analysis.baseline_total_cache_read_tokens());
        println!(
            "    Avg:   {:.2}",
            analysis.baseline_avg_cache_read_tokens()
        );
        println!("    P99:   {}", analysis.baseline_p99_cache_read_tokens());
        println!();
    }

    if verbose {
        println!("Wall Clock Time:");
        println!("----------------");
        println!("  PolicyAI:");
        println!("    Avg: {:.2} ms", analysis.policyai_avg_wall_clock_ms());
        println!("    P50: {} ms", analysis.policyai_p50_wall_clock_ms());
        println!("    P99: {} ms", analysis.policyai_p99_wall_clock_ms());
        println!();
        println!("  Baseline:");
        println!("    Avg: {:.2} ms", analysis.baseline_avg_wall_clock_ms());
        println!("    P50: {} ms", analysis.baseline_p50_wall_clock_ms());
        println!("    P99: {} ms", analysis.baseline_p99_wall_clock_ms());
        println!();
    }

    Ok(())
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
