use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use arrrg::CommandLine;
use claudius::{Anthropic, Effort, Model};
use policyai::data::{EffortArg, ThinkingArg};
use rand::prelude::*;

#[derive(Clone, Default, Debug, Eq, PartialEq, arrrg_derive::CommandLine)]
struct Options {
    #[arrrg(required, "This many negative policies will be selected per text.")]
    policies: usize,
    #[arrrg(
        required,
        "The number of successful verifications required to select a policy."
    )]
    success: usize,
    #[arrrg(
        required,
        "The number of total verifications to perform for each policy."
    )]
    total: usize,
    #[arrrg(optional, "Anthropic model to use for verification.")]
    model: Option<String>,
    #[arrrg(optional, "Maximum output tokens for each verification request.")]
    max_tokens: Option<u32>,
    #[arrrg(optional, "Thinking config: adaptive, disabled, or a token budget.")]
    thinking: Option<ThinkingArg>,
    #[arrrg(optional, "Adaptive thinking effort: low, medium, or high.")]
    effort: Option<EffortArg>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (options, free) = Options::from_command_line_relaxed(
        "USAGE: policyai-generate-decidables [OPTIONS] SEMANTIC-INJECTIONS",
    );
    if free.len() != 1 {
        eprintln!("expected SEMANTIC-INJECTIONS");
        std::process::exit(13);
    }
    let model = options
        .model
        .as_deref()
        .map(|model| model.parse::<Model>().unwrap())
        .unwrap_or(policyai::DEFAULT_MODEL);
    let max_tokens = options.max_tokens.unwrap_or(1030);
    let thinking = Some(
        options
            .thinking
            .map(Into::into)
            .unwrap_or_else(claudius::ThinkingConfig::adaptive),
    );
    let effort = Some(options.effort.map(Into::into).unwrap_or(Effort::High));
    let client = Anthropic::new(None)
        .expect("could not connect to claude")
        .with_max_retries(10)
        .with_backoff_params(10.0, 1.0);
    let semantic_injections_file =
        BufReader::new(OpenOptions::new().read(true).open(&free[0]).unwrap());
    let mut semantic_injections = vec![];
    let mut policy_fragments = vec![];
    for line in semantic_injections_file.lines() {
        let line = line?;
        let injection: policyai::data::SemanticInjection = serde_json::from_str(&line)?;
        policy_fragments.extend(injection.injections.clone());
        semantic_injections.push(injection);
    }
    let mut rng = rand::rng();
    for (sample_number, injection) in semantic_injections.into_iter().enumerate() {
        eprintln!("done {sample_number} samples");
        let mut negatives: Vec<String> = vec![];
        while negatives.len() < options.policies {
            let policy_fragment = policy_fragments.choose(&mut rng).unwrap();
            if policyai::data::policy_does_not_apply(
                &client,
                &injection.text,
                policy_fragment,
                options.success,
                options.total,
                model.clone(),
                max_tokens,
                thinking,
                effort,
            )
            .await?
            {
                negatives.push(policy_fragment.clone());
            }
            eprintln!("generated {} negatives", negatives.len());
        }
        println!(
            "{}",
            serde_json::to_string(&policyai::data::DecidableSemanticInjection {
                positives: injection.injections.clone(),
                negatives,
                text: injection.text.clone(),
            })
            .unwrap()
        );
    }
    Ok(())
}
