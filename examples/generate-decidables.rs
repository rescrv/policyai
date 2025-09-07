use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use arrrg::CommandLine;
use claudius::Anthropic;
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
}

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    let (options, free) = Options::from_command_line_relaxed(
        "USAGE: policyai-generate-decidables [OPTIONS] SEMANTIC-INJECTIONS",
    );
    if free.len() != 1 {
        eprintln!("expected SEMANTIC-INJECTIONS");
        std::process::exit(13);
    }
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
            )
            .await
            .unwrap()
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
