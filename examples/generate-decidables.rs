use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use arrrg::CommandLine;
use rand::prelude::*;

#[derive(Clone, Default, Debug, Eq, PartialEq, arrrg_derive::CommandLine)]
struct Options {
    #[arrrg(optional, "The ollama host to connect to.")]
    host: Option<String>,
    #[arrrg(required, "This many text will be selected to have policies applied.")]
    samples: usize,
    #[arrrg(required, "This many policies will be selected per text.")]
    policies: usize,
    #[arrrg(required, "The model to use for generating policies.")]
    model: String,
    #[arrrg(nested)]
    param: yammer::Parameters,
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
    for sample_number in 0..options.samples {
        eprintln!("done {} samples", sample_number);
        let injection = semantic_injections.choose(&mut rng).unwrap();
        let mut negatives: Vec<String> = vec![];
        while negatives.len() < options.policies {
            eprintln!("generated {} negatives", negatives.len());
            let policy_fragment = policy_fragments.choose(&mut rng).unwrap();
            let mut successes = 0;
            let mut failures = 0;
            while successes < options.success && successes + failures < options.total {
                let negative_applies = !policyai::data::policy_applies(
                    None,
                    yammer::GenerateRequest {
                        model: options.model.to_string(),
                        prompt: "".to_string(),
                        format: None,
                        images: None,
                        keep_alive: None,
                        suffix: None,
                        system: None,
                        template: None,
                        stream: Some(false),
                        raw: None,
                        options: Some(options.param.clone().into()),
                    },
                    &injection.text,
                    policy_fragment,
                    options.success,
                    options.total,
                )
                .await
                .unwrap();
                if negative_applies {
                    successes += 1;
                } else {
                    failures += 1;
                }
                eprintln!("done {} negative tests", successes + failures);
            }
            if successes >= options.success {
                negatives.push(policy_fragment.clone());
            }
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
