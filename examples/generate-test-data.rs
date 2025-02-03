use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use arrrg::CommandLine;
use rand::prelude::*;

use policyai::data::InjectableAction;
use policyai::{Policy, PolicyType};

#[derive(Clone, Default, Debug, Eq, PartialEq, arrrg_derive::CommandLine)]
struct Options {
    #[arrrg(required, "The decidable semantic injections.")]
    decidables: String,
    #[arrrg(required, "The actions.")]
    actions: String,
    #[arrrg(
        required,
        "This many tweets will be selected to have policies applied."
    )]
    samples: usize,
    #[arrrg(required, "The policy type definition.")]
    policy: String,
    #[arrrg(required, "This many policies will be selected per tweet.")]
    policies: usize,
    #[arrrg(required, "This many policies will be enforced to match per tweet.")]
    matching: usize,
}

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    let (options, free) =
        Options::from_command_line_relaxed("USAGE: policyai-generate-test-data [OPTIONS]");
    if !free.is_empty() {
        eprintln!("command takes no positional arguments");
        std::process::exit(13);
    }
    let semantic_injections_file = BufReader::new(
        OpenOptions::new()
            .read(true)
            .open(&options.decidables)
            .unwrap(),
    );
    let mut semantic_injections = vec![];
    for line in semantic_injections_file.lines() {
        let line = line?;
        let injection: policyai::data::DecidableSemanticInjection = serde_json::from_str(&line)?;
        semantic_injections.push(injection);
    }
    let actions_file = BufReader::new(
        OpenOptions::new()
            .read(true)
            .open(&options.actions)
            .unwrap(),
    );
    let mut actions = vec![];
    for line in actions_file.lines() {
        let line = line?;
        let action: policyai::data::InjectableAction = serde_json::from_str(&line)?;
        actions.push(action);
    }
    let policy_type = PolicyType::parse(&std::fs::read_to_string(options.policy).unwrap()).unwrap();
    let mut rng = rand::rng();
    for _ in 0..options.samples {
        let injection = semantic_injections.choose(&mut rng).unwrap();
        assert!(injection.positives.len() >= options.matching);
        assert!(injection.negatives.len() >= options.policies - options.matching);
        let mut policies = vec![];
        let mut expected = serde_json::json! {{}};
        while policies.len() < options.policies {
            let (prompt, action) = if policies.len() < options.matching {
                let Some(prompt) = injection.positives.choose(&mut rng) else {
                    continue;
                };
                let mut prompt = prompt.clone();
                fn is_compatible_action(
                    output: &serde_json::Value,
                    action: &InjectableAction,
                ) -> bool {
                    let serde_json::Value::Object(obj) = &action.action else {
                        return false;
                    };
                    for (k, v) in obj.iter() {
                        let see = output.get(k);
                        if see.is_some() && see != Some(v) {
                            return false;
                        }
                    }
                    true
                }
                fn pick_compatible_action<'a>(
                    rng: &mut ThreadRng,
                    output: &serde_json::Value,
                    actions: &'a [InjectableAction],
                ) -> &'a InjectableAction {
                    loop {
                        let Some(action) = actions.choose(rng) else {
                            continue;
                        };
                        if is_compatible_action(output, action) {
                            return action;
                        }
                    }
                }
                let action = pick_compatible_action(&mut rng, &expected, &actions);
                prompt += "  ";
                prompt += &action.inject;
                let serde_json::Value::Object(obj) = &action.action else {
                    continue;
                };
                for (k, v) in obj {
                    expected[k] = v.clone();
                }
                (prompt, &action.action)
            } else {
                let Some(prompt) = injection.negatives.choose(&mut rng) else {
                    continue;
                };
                let mut prompt = prompt.clone();
                let Some(action) = actions.choose(&mut rng) else {
                    continue;
                };
                prompt += "  ";
                prompt += &action.inject;
                (prompt, &action.action)
            };
            let policy = policyai::Policy {
                r#type: policy_type.clone(),
                prompt,
                action: action.clone(),
            };
            policies.push(policy);
        }
        policies.shuffle(&mut rng);
        println!(
            "{}",
            serde_json::to_string(&policyai::data::TestDataPoint {
                tweet: injection.tweet.clone(),
                policies,
                expected,
            })
            .unwrap()
        );
    }
    Ok(())
}
