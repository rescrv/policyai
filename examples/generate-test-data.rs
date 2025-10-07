use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use arrrg::CommandLine;
use rand::prelude::*;

use policyai::data::{ConflictField, InjectableAction};
use policyai::{Field, OnConflict, PolicyType};

#[derive(Clone, Default, Debug, arrrg_derive::CommandLine)]
struct Options {
    #[arrrg(required, "The decidable semantic injections.")]
    decidables: String,
    #[arrrg(required, "The actions.")]
    actions: String,
    #[arrrg(required, "This many texts will be selected to have policies applied.")]
    samples: usize,
    #[arrrg(required, "The policy type definition.")]
    policy: String,
    #[arrrg(required, "This many policies will be selected per text.")]
    policies: usize,
    #[arrrg(required, "This many policies will be enforced to match per text.")]
    matching: usize,
    #[arrrg(
        optional,
        "Rate of test cases that should contain conflicts (0.0 to 1.0)."
    )]
    conflict_rate: Option<f64>,
}

impl Eq for Options {}

impl PartialEq for Options {
    fn eq(&self, other: &Self) -> bool {
        self.decidables == other.decidables
            && self.actions == other.actions
            && self.samples == other.samples
            && self.policy == other.policy
            && self.policies == other.policies
            && self.matching == other.matching
            && match (self.conflict_rate, other.conflict_rate) {
                (None, None) => true,
                (Some(a), Some(b)) => (a - b).abs() < f64::EPSILON,
                _ => false,
            }
    }
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
    let policy_type =
        PolicyType::parse(&std::fs::read_to_string(&options.policy).unwrap()).unwrap();
    let conflict_rate = options.conflict_rate.unwrap_or(0.0);
    let mut rng = rand::rng();
    for _ in 0..options.samples {
        let injection = semantic_injections.choose(&mut rng).unwrap();
        assert!(injection.positives.len() >= options.matching);
        assert!(injection.negatives.len() >= options.policies - options.matching);

        // Decide whether to generate conflicts
        let should_generate_conflicts = rng.random_bool(conflict_rate);

        if should_generate_conflicts {
            // Find fields with OnConflict::Agreement
            let agreement_fields: Vec<&Field> = policy_type
                .fields
                .iter()
                .filter(|field| {
                    matches!(
                        field,
                        Field::Bool {
                            on_conflict: OnConflict::Agreement,
                            ..
                        } | Field::String {
                            on_conflict: OnConflict::Agreement,
                            ..
                        } | Field::StringEnum {
                            on_conflict: OnConflict::Agreement,
                            ..
                        } | Field::Number {
                            on_conflict: OnConflict::Agreement,
                            ..
                        }
                    )
                })
                .collect();

            if agreement_fields.is_empty() {
                eprintln!("no fields require agreement; cannot generate conflicts");
                std::process::exit(13);
            } else {
                generate_conflict_test_case(
                    &mut rng,
                    injection,
                    &actions,
                    &policy_type,
                    &options,
                    &agreement_fields,
                );
            }
        } else {
            generate_normal_test_case(&mut rng, injection, &actions, &policy_type, &options);
        }
    }
    Ok(())
}

fn generate_normal_test_case(
    rng: &mut impl Rng,
    injection: &policyai::data::DecidableSemanticInjection,
    actions: &[InjectableAction],
    policy_type: &PolicyType,
    options: &Options,
) {
    let mut policies = vec![];
    let mut expected = serde_json::json! {{}};
    while policies.len() < options.policies {
        let (prompt, inject, action) = if policies.len() < options.matching {
            let Some(prompt) = injection.positives.choose(rng) else {
                continue;
            };
            let prompt = prompt.clone();
            fn is_compatible_action(output: &serde_json::Value, action: &InjectableAction) -> bool {
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
                rng: &mut impl Rng,
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
            let action = pick_compatible_action(rng, &expected, actions);
            let serde_json::Value::Object(obj) = &action.action else {
                continue;
            };
            for (k, v) in obj {
                expected[k] = v.clone();
            }
            (prompt, &action.inject, &action.action)
        } else {
            let Some(prompt) = injection.negatives.choose(rng) else {
                continue;
            };
            let prompt = prompt.clone();
            let Some(action) = actions.choose(rng) else {
                continue;
            };
            (prompt, &action.inject, &action.action)
        };
        let policy = policyai::Policy {
            r#type: policy_type.clone(),
            prompt: format!("<match>{prompt}</match><action>{inject}</action>"),
            action: action.clone(),
        };
        policies.push(policy);
    }
    policies.shuffle(rng);
    println!(
        "{}",
        serde_json::to_string(&policyai::data::TestDataPoint {
            text: injection.text.clone(),
            policies,
            expected: Some(expected),
            conflicts: None,
        })
        .unwrap()
    );
}

fn generate_conflict_test_case(
    rng: &mut impl Rng,
    injection: &policyai::data::DecidableSemanticInjection,
    actions: &[InjectableAction],
    policy_type: &PolicyType,
    options: &Options,
    agreement_fields: &[&Field],
) {
    let mut policies = vec![];
    let mut conflicts = vec![];

    // Decide how many fields to create conflicts for (at least 1, up to all)
    let num_conflict_fields = rng.random_range(1..=agreement_fields.len());
    let selected_fields: Vec<_> = agreement_fields
        .choose_multiple(rng, num_conflict_fields)
        .cloned()
        .collect();

    // For each selected field, generate conflicting values
    for field in selected_fields {
        let field_name = field.name();
        conflicts.push(ConflictField {
            conflict_type: "agreement".to_string(),
            field_name: field_name.to_string(),
        });

        // Generate 2-4 different values for this field
        let num_values = rng.random_range(2..=4.min(options.matching));
        let mut seen_values = std::collections::HashSet::<String>::new();

        for _ in 0..num_values {
            if policies.len() >= options.policies {
                break;
            }

            // Find an action with a different value for this field
            let action = loop {
                let candidate = actions.choose(rng).unwrap();
                if let serde_json::Value::Object(obj) = &candidate.action {
                    if let Some(val) = obj.get(field_name) {
                        let val_str = serde_json::to_string(val).unwrap();
                        if !seen_values.contains(&val_str) {
                            seen_values.insert(val_str);
                            break candidate;
                        }
                    }
                }
            };

            // Use positive prompt since we want these to match
            let prompt = injection.positives.choose(rng).unwrap();
            let prompt = prompt.clone();
            let inject = &action.inject;
            let policy = policyai::Policy {
                r#type: policy_type.clone(),
                prompt: format!("<match>{prompt}</match><action>{inject}</action>"),
                action: action.action.clone(),
            };
            policies.push(policy);
        }
    }

    // Fill remaining slots with non-matching policies
    while policies.len() < options.policies {
        let prompt = injection.negatives.choose(rng).unwrap();
        let mut prompt = prompt.clone();
        let action = actions.choose(rng).unwrap();
        prompt += "  ";
        prompt += &action.inject;

        let policy = policyai::Policy {
            r#type: policy_type.clone(),
            prompt,
            action: action.action.clone(),
        };
        policies.push(policy);
    }

    policies.shuffle(rng);
    println!(
        "{}",
        serde_json::to_string(&policyai::data::TestDataPoint {
            text: injection.text.clone(),
            policies,
            expected: None,
            conflicts: Some(conflicts),
        })
        .unwrap()
    );
}
