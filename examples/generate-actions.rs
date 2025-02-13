use std::io::Read;

use guacamole::combinators::*;
use guacamole::Guacamole;

use policyai::data::InjectableAction;
use policyai::{Field, PolicyType};

fn generate_case(
    guac: &mut Guacamole,
    _policy_type: &PolicyType,
    index: usize,
    field: &Field,
) -> serde_json::Value {
    match field {
        Field::Bool {
            name,
            on_conflict: _,
            default: _,
        } => {
            let (semantic_injection, truth) = if coin()(guac) {
                (
                    format!("When this rule matches, output {{{:?}: true}}.", name),
                    true,
                )
            } else {
                (
                    format!("When this rule matches, output {{{:?}: false}}.", name),
                    false,
                )
            };
            serde_json::to_value(InjectableAction {
                inject: semantic_injection,
                action: serde_json::json! {{ name : truth }},
            })
            .unwrap()
        }
        Field::Number {
            name,
            on_conflict: _,
            default: _,
        } => {
            let semantic_injection =
                format!("When this rule matches, output {{{:?}: {}}}.", name, index);
            serde_json::to_value(InjectableAction {
                inject: semantic_injection,
                action: serde_json::json! {{ name : index }},
            })
            .unwrap()
        }
        Field::String {
            name,
            on_conflict: _,
            default: _,
        } => {
            let semantic_injection = format!(
                "When this rule matches, output {{{:?}: \"{}\"}}.",
                name, index
            );
            serde_json::to_value(InjectableAction {
                inject: semantic_injection,
                action: serde_json::json! {{ name : index.to_string() }},
            })
            .unwrap()
        }
        Field::StringArray { name } => {
            let semantic_injection = format!(
                "When this rule matches, output {{{:?}: [\"{}\"]}}.",
                name, index
            );
            serde_json::to_value(InjectableAction {
                inject: semantic_injection,
                action: serde_json::json! {{ name : vec![index.to_string()] }},
            })
            .unwrap()
        }
        Field::StringEnum {
            name,
            values,
            on_conflict: _,
            default: _,
        } => {
            let value = select(range_to(values.len()), values)(guac);
            let semantic_injection = format!(
                "When this rule matches, output {{{:?}: {:?}}}.",
                name, value
            );
            serde_json::to_value(InjectableAction {
                inject: semantic_injection,
                action: serde_json::json! {{ name : value }},
            })
            .unwrap()
        }
    }
}

fn main() {
    let mut guac = Guacamole::new(0);
    let mut buf = vec![];
    std::io::stdin()
        .read_to_end(&mut buf)
        .expect("could not read policy type on stdin");
    let buf = String::from_utf8(buf).expect("policy type should be UTF8");
    let policy_type = PolicyType::parse(&buf).expect("policy type should be valid");
    let mut line_number = 0;
    let mut cases = 0;
    while cases < 1_000 {
        line_number += 1;
        if !matches!(
            policy_type.fields[line_number % policy_type.fields.len()],
            Field::Bool { .. }
        ) {
            continue;
        }
        println!(
            "{}",
            serde_json::to_string(&generate_case(
                &mut guac,
                &policy_type,
                line_number,
                &policy_type.fields[line_number % policy_type.fields.len()],
            ))
            .unwrap()
        );
        cases += 1;
    }
}
