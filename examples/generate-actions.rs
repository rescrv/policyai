#![allow(clippy::uninlined_format_args)]

use std::io::Read;

use arrrg::CommandLine;
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
                    format!("When this rule matches, output {{{name:?}: true}}."),
                    true,
                )
            } else {
                (
                    format!("When this rule matches, output {{{name:?}: false}}."),
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
            let numbers = [
                0.0,
                index as f64,
                -(index as f64),
                (index as f64) * 0.5,
                (index as f64) * 100.0,
                std::f64::consts::PI * (index as f64),
            ];
            let idx = range_to(numbers.len())(guac);
            let number = numbers[idx];
            let semantic_injection =
                format!("When this rule matches, output {{{name:?}: {number}}}.");
            serde_json::to_value(InjectableAction {
                inject: semantic_injection,
                action: serde_json::json! {{ name : number }},
            })
            .unwrap()
        }
        Field::String {
            name,
            on_conflict: _,
            default: _,
        } => {
            let strings = [
                "".to_string(),
                index.to_string(),
                format!("string_{}", index),
                format!("This is a longer string with index {}", index),
                "special!@#$%^&*()chars".to_string(),
                "unicode: 你好世界 🌍".to_string(),
                format!("line1\nline2\nindex:{}", index),
            ];
            let idx = range_to(strings.len())(guac);
            let string = strings[idx].clone();
            let semantic_injection = format!(
                "When this rule matches, output JSON {{{name:?}: {}}}.",
                serde_json::to_string(&string).unwrap()
            );
            serde_json::to_value(InjectableAction {
                inject: semantic_injection,
                action: serde_json::json! {{ name : string }},
            })
            .unwrap()
        }
        Field::StringArray { name } => {
            let arrays: Vec<Vec<String>> = vec![
                vec![],
                vec![index.to_string()],
                vec!["item1".to_string(), "item2".to_string()],
                vec![
                    "a".to_string(),
                    "b".to_string(),
                    "c".to_string(),
                    "d".to_string(),
                ],
                (0..index.min(10)).map(|i| format!("item_{}", i)).collect(),
            ];
            let idx = range_to(arrays.len())(guac);
            let array = arrays[idx].clone();
            let semantic_injection = if array.is_empty() {
                format!("When this rule matches, output JSON {{{name:?}: []}}.")
            } else {
                format!(
                    "When this rule matches, output JSON {{{name:?}: {}}}.",
                    serde_json::to_string(&array).unwrap()
                )
            };
            serde_json::to_value(InjectableAction {
                inject: semantic_injection,
                action: serde_json::json! {{ name : array }},
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
                "When this rule matches, output JSON {{{name:?}: {}}}.",
                serde_json::to_string(&value).unwrap()
            );
            serde_json::to_value(InjectableAction {
                inject: semantic_injection,
                action: serde_json::json! {{ name : value }},
            })
            .unwrap()
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, arrrg_derive::CommandLine)]
struct Options {
    #[arrrg(flag, "Generate actions for all field types (default)")]
    all: bool,
    #[arrrg(flag, "Generate actions only for bool fields")]
    bool: bool,
    #[arrrg(flag, "Generate actions only for number fields")]
    number: bool,
    #[arrrg(flag, "Generate actions only for string fields")]
    string: bool,
    #[arrrg(flag, "Generate actions only for enum fields")]
    enum_field: bool,
    #[arrrg(flag, "Generate actions only for array fields")]
    array: bool,
    #[arrrg(optional, "Total number of actions to generate (default: 1000)")]
    count: Option<usize>,
}

fn should_generate_for_field(field: &Field, options: &Options) -> bool {
    // If no specific type is selected, or --all is specified, generate for all
    if options.all
        || (!options.bool
            && !options.number
            && !options.string
            && !options.enum_field
            && !options.array)
    {
        return true;
    }

    match field {
        Field::Bool { .. } => options.bool,
        Field::Number { .. } => options.number,
        Field::String { .. } => options.string,
        Field::StringEnum { .. } => options.enum_field,
        Field::StringArray { .. } => options.array,
    }
}

fn main() {
    let (options, _free) = Options::from_command_line(
        "Usage: generate-actions [options] < policy.txt > actions.jsonl",
    );

    let mut guac = Guacamole::new(0);
    let mut buf = vec![];
    std::io::stdin()
        .read_to_end(&mut buf)
        .expect("could not read policy type on stdin");
    let buf = String::from_utf8(buf).expect("policy type should be UTF8");
    let policy_type = PolicyType::parse(&buf).expect("policy type should be valid");

    let total_actions = options.count.unwrap_or(1000);

    // Filter fields based on options
    let eligible_fields: Vec<&Field> = policy_type
        .fields
        .iter()
        .filter(|field| should_generate_for_field(field, &options))
        .collect();

    if eligible_fields.is_empty() {
        eprintln!("No fields match the specified criteria");
        return;
    }

    // Generate actions cycling through eligible fields
    for i in 0..total_actions {
        let field = eligible_fields[i % eligible_fields.len()];
        let line_number = i + 1;
        println!(
            "{}",
            serde_json::to_string(&generate_case(&mut guac, &policy_type, line_number, field,))
                .unwrap()
        );
    }
}
