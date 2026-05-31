use claudius::{
    Anthropic, ContentBlock, JsonSchema, KnownModel, MessageCreateParams, MessageParam,
    MessageRole, Model, OutputFormat,
};
use uuid::Uuid;

use crate::{parser, Field, ParseError, Policy};

/// Represents a policy type definition with a name and a set of typed fields.
///
/// A PolicyType defines the structure of data that policies will work with,
/// including field names, types, defaults, and conflict resolution strategies.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct PolicyType {
    /// The name of this policy type (e.g., "policyai::EmailPolicy")
    pub name: String,
    /// The fields that make up this policy type
    pub fields: Vec<Field>,
}

impl PolicyType {
    /// Parse a PolicyType from its textual representation.
    ///
    /// # Example
    /// ```
    /// use policyai::PolicyType;
    /// let policy_type = PolicyType::parse("type MyPolicy { unread: bool = true }").unwrap();
    /// ```
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        parser::parse(input.trim())
    }

    /// Get the default value for this policy type.
    ///
    /// Returns a JSON object where each field name maps to its default value.
    /// Fields without defaults will have null values (for String, Number, StringEnum)
    /// or their type-specific defaults (bool fields always have a default, arrays default to []).
    pub fn default_value(&self) -> serde_json::Value {
        let mut defaults = serde_json::Map::new();
        for field in self.fields.iter() {
            let v = field.default_value();
            match &v {
                serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
                    defaults.insert(field.name().to_string(), v);
                }
                serde_json::Value::String(s) if !s.is_empty() => {
                    defaults.insert(field.name().to_string(), v);
                }
                serde_json::Value::Array(a) if !a.is_empty() => {
                    defaults.insert(field.name().to_string(), v);
                }
                serde_json::Value::Object(o) if !o.is_empty() => {
                    defaults.insert(field.name().to_string(), v);
                }
                _ => {}
            }
        }
        serde_json::Value::Object(defaults)
    }

    /// Create a new Policy by applying a semantic injection to this PolicyType.
    ///
    /// The semantic injection is a natural language description that gets converted
    /// into structured actions that conform to this PolicyType's schema.
    pub async fn with_semantic_injection(
        &self,
        client: &Anthropic,
        injection: &str,
    ) -> Result<Policy, claudius::Error> {
        let mut schema = serde_json::json! {{}};
        let mut action_masks = Vec::new();
        for field in self.fields.iter() {
            let field_mask = Uuid::new_v4().to_string();
            let (name, schema, enum_values) = match field {
                Field::Bool {
                    name,
                    default: _,
                    on_conflict: _,
                } => (name.clone(), bool::json_schema(), Vec::new()),
                Field::Number {
                    name,
                    default: _,
                    on_conflict: _,
                } => (name.clone(), f64::json_schema(), Vec::new()),
                Field::String {
                    name,
                    default: _,
                    on_conflict: _,
                } => (name.clone(), String::json_schema(), Vec::new()),
                Field::StringEnum {
                    name,
                    values,
                    default: _,
                    on_conflict: _,
                } => {
                    let enum_values = values
                        .iter()
                        .map(|value| (value.clone(), Uuid::new_v4().to_string()))
                        .collect::<Vec<_>>();
                    let mut schema = String::json_schema();
                    schema["enum"] = enum_values
                        .iter()
                        .map(|(_, mask)| mask.clone())
                        .collect::<Vec<_>>()
                        .into();
                    (name.clone(), schema, enum_values)
                }
                Field::StringArray { name } => {
                    (name.clone(), Vec::<String>::json_schema(), Vec::new())
                }
            };
            action_masks.push(SemanticActionMask {
                name,
                mask: field_mask,
                enum_values,
                schema,
            });
        }
        let mut masked_injection = semantic_action_clause(injection).to_string();
        for action_mask in &action_masks {
            masked_injection =
                replace_token(&masked_injection, &action_mask.name, &action_mask.mask);
            if let Some(singular) = action_mask.name.strip_suffix('s') {
                masked_injection = replace_token(&masked_injection, singular, &action_mask.mask);
            }
            for (value, mask) in &action_mask.enum_values {
                masked_injection =
                    masked_injection.replace(&format!("{value:?}"), &format!("{mask:?}"));
            }
        }
        let mut properties = serde_json::json! {{}};
        let mut required = Vec::new();
        for action_mask in &action_masks {
            if masked_injection.contains(&action_mask.mask) {
                properties[&action_mask.mask] = action_mask.schema.clone();
                required.push(action_mask.mask.clone());
            }
        }
        schema["required"] = required.into();
        schema["type"] = "object".into();
        schema["properties"] = properties;
        schema["additionalProperties"] = false.into();
        let system = include_str!("../prompts/generate-semantic-injection.md").to_string();
        let req = MessageCreateParams {
            max_tokens: 2048,
            model: Model::Known(KnownModel::ClaudeOpus48),
            cache_control: None,
            messages: vec![MessageParam::new_with_string(
                format!("<ask>{masked_injection}</ask>"),
                MessageRole::User,
            )],
            system: Some(system.into()),
            thinking: None,
            metadata: None,
            output_config: None,
            output_format: Some(OutputFormat::json_schema(schema)),
            stop_sequences: None,
            temperature: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            stream: false,
            betas: None,
        };
        if std::env::var_os("POLICYAI_LOG_SEMANTIC_REQUEST").is_some() {
            eprintln!(
                "semantic injection request:\n{}",
                serde_json::to_string_pretty(&req)?
            );
        }
        let resp = client.send(req).await?;
        let prompt = injection.to_string();
        let raw_response = resp
            .content
            .iter()
            .flat_map(|c| {
                if let ContentBlock::Text(t) = c {
                    Some(t.text.clone())
                } else {
                    None
                }
            })
            .collect::<String>();

        // Extract JSON from markdown code blocks if present
        let json_content = if let Some(start) = raw_response.find("```json") {
            if let Some(end) = raw_response[start + 7..].find("```") {
                raw_response[start + 7..start + 7 + end].trim()
            } else {
                raw_response.trim()
            }
        } else if let Some(start) = raw_response.find('{') {
            if let Some(end) = raw_response.rfind('}') {
                &raw_response[start..=end]
            } else {
                raw_response.trim()
            }
        } else {
            raw_response.trim()
        };

        let mut action = serde_json::from_str(json_content)?;
        unmask_semantic_action(&action_masks, &mut action);
        self.sanitize_semantic_action(&mut action);
        Ok(Policy {
            r#type: self.clone(),
            prompt,
            action,
        })
    }

    fn sanitize_semantic_action(&self, action: &mut serde_json::Value) {
        let serde_json::Value::Object(action) = action else {
            return;
        };
        action.retain(|name, value| {
            let Some(field) = self.fields.iter().find(|field| field.name() == name) else {
                return false;
            };
            coerce_action_value(field, value)
        });
    }
}

struct SemanticActionMask {
    name: String,
    mask: String,
    enum_values: Vec<(String, String)>,
    schema: serde_json::Value,
}

fn unmask_semantic_action(masks: &[SemanticActionMask], action: &mut serde_json::Value) {
    let serde_json::Value::Object(object) = action else {
        return;
    };
    let mut unmasked = serde_json::Map::new();
    for mask in masks {
        let Some(mut value) = object.remove(&mask.mask) else {
            continue;
        };
        if let Some(enum_value) = value.as_str().and_then(|value_string| {
            mask.enum_values
                .iter()
                .find(|(_, enum_mask)| enum_mask == value_string)
                .map(|(enum_value, _)| enum_value)
        }) {
            value = serde_json::Value::String(enum_value.clone());
        }
        unmasked.insert(mask.name.clone(), value);
    }
    *action = serde_json::Value::Object(unmasked);
}

fn semantic_action_clause(injection: &str) -> &str {
    let trimmed = injection.trim();
    if let Some((_, action)) = trimmed.split_once(':') {
        return action.trim();
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("if ") || lower.starts_with("when ") {
        if let Some(index) = trimmed.find(',') {
            return trimmed[index + 1..].trim();
        }
    }
    trimmed
}

fn replace_token(input: &str, token: &str, replacement: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut token_start = None;

    for (index, ch) in input.char_indices() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            token_start.get_or_insert(index);
        } else {
            if let Some(start) = token_start.take() {
                push_replaced_token(input, start, index, token, replacement, &mut output);
            }
            output.push(ch);
        }
    }

    if let Some(start) = token_start {
        push_replaced_token(input, start, input.len(), token, replacement, &mut output);
    }

    output
}

fn push_replaced_token(
    input: &str,
    start: usize,
    end: usize,
    token: &str,
    replacement: &str,
    output: &mut String,
) {
    let found = &input[start..end];
    if found.eq_ignore_ascii_case(token) {
        output.push_str(replacement);
    } else {
        output.push_str(found);
    }
}

fn coerce_action_value(field: &Field, value: &mut serde_json::Value) -> bool {
    match field {
        Field::Bool { .. } => match value {
            serde_json::Value::Bool(_) => true,
            serde_json::Value::String(s) if s == "true" => {
                *value = serde_json::Value::Bool(true);
                true
            }
            serde_json::Value::String(s) if s == "false" => {
                *value = serde_json::Value::Bool(false);
                true
            }
            _ => false,
        },
        Field::Number { .. } => match value {
            serde_json::Value::Number(_) => true,
            serde_json::Value::String(s) => s
                .parse::<f64>()
                .ok()
                .and_then(serde_json::Number::from_f64)
                .map(|number| {
                    *value = serde_json::Value::Number(number);
                })
                .is_some(),
            _ => false,
        },
        Field::String { .. } => value.is_string(),
        Field::StringEnum { values, .. } => value
            .as_str()
            .is_some_and(|value| values.iter().any(|allowed| allowed == value)),
        Field::StringArray { .. } => value
            .as_array()
            .is_some_and(|values| values.iter().all(serde_json::Value::is_string)),
    }
}

impl std::fmt::Display for PolicyType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        writeln!(f, "type {} {{", self.name)?;
        for field in self.fields.iter() {
            writeln!(f, "    {field},")?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::OnConflict;

    fn create_test_policy_type() -> PolicyType {
        PolicyType {
            name: "TestPolicy".to_string(),
            fields: vec![
                Field::Bool {
                    name: "active".to_string(),
                    default: Some(true),
                    on_conflict: OnConflict::Default,
                },
                Field::String {
                    name: "title".to_string(),
                    default: Some("untitled".to_string()),
                    on_conflict: OnConflict::Agreement,
                },
                Field::StringEnum {
                    name: "priority".to_string(),
                    values: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
                    default: Some("low".to_string()),
                    on_conflict: OnConflict::LargestValue,
                },
                Field::StringArray {
                    name: "tags".to_string(),
                },
                Field::Number {
                    name: "score".to_string(),
                    default: Some(crate::t64(0.0)),
                    on_conflict: OnConflict::LargestValue,
                },
            ],
        }
    }

    #[test]
    fn policy_type_creation() {
        let policy_type = create_test_policy_type();
        assert_eq!(policy_type.name, "TestPolicy");
        assert_eq!(policy_type.fields.len(), 5);
    }

    #[test]
    fn policy_type_parse_simple() {
        let input = "type SimplePolicy { active: bool = true }";
        let result = PolicyType::parse(input);
        assert!(result.is_ok());

        let policy_type = result.unwrap();
        assert_eq!(policy_type.name, "SimplePolicy");
        assert_eq!(policy_type.fields.len(), 1);

        match &policy_type.fields[0] {
            Field::Bool { name, default, .. } => {
                assert_eq!(name, "active");
                assert_eq!(*default, Some(true));
            }
            _ => panic!("Expected Bool field"),
        }
    }

    #[test]
    fn policy_type_parse_multiple_fields() {
        let input = r#"type ComplexPolicy {
            enabled: bool = false,
            message: string = "hello",
            count: number = 42
        }"#;

        let result = PolicyType::parse(input);
        assert!(result.is_ok());

        let policy_type = result.unwrap();
        assert_eq!(policy_type.name, "ComplexPolicy");
        assert_eq!(policy_type.fields.len(), 3);
    }

    #[test]
    fn policy_type_parse_with_enum() {
        let input = r#"type PolicyWithEnum {
            status: ["pending", "active", "completed"] = "pending"
        }"#;

        let result = PolicyType::parse(input);
        println!("Parse enum result: {result:?}"); // Debug output
        assert!(result.is_ok());

        let policy_type = result.unwrap();
        assert_eq!(policy_type.fields.len(), 1);

        match &policy_type.fields[0] {
            Field::StringEnum {
                name,
                values,
                default,
                ..
            } => {
                assert_eq!(name, "status");
                assert_eq!(values.len(), 3);
                assert_eq!(values[0], "pending");
                assert_eq!(values[1], "active");
                assert_eq!(values[2], "completed");
                assert_eq!(*default, Some("pending".to_string()));
            }
            _ => panic!("Expected StringEnum field"),
        }
    }

    #[test]
    fn policy_type_parse_with_array() {
        let input = "type PolicyWithArray { tags: [string] }";
        let result = PolicyType::parse(input);
        println!("Parse result for '{input}': {result:?}"); // Debug output
        assert!(result.is_ok());

        let policy_type = result.unwrap();
        assert_eq!(policy_type.fields.len(), 1);

        match &policy_type.fields[0] {
            Field::StringArray { name } => {
                assert_eq!(name, "tags");
            }
            _ => panic!("Expected StringArray field"),
        }
    }

    #[test]
    fn policy_type_parse_with_conflict_strategies() {
        let input = r#"type ConflictPolicy {
            field1: bool @ agreement = false,
            field2: string @ agreement = "test",
            field3: number @ last wins = 10
        }"#;

        let result = PolicyType::parse(input);
        println!("Parse conflicts result: {result:?}"); // Debug output
        assert!(result.is_ok());

        let policy_type = result.unwrap();
        assert_eq!(policy_type.fields.len(), 3);

        match &policy_type.fields[0] {
            Field::Bool { on_conflict, .. } => {
                assert_eq!(*on_conflict, OnConflict::Agreement);
            }
            _ => panic!("Expected Bool field"),
        }

        match &policy_type.fields[1] {
            Field::String { on_conflict, .. } => {
                assert_eq!(*on_conflict, OnConflict::Agreement);
            }
            _ => panic!("Expected String field"),
        }

        match &policy_type.fields[2] {
            Field::Number { on_conflict, .. } => {
                assert_eq!(*on_conflict, OnConflict::LargestValue);
            }
            _ => panic!("Expected Number field"),
        }
    }

    #[test]
    fn policy_type_parse_invalid_syntax() {
        let invalid_inputs = vec![
            "type",
            "type InvalidField { field: unknown }",
            "type MissingBrace { field: bool",
            "InvalidType { field: bool }",
            "type 123Invalid { field: bool }",
        ];

        for input in invalid_inputs {
            let result = PolicyType::parse(input);
            assert!(result.is_err(), "Expected parse error for: {input}");
        }
    }

    #[test]
    fn policy_type_display() {
        let policy_type = PolicyType {
            name: "DisplayTest".to_string(),
            fields: vec![
                Field::Bool {
                    name: "flag".to_string(),
                    default: Some(false),
                    on_conflict: OnConflict::Default,
                },
                Field::String {
                    name: "text".to_string(),
                    default: None,
                    on_conflict: OnConflict::Agreement,
                },
            ],
        };

        let display_str = format!("{policy_type}");
        assert!(display_str.contains("type DisplayTest {"));
        assert!(display_str.contains("flag"));
        assert!(display_str.contains("text"));
        assert!(display_str.contains("}"));
    }

    #[test]
    fn policy_type_equality() {
        let type1 = PolicyType {
            name: "TestPolicy".to_string(),
            fields: vec![Field::Bool {
                name: "active".to_string(),
                default: Some(true),
                on_conflict: OnConflict::Default,
            }],
        };

        let type2 = PolicyType {
            name: "TestPolicy".to_string(),
            fields: vec![Field::Bool {
                name: "active".to_string(),
                default: Some(true),
                on_conflict: OnConflict::Default,
            }],
        };

        let type3 = PolicyType {
            name: "DifferentPolicy".to_string(),
            fields: vec![Field::Bool {
                name: "active".to_string(),
                default: Some(true),
                on_conflict: OnConflict::Default,
            }],
        };

        assert_eq!(type1, type2);
        assert_ne!(type1, type3);
    }

    #[test]
    fn policy_type_clone() {
        let original = create_test_policy_type();
        let cloned = original.clone();

        assert_eq!(original.name, cloned.name);
        assert_eq!(original.fields.len(), cloned.fields.len());
        assert_eq!(original, cloned);
    }

    #[test]
    fn policy_type_debug() {
        let policy_type = PolicyType {
            name: "DebugTest".to_string(),
            fields: vec![],
        };

        let debug_str = format!("{policy_type:?}");
        assert!(debug_str.contains("PolicyType"));
        assert!(debug_str.contains("DebugTest"));
        assert!(debug_str.contains("fields"));
    }

    #[test]
    fn sanitize_semantic_action_drops_unknown_fields() {
        let policy_type = create_test_policy_type();
        let mut action = serde_json::json!({
            "active": true,
            "priority": "high",
            "extra": "ignored",
        });

        policy_type.sanitize_semantic_action(&mut action);

        assert_eq!(
            serde_json::json!({
                "active": true,
                "priority": "high",
            }),
            action
        );
    }

    #[test]
    fn sanitize_semantic_action_coerces_simple_values() {
        let policy_type = create_test_policy_type();
        let mut action = serde_json::json!({
            "active": "true",
            "score": "42.5",
            "tags": ["example"],
            "extra": "ignored",
        });

        policy_type.sanitize_semantic_action(&mut action);

        assert_eq!(
            serde_json::json!({
                "active": true,
                "score": 42.5,
                "tags": ["example"],
            }),
            action
        );
    }

    #[test]
    fn semantic_action_clause_removes_condition() {
        assert_eq!(
            "Set \"priority\" to \"low\" and \"unread\" to true.",
            semantic_action_clause(
                "When the email is about AI:  Set \"priority\" to \"low\" and \"unread\" to true."
            )
        );
        assert_eq!(
            "set \"category\" to \"distributed systems\".",
            semantic_action_clause(
                "If the user talks about Paxos, set \"category\" to \"distributed systems\"."
            )
        );
        assert_eq!(
            "Assign weight to the email.",
            semantic_action_clause("Assign weight to the email.")
        );
    }

    #[test]
    fn policy_type_serialization() {
        let policy_type = PolicyType {
            name: "SerializeTest".to_string(),
            fields: vec![Field::Bool {
                name: "enabled".to_string(),
                default: Some(true),
                on_conflict: OnConflict::Default,
            }],
        };

        let serialized = serde_json::to_string(&policy_type).unwrap();
        assert!(serialized.contains("SerializeTest"));
        assert!(serialized.contains("enabled"));

        let deserialized: PolicyType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(policy_type, deserialized);
    }

    #[test]
    fn policy_type_display_parse_roundtrip_simple() {
        let original = PolicyType {
            name: "RoundTripTest".to_string(),
            fields: vec![Field::Bool {
                name: "active".to_string(),
                default: Some(true),
                on_conflict: OnConflict::Default,
            }],
        };

        let displayed = format!("{original}");
        println!("Displayed PolicyType:\n{displayed}");
        let parsed = PolicyType::parse(&displayed).expect("Failed to parse displayed PolicyType");
        assert_eq!(original, parsed);
    }

    #[test]
    fn policy_type_display_parse_roundtrip_complex() {
        let original = PolicyType {
            name: "ComplexRoundTrip".to_string(),
            fields: vec![
                Field::Bool {
                    name: "enabled".to_string(),
                    default: Some(false),
                    on_conflict: OnConflict::Agreement,
                },
                Field::String {
                    name: "title".to_string(),
                    default: Some("default_title".to_string()),
                    on_conflict: OnConflict::Default,
                },
                Field::Number {
                    name: "count".to_string(),
                    default: Some(crate::t64(42.0)),
                    on_conflict: OnConflict::LargestValue,
                },
                Field::StringEnum {
                    name: "priority".to_string(),
                    values: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
                    default: Some("medium".to_string()),
                    on_conflict: OnConflict::LargestValue,
                },
                Field::StringArray {
                    name: "tags".to_string(),
                },
            ],
        };

        let displayed = format!("{original}");
        println!("Displayed complex PolicyType:\n{displayed}"); // Debug output
        let parsed = PolicyType::parse(&displayed).expect("Failed to parse displayed PolicyType");
        assert_eq!(original, parsed);
    }

    #[test]
    fn policy_type_display_parse_roundtrip_with_all_conflict_types() {
        let original = PolicyType {
            name: "ConflictRoundTrip".to_string(),
            fields: vec![
                Field::Bool {
                    name: "field1".to_string(),
                    default: Some(true),
                    on_conflict: OnConflict::Default,
                },
                Field::String {
                    name: "field2".to_string(),
                    default: Some("test".to_string()),
                    on_conflict: OnConflict::Agreement,
                },
                Field::Number {
                    name: "field3".to_string(),
                    default: Some(crate::t64(100.0)),
                    on_conflict: OnConflict::LargestValue,
                },
            ],
        };

        let displayed = format!("{original}");
        let parsed = PolicyType::parse(&displayed).expect("Failed to parse displayed PolicyType");
        assert_eq!(original, parsed);
    }

    #[test]
    fn debug_parse_simple_with_default() {
        let input = r#"type Test {
    field1: bool = true,
}"#;
        let _pt = PolicyType::parse(input).expect("Failed to parse simple bool with default");
    }

    #[test]
    fn debug_parse_with_conflict() {
        let input = r#"type Test {
    field2: string @ agreement = "test",
}"#;
        let _pt = PolicyType::parse(input).expect("Failed to parse string with agreement conflict");
    }

    #[test]
    fn debug_parse_exact_failing_case() {
        let input = r#"type ConflictRoundTrip {
    field1: bool = true,
    field2: string @ agreement = "test",
    field3: number @ last wins = 100,
}"#;
        let _pt = PolicyType::parse(input).expect("Failed to parse exact failing case");
    }

    #[test]
    fn policy_type_display_parse_roundtrip_empty_fields() {
        let original = PolicyType {
            name: "EmptyFieldsRoundTrip".to_string(),
            fields: vec![],
        };

        let displayed = format!("{original}");
        let parsed = PolicyType::parse(&displayed).expect("Failed to parse displayed PolicyType");
        assert_eq!(original, parsed);
    }

    #[test]
    fn policy_type_display_parse_roundtrip_no_defaults() {
        let original = PolicyType {
            name: "NoDefaultsRoundTrip".to_string(),
            fields: vec![
                Field::String {
                    name: "optional_string".to_string(),
                    default: None,
                    on_conflict: OnConflict::Agreement,
                },
                Field::Number {
                    name: "optional_number".to_string(),
                    default: None,
                    on_conflict: OnConflict::Default,
                },
                Field::StringEnum {
                    name: "optional_enum".to_string(),
                    values: vec!["a".to_string(), "b".to_string()],
                    default: None,
                    on_conflict: OnConflict::LargestValue,
                },
            ],
        };

        let displayed = format!("{original}");
        let parsed = PolicyType::parse(&displayed).expect("Failed to parse displayed PolicyType");
        assert_eq!(original, parsed);
    }
}
