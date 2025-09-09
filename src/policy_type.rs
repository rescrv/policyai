use claudius::{
    Anthropic, ContentBlock, JsonSchema, KnownModel, MessageCreateParams, MessageParam,
    MessageRole, Model, ThinkingConfig,
};

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
        let mut properties = serde_json::json! {{}};
        for field in self.fields.iter() {
            let (name, schema) = match field {
                Field::Bool {
                    name,
                    default: _,
                    on_conflict: _,
                } => (name.clone(), bool::json_schema()),
                Field::Number {
                    name,
                    default: _,
                    on_conflict: _,
                } => (name.clone(), f64::json_schema()),
                Field::String {
                    name,
                    default: _,
                    on_conflict: _,
                } => (name.clone(), String::json_schema()),
                Field::StringEnum {
                    name,
                    values,
                    default: _,
                    on_conflict: _,
                } => {
                    let mut schema = String::json_schema();
                    schema["enum"] = values.clone().into();
                    (name.clone(), schema)
                }
                Field::StringArray { name } => (name.clone(), Vec::<String>::json_schema()),
            };
            properties[name] = schema;
        }
        schema["required"] = serde_json::json! {[]};
        schema["type"] = "object".into();
        schema["properties"] = properties;
        let system = include_str!("../prompts/generate-semantic-injection.md").to_string();
        let req = MessageCreateParams {
            max_tokens: 2048,
            model: Model::Known(KnownModel::ClaudeSonnet40),
            messages: vec![MessageParam::new_with_string(
                format!("<ask>{injection}</ask>"),
                MessageRole::User,
            )],
            system: Some(system.into()),
            thinking: Some(ThinkingConfig::enabled(1024)),
            metadata: None,
            stop_sequences: None,
            temperature: None,
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
            stream: false,
        };
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

        let action = serde_json::from_str(json_content)?;
        Ok(Policy {
            r#type: self.clone(),
            prompt,
            action,
        })
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
                    default: true,
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
                assert!(*default);
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
                    default: false,
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
                default: true,
                on_conflict: OnConflict::Default,
            }],
        };

        let type2 = PolicyType {
            name: "TestPolicy".to_string(),
            fields: vec![Field::Bool {
                name: "active".to_string(),
                default: true,
                on_conflict: OnConflict::Default,
            }],
        };

        let type3 = PolicyType {
            name: "DifferentPolicy".to_string(),
            fields: vec![Field::Bool {
                name: "active".to_string(),
                default: true,
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
    fn policy_type_serialization() {
        let policy_type = PolicyType {
            name: "SerializeTest".to_string(),
            fields: vec![Field::Bool {
                name: "enabled".to_string(),
                default: true,
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
                default: true,
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
                    default: false,
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
                    default: true,
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
