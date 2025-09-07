use claudius::{
    Anthropic, CacheControlEphemeral, ContentBlock, KnownModel, MessageCreateParams, MessageParam,
    MessageParamContent, MessageRole, Model, StopReason, SystemPrompt, TextBlock, ThinkingConfig,
};

use crate::{Policy, Usage};

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SemanticInjection {
    pub injections: Vec<String>,
    pub rationales: Vec<String>,
    pub text: String,
}

pub async fn policy_applies(
    client: &Anthropic,
    text: &str,
    semantic_injection: &str,
    k: usize,
    n: usize,
) -> Result<bool, claudius::Error> {
    Ok(apply_policy_fractional(client, text, semantic_injection, k, n).await? >= k)
}

pub async fn policy_does_not_apply(
    client: &Anthropic,
    text: &str,
    semantic_injection: &str,
    k: usize,
    n: usize,
) -> Result<bool, claudius::Error> {
    Ok(
        apply_policy_fractional(client, text, semantic_injection, k, n).await?
            <= n.saturating_sub(k),
    )
}

async fn apply_policy_fractional(
    client: &Anthropic,
    text: &str,
    semantic_injection: &str,
    k: usize,
    n: usize,
) -> Result<usize, claudius::Error> {
    let mut success = 0;
    let mut total = 0;
    while success < k && total < n {
        total += 1;
        let system = r#"
You are an expert writer.  We are developing an instruction-processing engine that takes as input
instructions and text to output JSON.  Every instruction has two parts, first it has the _semantic
injection_.  This is natural language text that says something about the content being processed.
Second, an instruction has an associated output.  This, too, is a natural language text but it says
something about the JSON we are constructing.

You are evaluating semantic injections to see if they would "match" the text.

Output just one word indicating that the policy does or does not apply.
Say, "yes" to indicate the policy applies.
Say, "no" to indicate the policy does not apply.

Output just this one-word answer
"#
        .to_string();
        let req = MessageCreateParams {
            max_tokens: 1030,
            model: Model::Known(KnownModel::ClaudeSonnet40),
            system: Some(SystemPrompt::from_blocks(vec![TextBlock {
                text: system.to_string(),
                cache_control: Some(CacheControlEphemeral::new()),
                citations: None,
            }])),
            messages: vec![MessageParam {
                content: MessageParamContent::Array(vec![
                    ContentBlock::Text(TextBlock {
                        text: format!("<policy>{semantic_injection}</policy>"),
                        cache_control: None,
                        citations: None,
                    }),
                    ContentBlock::Text(TextBlock {
                        text: format!("<text>{text}</text>"),
                        cache_control: None,
                        citations: None,
                    }),
                ]),
                role: MessageRole::User,
            }],
            stop_sequences: Some(vec!["yes".to_string(), "no".to_string()]),
            thinking: Some(ThinkingConfig::enabled(1024)),
            stream: false,
            metadata: None,
            temperature: None,
            tools: None,
            tool_choice: None,
            top_p: None,
            top_k: None,
        };
        let resp = client.send(req).await?;
        if !matches!(resp.stop_reason, Some(StopReason::StopSequence)) {
            return Err(claudius::Error::unknown(
                "did not get a stop sequence".to_string(),
            ));
        }
        match resp.stop_sequence.as_deref() {
            Some("yes") => {
                success += 1;
            }
            Some("no") => {}
            Some(_) => {
                return Err(claudius::Error::unknown(
                    "expected yes/no stop sequence".to_string(),
                ));
            }
            None => {
                return Err(claudius::Error::unknown(
                    "expected stop sequence".to_string(),
                ));
            }
        }
    }
    Ok(success)
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct DecidableSemanticInjection {
    pub positives: Vec<String>,
    pub negatives: Vec<String>,
    pub text: String,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct InjectableAction {
    pub inject: String,
    pub action: serde_json::Value,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ConflictField {
    pub conflict_type: String,
    pub field_name: String,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct TestDataPoint {
    pub text: String,
    pub policies: Vec<Policy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conflicts: Option<Vec<ConflictField>>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Metrics {
    pub policyai_fields_matched: usize,
    pub baseline_fields_matched: usize,
    pub policyai_fields_with_wrong_value: usize,
    pub baseline_fields_with_wrong_value: usize,
    pub policyai_fields_missing: usize,
    pub baseline_fields_missing: usize,
    pub policyai_extra_fields: usize,
    pub baseline_extra_fields: usize,
    pub policyai_error: Option<String>,
    pub baseline_error: Option<String>,
    pub policyai_apply_duration_ms: u32,
    pub baseline_apply_duration_ms: u32,
    pub policyai_usage: Option<Usage>,
    pub baseline_usage: Option<Usage>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EvaluationReport {
    pub input: TestDataPoint,
    pub metrics: Metrics,
    pub output: serde_json::Value,
    pub baseline: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_injection_default() {
        let injection = SemanticInjection::default();
        assert!(injection.injections.is_empty());
        assert!(injection.rationales.is_empty());
        assert!(injection.text.is_empty());
    }

    #[test]
    fn semantic_injection_serialization() {
        let injection = SemanticInjection {
            injections: vec!["injection1".to_string(), "injection2".to_string()],
            rationales: vec!["rationale1".to_string()],
            text: "test text".to_string(),
        };

        let serialized = serde_json::to_string(&injection).unwrap();
        assert!(serialized.contains("injection1"));
        assert!(serialized.contains("rationale1"));
        assert!(serialized.contains("test text"));

        let deserialized: SemanticInjection = serde_json::from_str(&serialized).unwrap();
        assert_eq!(injection.injections, deserialized.injections);
        assert_eq!(injection.rationales, deserialized.rationales);
        assert_eq!(injection.text, deserialized.text);
    }

    #[test]
    fn semantic_injection_clone() {
        let original = SemanticInjection {
            injections: vec!["test".to_string()],
            rationales: vec!["reason".to_string()],
            text: "content".to_string(),
        };

        let cloned = original.clone();
        assert_eq!(original.injections, cloned.injections);
        assert_eq!(original.rationales, cloned.rationales);
        assert_eq!(original.text, cloned.text);
    }

    #[test]
    fn decidable_semantic_injection_default() {
        let decidable = DecidableSemanticInjection::default();
        assert!(decidable.positives.is_empty());
        assert!(decidable.negatives.is_empty());
        assert!(decidable.text.is_empty());
    }

    #[test]
    fn decidable_semantic_injection_serialization() {
        let decidable = DecidableSemanticInjection {
            positives: vec!["pos1".to_string(), "pos2".to_string()],
            negatives: vec!["neg1".to_string()],
            text: "test content".to_string(),
        };

        let serialized = serde_json::to_string(&decidable).unwrap();
        let deserialized: DecidableSemanticInjection = serde_json::from_str(&serialized).unwrap();

        assert_eq!(decidable.positives, deserialized.positives);
        assert_eq!(decidable.negatives, deserialized.negatives);
        assert_eq!(decidable.text, deserialized.text);
    }

    #[test]
    fn injectable_action_serialization() {
        let action = InjectableAction {
            inject: "if urgent then".to_string(),
            action: serde_json::json!({"priority": "high"}),
        };

        let serialized = serde_json::to_string(&action).unwrap();
        assert!(serialized.contains("if urgent then"));
        assert!(serialized.contains("priority"));
        assert!(serialized.contains("high"));

        let deserialized: InjectableAction = serde_json::from_str(&serialized).unwrap();
        assert_eq!(action.inject, deserialized.inject);
        assert_eq!(action.action, deserialized.action);
    }

    #[test]
    fn conflict_field_serialization() {
        let conflict = ConflictField {
            conflict_type: "agreement".to_string(),
            field_name: "status".to_string(),
        };

        let serialized = serde_json::to_string(&conflict).unwrap();
        assert!(serialized.contains("agreement"));
        assert!(serialized.contains("status"));

        let deserialized: ConflictField = serde_json::from_str(&serialized).unwrap();
        assert_eq!(conflict.conflict_type, deserialized.conflict_type);
        assert_eq!(conflict.field_name, deserialized.field_name);
    }

    #[test]
    fn test_data_point_minimal() {
        use crate::{Field, PolicyType};

        let policy_type = PolicyType {
            name: "TestPolicy".to_string(),
            fields: vec![Field::Bool {
                name: "enabled".to_string(),
                default: false,
                on_conflict: crate::OnConflict::Default,
            }],
        };

        let point = TestDataPoint {
            text: "test text".to_string(),
            policies: vec![Policy {
                r#type: policy_type,
                prompt: "test prompt".to_string(),
                action: serde_json::json!({"enabled": true}),
            }],
            expected: None,
            conflicts: None,
        };

        let serialized = serde_json::to_string(&point).unwrap();
        assert!(serialized.contains("test text"));
        assert!(serialized.contains("test prompt"));
        assert!(!serialized.contains("expected"));
        assert!(!serialized.contains("conflicts"));
    }

    #[test]
    fn test_data_point_with_expected() {
        use crate::{Field, PolicyType};

        let policy_type = PolicyType {
            name: "TestPolicy".to_string(),
            fields: vec![Field::String {
                name: "message".to_string(),
                default: None,
                on_conflict: crate::OnConflict::Agreement,
            }],
        };

        let point = TestDataPoint {
            text: "hello world".to_string(),
            policies: vec![Policy {
                r#type: policy_type,
                prompt: "greeting".to_string(),
                action: serde_json::json!({"message": "hello"}),
            }],
            expected: Some(serde_json::json!({"message": "hello"})),
            conflicts: None,
        };

        let serialized = serde_json::to_string(&point).unwrap();
        let deserialized: TestDataPoint = serde_json::from_str(&serialized).unwrap();

        assert_eq!(point.text, deserialized.text);
        assert_eq!(point.policies.len(), deserialized.policies.len());
        assert_eq!(point.expected, deserialized.expected);
    }

    #[test]
    fn test_data_point_with_conflicts() {
        use crate::{Field, PolicyType};

        let policy_type = PolicyType {
            name: "TestPolicy".to_string(),
            fields: vec![Field::Number {
                name: "count".to_string(),
                default: Some(crate::t64(0.0)),
                on_conflict: crate::OnConflict::LargestValue,
            }],
        };

        let point = TestDataPoint {
            text: "data".to_string(),
            policies: vec![
                Policy {
                    r#type: policy_type.clone(),
                    prompt: "first".to_string(),
                    action: serde_json::json!({"count": 10}),
                },
                Policy {
                    r#type: policy_type,
                    prompt: "second".to_string(),
                    action: serde_json::json!({"count": 20}),
                },
            ],
            expected: Some(serde_json::json!({"count": 20})),
            conflicts: Some(vec![ConflictField {
                conflict_type: "largest".to_string(),
                field_name: "count".to_string(),
            }]),
        };

        let serialized = serde_json::to_string(&point).unwrap();
        assert!(serialized.contains("conflicts"));

        let deserialized: TestDataPoint = serde_json::from_str(&serialized).unwrap();
        assert!(deserialized.conflicts.is_some());
        assert_eq!(deserialized.conflicts.unwrap().len(), 1);
    }

    #[test]
    fn semantic_injection_debug() {
        let injection = SemanticInjection {
            injections: vec!["test".to_string()],
            rationales: vec![],
            text: "text".to_string(),
        };

        let debug_str = format!("{injection:?}");
        assert!(debug_str.contains("SemanticInjection"));
        assert!(debug_str.contains("test"));
        assert!(debug_str.contains("text"));
    }

    #[test]
    fn injectable_action_clone() {
        let original = InjectableAction {
            inject: "condition".to_string(),
            action: serde_json::json!({"field": "value"}),
        };

        let cloned = original.clone();
        assert_eq!(original.inject, cloned.inject);
        assert_eq!(original.action, cloned.action);
    }

    #[test]
    fn conflict_field_clone() {
        let original = ConflictField {
            conflict_type: "default".to_string(),
            field_name: "field1".to_string(),
        };

        let cloned = original.clone();
        assert_eq!(original.conflict_type, cloned.conflict_type);
        assert_eq!(original.field_name, cloned.field_name);
    }
}
