//! Data structures and utilities for test data generation and policy evaluation.
//!
//! This module provides functionality for semantic injection testing, policy evaluation,
//! and test data generation. It includes utilities for determining policy applicability
//! and structures for evaluation metrics and test data points.

use claudius::{
    Anthropic, CacheControlEphemeral, ContentBlock, KnownModel, MessageCreateParams, MessageParam,
    MessageParamContent, MessageRole, Model, StopReason, SystemPrompt, TextBlock, ThinkingConfig,
};

use crate::{Policy, Report, Usage};

/// A semantic injection with multiple candidate injections and their rationales.
///
/// This structure represents test data for evaluating semantic injections against text.
/// It contains multiple injection candidates, their supporting rationales, and the
/// original text they were derived from.
///
/// # Examples
///
/// ```
/// use policyai::data::SemanticInjection;
///
/// let injection = SemanticInjection {
///     injections: vec!["If urgent, set priority high".to_string()],
///     rationales: vec!["Urgent emails need immediate attention".to_string()],
///     text: "URGENT: Please respond ASAP".to_string(),
/// };
/// ```
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SemanticInjection {
    /// Candidate semantic injection texts that could be applied to the input text.
    pub injections: Vec<String>,
    /// Rationales explaining why each semantic injection might be applicable.
    pub rationales: Vec<String>,
    /// The original text that the semantic injections are designed to match against.
    pub text: String,
}

/// Determine if a policy applies to given text with statistical confidence.
///
/// This function tests whether a semantic injection policy applies to the provided text
/// by making multiple attempts and requiring at least `k` successes out of `n` total attempts.
/// Uses Claude to evaluate policy applicability with natural language understanding.
///
/// # Arguments
///
/// * `client` - The Anthropic client for making API calls
/// * `text` - The input text to evaluate against the policy
/// * `semantic_injection` - The policy's semantic injection rule
/// * `k` - Minimum number of successes required
/// * `n` - Maximum number of attempts to make
///
/// # Returns
///
/// Returns `true` if the policy applies with sufficient confidence (≥k successes),
/// `false` otherwise.
///
/// # Errors
///
/// Returns [`claudius::Error`] if the API call fails or returns unexpected responses.
///
/// # Examples
///
/// ```no_run
/// use claudius::Anthropic;
/// use policyai::data::policy_applies;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Anthropic::new(None)?;
/// let applies = policy_applies(
///     &client,
///     "This is urgent!",
///     "If text indicates urgency, mark as high priority",
///     3,  // Need 3 successes
///     5   // Out of 5 attempts
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub async fn policy_applies(
    client: &Anthropic,
    text: &str,
    semantic_injection: &str,
    k: usize,
    n: usize,
) -> Result<bool, claudius::Error> {
    Ok(apply_policy_fractional(client, text, semantic_injection, k, n).await? >= k)
}

/// Determine if a policy does NOT apply to given text with statistical confidence.
///
/// This function tests whether a semantic injection policy should NOT apply to the provided text
/// by making multiple attempts and requiring sufficiently few successes. The policy is considered
/// to not apply if the number of successes is ≤ n-k.
///
/// # Arguments
///
/// * `client` - The Anthropic client for making API calls
/// * `text` - The input text to evaluate against the policy
/// * `semantic_injection` - The policy's semantic injection rule
/// * `k` - Minimum number of successes that would indicate the policy applies
/// * `n` - Maximum number of attempts to make
///
/// # Returns
///
/// Returns `true` if the policy does not apply with sufficient confidence (≤n-k successes),
/// `false` otherwise.
///
/// # Errors
///
/// Returns [`claudius::Error`] if the API call fails or returns unexpected responses.
///
/// # Examples
///
/// ```no_run
/// use claudius::Anthropic;
/// use policyai::data::policy_does_not_apply;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Anthropic::new(None)?;
/// let does_not_apply = policy_does_not_apply(
///     &client,
///     "Regular email content",
///     "If text indicates urgency, mark as high priority",
///     3,  // Would need 3 successes to apply
///     5   // Out of 5 attempts
/// ).await?;
/// # Ok(())
/// # }
/// ```
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

/// A semantic injection test case with positive and negative examples.
///
/// This structure represents a semantic injection along with sets of text examples
/// that should and should not trigger the injection. Used for testing policy
/// decision boundaries and generating training data.
///
/// # Examples
///
/// ```
/// use policyai::data::DecidableSemanticInjection;
///
/// let decidable = DecidableSemanticInjection {
///     positives: vec![
///         "URGENT: Please respond immediately".to_string(),
///         "High priority - needs attention today".to_string(),
///     ],
///     negatives: vec![
///         "Regular update for your information".to_string(),
///         "Weekly newsletter".to_string(),
///     ],
///     text: "If urgent, set priority to high".to_string(),
/// };
/// ```
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct DecidableSemanticInjection {
    /// Text examples that should trigger this semantic injection.
    pub positives: Vec<String>,
    /// Text examples that should NOT trigger this semantic injection.
    pub negatives: Vec<String>,
    /// The semantic injection rule or description.
    pub text: String,
}

/// An injectable action that pairs a semantic condition with a structured output.
///
/// This structure represents a policy rule that can be injected into the system,
/// consisting of a natural language condition and a corresponding JSON action
/// to take when that condition is met.
///
/// # Examples
///
/// ```
/// use policyai::data::InjectableAction;
/// use serde_json::json;
///
/// let action = InjectableAction {
///     inject: "If the email is marked as urgent".to_string(),
///     action: json!({
///         "priority": "high",
///         "notify_immediately": true
///     }),
/// };
/// ```
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct InjectableAction {
    /// The natural language condition that determines when this action should be applied.
    pub inject: String,
    /// The structured JSON action to take when the injection condition is met.
    pub action: serde_json::Value,
}

/// Represents a field that experienced a conflict during policy application.
///
/// This structure tracks fields where multiple policies attempted to set different
/// values, requiring conflict resolution. Used for testing and debugging policy
/// interactions.
///
/// # Examples
///
/// ```
/// use policyai::data::ConflictField;
///
/// let conflict = ConflictField {
///     conflict_type: "agreement".to_string(),
///     field_name: "priority".to_string(),
/// };
/// ```
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct ConflictField {
    /// The type of conflict resolution that was applied (e.g., "agreement", "default").
    pub conflict_type: String,
    /// The name of the field that experienced the conflict.
    pub field_name: String,
}

/// A complete test case for policy evaluation.
///
/// This structure represents a single test case containing input text, the policies
/// to apply, expected output, and any expected conflicts. Used for systematic
/// testing of policy behavior and regression detection.
///
/// # Examples
///
/// ```
/// use policyai::data::{TestDataPoint, ConflictField};
/// use policyai::{Policy, PolicyType, Field, OnConflict};
/// use serde_json::json;
///
/// let policy_type = PolicyType {
///     name: "EmailPolicy".to_string(),
///     fields: vec![
///         Field::Bool {
///             name: "urgent".to_string(),
///             default: Some(false),
///             on_conflict: OnConflict::Default,
///         }
///     ],
/// };
///
/// let test_point = TestDataPoint {
///     text: "URGENT: Please respond immediately!".to_string(),
///     policies: vec![Policy {
///         r#type: policy_type,
///         prompt: "Mark urgent emails".to_string(),
///         action: json!({"urgent": true}),
///     }],
///     expected: Some(json!({"urgent": true})),
///     conflicts: None,
/// };
/// ```
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct TestDataPoint {
    /// The input text to process with the policies.
    pub text: String,
    /// The policies to apply to the input text.
    pub policies: Vec<Policy>,
    /// The expected structured output after applying all policies.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<serde_json::Value>,
    /// Expected conflicts that should occur during policy application.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conflicts: Option<Vec<ConflictField>>,
}

/// Performance and accuracy metrics for policy evaluation.
///
/// This structure tracks detailed metrics comparing PolicyAI performance
/// against a baseline system, including field-level accuracy, timing,
/// and resource usage statistics.
///
/// # Examples
///
/// ```
/// use policyai::data::Metrics;
/// use policyai::Usage;
///
/// let metrics = Metrics {
///     policyai_fields_matched: 8,
///     baseline_fields_matched: 6,
///     policyai_fields_with_wrong_value: 1,
///     baseline_fields_with_wrong_value: 2,
///     policyai_apply_duration_ms: 150,
///     baseline_apply_duration_ms: 300,
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Metrics {
    /// Number of fields where PolicyAI output exactly matched the expected value.
    pub policyai_fields_matched: usize,
    /// Number of fields where baseline output exactly matched the expected value.
    pub baseline_fields_matched: usize,
    /// Number of fields where PolicyAI provided a value but it was incorrect.
    pub policyai_fields_with_wrong_value: usize,
    /// Number of fields where baseline provided a value but it was incorrect.
    pub baseline_fields_with_wrong_value: usize,
    /// Number of expected fields that PolicyAI failed to provide.
    pub policyai_fields_missing: usize,
    /// Number of expected fields that baseline failed to provide.
    pub baseline_fields_missing: usize,
    /// Number of unexpected fields that PolicyAI provided.
    pub policyai_extra_fields: usize,
    /// Number of unexpected fields that baseline provided.
    pub baseline_extra_fields: usize,
    /// Error message if PolicyAI evaluation failed.
    pub policyai_error: Option<String>,
    /// Error message if baseline evaluation failed.
    pub baseline_error: Option<String>,
    /// Time in milliseconds taken by PolicyAI to process the input.
    pub policyai_apply_duration_ms: u32,
    /// Time in milliseconds taken by baseline to process the input.
    pub baseline_apply_duration_ms: u32,
    /// Token and API usage statistics for PolicyAI evaluation.
    pub policyai_usage: Option<Usage>,
    /// Token and API usage statistics for baseline evaluation.
    pub baseline_usage: Option<Usage>,
}

/// A complete evaluation report comparing PolicyAI performance against a baseline.
///
/// This structure contains all the information from a single evaluation run,
/// including the input test case, performance metrics, and both PolicyAI and
/// baseline outputs for comparison.
///
/// # Examples
///
/// ```
/// use policyai::data::{EvaluationReport, TestDataPoint, Metrics};
/// use policyai::Report;
/// use serde_json::json;
///
/// let report = EvaluationReport {
///     input: TestDataPoint {
///         text: "Test email".to_string(),
///         policies: vec![],
///         expected: None,
///         conflicts: None,
///     },
///     metrics: Metrics::default(),
///     report: Report::default(),
///     output: json!({"processed": true}),
///     baseline: Some(json!({"processed": false})),
/// };
/// ```
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EvaluationReport {
    /// The input test data point that was evaluated.
    pub input: TestDataPoint,
    /// Performance and accuracy metrics from the evaluation.
    pub metrics: Metrics,
    /// The report produced by PolicyAI.
    pub report: Report,
    /// The structured output produced by PolicyAI.
    pub output: serde_json::Value,
    /// The structured output produced by the baseline system, if available.
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
                default: Some(false),
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
