use crate::PolicyType;

/// Represents a policy with its type definition, prompt, and resulting action.
///
/// A Policy is created by applying a semantic injection to a PolicyType,
/// resulting in structured actions that can be composed with other policies.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Policy {
    pub r#type: PolicyType,
    pub prompt: String,
    pub action: serde_json::Value,
}
