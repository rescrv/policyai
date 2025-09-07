use crate::PolicyType;

/// Represents a policy with its type definition, prompt, and resulting action.
///
/// A Policy is created by applying a semantic injection to a PolicyType,
/// resulting in structured actions that can be composed with other policies.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Policy {
    /// The type definition that defines the structure and constraints for this policy
    pub r#type: PolicyType,
    /// The natural language prompt that describes when and how this policy should be applied
    pub prompt: String,
    /// The structured action data that conforms to the policy type schema
    pub action: serde_json::Value,
}
