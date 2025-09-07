//! Error types for PolicyAI operations.
//!
//! This module defines the various error conditions that can occur during policy
//! processing, including policy validation errors, conflict resolution failures,
//! and LLM communication issues.

use crate::Field;

/// Errors that can occur when working with policies
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum PolicyError {
    /// Field configuration is inconsistent with policy requirements
    Inconsistent {
        /// The field that has an inconsistent configuration.
        field: Field,
    },
    /// Expected a boolean value but got something else
    ExpectedBool {
        /// Name of the field that was expected to be a boolean.
        field_name: String,
        /// The actual type that was provided instead.
        actual_type: String,
    },
    /// Expected a numeric value but got something else
    ExpectedNumber {
        /// Name of the field that was expected to be a number.
        field_name: String,
        /// The actual type that was provided instead.
        actual_type: String,
    },
    /// Expected a string value but got something else
    ExpectedString {
        /// Name of the field that was expected to be a string.
        field_name: String,
        /// The actual type that was provided instead.
        actual_type: String,
    },
    /// Default values conflict between policies
    DefaultConflict {
        /// Name of the field with conflicting defaults.
        field: String,
        /// The existing default value.
        existing: serde_json::Value,
        /// The new default value that conflicts.
        new: serde_json::Value,
        /// Suggested resolution for the conflict.
        suggestion: String,
    },
    /// Internal invariant was violated
    InvariantViolation {
        /// Source file where the violation occurred.
        file: String,
        /// Line number where the violation occurred.
        line: u32,
        /// Description of the invariant violation.
        message: String,
    },
    /// Type checking failed
    TypeCheckFailure {
        /// Source file where the type check failed.
        file: String,
        /// Line number where the type check failed.
        line: u32,
        /// Description of the type check failure.
        message: String,
        /// The expected type.
        expected: String,
        /// The actual type that was found.
        actual: String,
    },
}

impl PolicyError {
    /// Create an ExpectedBool error with type information
    pub fn expected_bool(field_name: impl Into<String>, actual_value: &serde_json::Value) -> Self {
        let actual_type = match actual_value {
            serde_json::Value::Null => "null",
            serde_json::Value::Bool(_) => "bool",
            serde_json::Value::Number(_) => "number",
            serde_json::Value::String(_) => "string",
            serde_json::Value::Array(_) => "array",
            serde_json::Value::Object(_) => "object",
        };
        Self::ExpectedBool {
            field_name: field_name.into(),
            actual_type: actual_type.to_string(),
        }
    }

    /// Create an ExpectedNumber error with type information
    pub fn expected_number(
        field_name: impl Into<String>,
        actual_value: &serde_json::Value,
    ) -> Self {
        let actual_type = match actual_value {
            serde_json::Value::Null => "null",
            serde_json::Value::Bool(_) => "bool",
            serde_json::Value::Number(_) => "number",
            serde_json::Value::String(_) => "string",
            serde_json::Value::Array(_) => "array",
            serde_json::Value::Object(_) => "object",
        };
        Self::ExpectedNumber {
            field_name: field_name.into(),
            actual_type: actual_type.to_string(),
        }
    }

    /// Create an ExpectedString error with type information
    pub fn expected_string(
        field_name: impl Into<String>,
        actual_value: &serde_json::Value,
    ) -> Self {
        let actual_type = match actual_value {
            serde_json::Value::Null => "null",
            serde_json::Value::Bool(_) => "bool",
            serde_json::Value::Number(_) => "number",
            serde_json::Value::String(_) => "string",
            serde_json::Value::Array(_) => "array",
            serde_json::Value::Object(_) => "object",
        };
        Self::ExpectedString {
            field_name: field_name.into(),
            actual_type: actual_type.to_string(),
        }
    }
}

impl std::fmt::Display for PolicyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolicyError::Inconsistent { field } => {
                write!(f, "Inconsistent field configuration: {field}\nSuggestion: Check that the field type and conflict resolution strategy are compatible")
            }
            PolicyError::ExpectedBool {
                field_name,
                actual_type,
            } => {
                write!(f, "Type mismatch for field '{field_name}': expected boolean value but got {actual_type}\nSuggestion: Ensure the policy action provides a boolean value for this field")
            }
            PolicyError::ExpectedNumber {
                field_name,
                actual_type,
            } => {
                write!(f, "Type mismatch for field '{field_name}': expected numeric value but got {actual_type}\nSuggestion: Ensure the policy action provides a number for this field")
            }
            PolicyError::ExpectedString {
                field_name,
                actual_type,
            } => {
                write!(f, "Type mismatch for field '{field_name}': expected string value but got {actual_type}\nSuggestion: Ensure the policy action provides a string for this field")
            }
            PolicyError::DefaultConflict {
                field,
                existing,
                new,
                suggestion,
            } => {
                write!(
                    f,
                    "Default value conflict for field '{field}':\n  Existing: {existing}\n  New: {new}\nSuggestion: {suggestion}"
                )
            }
            PolicyError::InvariantViolation {
                file,
                line,
                message,
            } => {
                write!(f, "Internal error at {file}:{line}: {message}\nThis is likely a bug in PolicyAI. Please report it at https://github.com/rescrv/policyai/issues")
            }
            PolicyError::TypeCheckFailure {
                file,
                line,
                message,
                expected,
                actual,
            } => {
                write!(f, "Type check failure at {file}:{line}: {message}\n  Expected: {expected}\n  Actual: {actual}\nSuggestion: Verify that your policy actions match the policy type definition")
            }
        }
    }
}

impl std::error::Error for PolicyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

///////////////////////////////////////////// Conflict /////////////////////////////////////////////

/// Represents conflicts that occur when multiple policies attempt to set different values for the same field.
///
/// This enum captures the specific type of conflict and the conflicting values, enabling
/// proper conflict resolution strategies and detailed error reporting.
///
/// # Examples
///
/// ```
/// use policyai::Conflict;
/// use serde_json::json;
///
/// let conflict = Conflict::BoolConflict {
///     field: "urgent".to_string(),
///     val1: true,
///     val2: false,
/// };
/// ```
#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub enum Conflict {
    /// Conflict between two boolean values for the same field.
    BoolConflict {
        /// Name of the field experiencing the conflict.
        field: String,
        /// First boolean value from one policy.
        val1: bool,
        /// Second boolean value from another policy.
        val2: bool,
    },
    /// Conflict between two numeric values for the same field.
    NumberConflict {
        /// Name of the field experiencing the conflict.
        field: String,
        /// First numeric value from one policy.
        val1: serde_json::Number,
        /// Second numeric value from another policy.
        val2: serde_json::Number,
    },
    /// Conflict between two string values for the same field.
    StringConflict {
        /// Name of the field experiencing the conflict.
        field: String,
        /// First string value from one policy.
        val1: String,
        /// Second string value from another policy.
        val2: String,
    },
    /// Generic disagreement between two values that cannot be reconciled.
    Disagree {
        /// Name of the field experiencing the disagreement.
        name: String,
        /// First value from one policy.
        value1: serde_json::Value,
        /// Second value from another policy.
        value2: serde_json::Value,
    },
}

//////////////////////////////////////////// ApplyError ////////////////////////////////////////////

/// Errors that can occur when applying policies to unstructured data
#[derive(Debug)]
pub enum ApplyError {
    /// A policy-specific error occurred
    Policy(PolicyError),
    /// An error occurred while communicating with the LLM
    Claudius(claudius::Error),
    /// Policies have conflicting values that cannot be resolved
    Conflict(Conflict),
    /// Too many retry attempts were made to resolve inconsistencies
    TooManyIterations {
        /// Number of attempts that were made.
        attempts: usize,
        /// The error from the final attempt.
        last_error: String,
    },
    /// The LLM response was invalid or unexpected
    InvalidResponse {
        /// Description of what made the response invalid.
        message: String,
        /// Suggested action to resolve the issue.
        suggestion: String,
    },
}

impl ApplyError {
    /// Create a TooManyIterations error with context
    pub fn too_many_iterations(attempts: usize, last_error: impl Into<String>) -> Self {
        Self::TooManyIterations {
            attempts,
            last_error: last_error.into(),
        }
    }

    /// Create an InvalidResponse error with a suggestion
    pub fn invalid_response(message: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self::InvalidResponse {
            message: message.into(),
            suggestion: suggestion.into(),
        }
    }
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApplyError::Policy(err) => write!(f, "Policy error: {err}"),
            ApplyError::Claudius(err) => write!(f, "LLM communication error: {err}"),
            ApplyError::Conflict(conflict) => write!(f, "Policy conflict: {conflict:?}\nSuggestion: Review your policies for conflicting rules and adjust their conflict resolution strategies"),
            ApplyError::TooManyIterations { attempts, last_error } => {
                write!(f, "Failed to apply policies after {attempts} attempts\nLast error: {last_error}\nSuggestion: Simplify your policies or check for contradictory rules")
            }
            ApplyError::InvalidResponse { message, suggestion } => {
                write!(f, "Invalid LLM response: {message}\nSuggestion: {suggestion}")
            }
        }
    }
}

impl std::error::Error for ApplyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ApplyError::Policy(err) => Some(err),
            ApplyError::Claudius(err) => Some(err),
            _ => None,
        }
    }
}

impl From<PolicyError> for ApplyError {
    fn from(err: PolicyError) -> Self {
        Self::Policy(err)
    }
}

impl<T: Into<claudius::Error>> From<T> for ApplyError {
    fn from(err: T) -> Self {
        Self::Claudius(err.into())
    }
}
