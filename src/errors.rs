use crate::Field;

/// Errors that can occur when working with policies
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum PolicyError {
    /// Field configuration is inconsistent with policy requirements
    Inconsistent { field: Field },
    /// Expected a boolean value but got something else
    ExpectedBool {
        field_name: String,
        actual_type: String,
    },
    /// Expected a numeric value but got something else
    ExpectedNumber {
        field_name: String,
        actual_type: String,
    },
    /// Expected a string value but got something else
    ExpectedString {
        field_name: String,
        actual_type: String,
    },
    /// Default values conflict between policies
    DefaultConflict {
        field: String,
        existing: serde_json::Value,
        new: serde_json::Value,
        suggestion: String,
    },
    /// Internal invariant was violated
    InvariantViolation {
        file: String,
        line: u32,
        message: String,
    },
    /// Type checking failed
    TypeCheckFailure {
        file: String,
        line: u32,
        message: String,
        expected: String,
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

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub enum Conflict {
    BoolConflict {
        field: String,
        val1: bool,
        val2: bool,
    },
    NumberConflict {
        field: String,
        val1: serde_json::Number,
        val2: serde_json::Number,
    },
    StringConflict {
        field: String,
        val1: String,
        val2: String,
    },
    Disagree {
        name: String,
        value1: serde_json::Value,
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
    TooManyIterations { attempts: usize, last_error: String },
    /// The LLM response was invalid or unexpected
    InvalidResponse { message: String, suggestion: String },
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
