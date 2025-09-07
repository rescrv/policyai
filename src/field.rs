//! Field definitions for PolicyAI types.
//!
//! This module defines the [`Field`] enum which represents the various types of fields
//! that can be included in a PolicyType. Each field has a name, type, optional default value,
//! and conflict resolution strategy.

use crate::{t64, OnConflict};

/// Represents a field in a PolicyType with its type, default value, and conflict resolution strategy.
///
/// Fields define the structure of data that policies work with. Each field has:
/// - A name that identifies it
/// - A type (bool, number, string, string enum, or string array)
/// - An optional default value
/// - A conflict resolution strategy for when multiple policies set the same field
///
/// # Example
///
/// ```
/// use policyai::{Field, OnConflict};
///
/// let field = Field::Bool {
///     name: "is_active".to_string(),
///     default: true,
///     on_conflict: OnConflict::Default,
/// };
/// ```
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Field {
    /// A boolean field that can be either true or false.
    #[serde(rename = "bool")]
    Bool {
        /// The name of this field.
        name: String,
        /// The default boolean value when no policy sets this field.
        default: bool,
        /// Strategy for resolving conflicts when multiple policies set this field.
        on_conflict: OnConflict,
    },
    /// A free-form string field.
    #[serde(rename = "string")]
    String {
        /// The name of this field.
        name: String,
        /// The default string value when no policy sets this field.
        default: Option<String>,
        /// Strategy for resolving conflicts when multiple policies set this field.
        on_conflict: OnConflict,
    },
    /// A string field constrained to a specific set of allowed values.
    #[serde(rename = "enum")]
    StringEnum {
        /// The name of this field.
        name: String,
        /// The allowed values for this field.
        values: Vec<String>,
        /// The default value when no policy sets this field.
        default: Option<String>,
        /// Strategy for resolving conflicts when multiple policies set this field.
        on_conflict: OnConflict,
    },
    /// An array of strings that policies can append to.
    #[serde(rename = "array")]
    StringArray {
        /// The name of this field.
        name: String,
    },
    /// A numeric field that can hold integer or floating-point values.
    #[serde(rename = "number")]
    Number {
        /// The name of this field.
        name: String,
        /// The default numeric value when no policy sets this field.
        default: Option<t64>,
        /// Strategy for resolving conflicts when multiple policies set this field.
        on_conflict: OnConflict,
    },
}

impl Field {
    /// Get the name of this field.
    pub fn name(&self) -> &str {
        match self {
            Self::Bool {
                name,
                default: _,
                on_conflict: _,
            } => name,
            Self::Number {
                name,
                default: _,
                on_conflict: _,
            } => name,
            Self::String {
                name,
                default: _,
                on_conflict: _,
            } => name,
            Self::StringEnum {
                name,
                values: _,
                default: _,
                on_conflict: _,
            } => name,
            Self::StringArray { name } => name,
        }
    }

    /// Get the default value for this field.
    ///
    /// Returns the configured default value, or null for fields without defaults.
    /// String arrays always default to an empty array.
    pub fn default_value(&self) -> serde_json::Value {
        match self {
            Self::Bool {
                name: _,
                default,
                on_conflict: _,
            } => (*default).into(),
            Self::Number {
                name: _,
                default,
                on_conflict: _,
            } => (*default).into(),
            Self::String {
                name: _,
                default,
                on_conflict: _,
            } => (*default).clone().into(),
            Self::StringEnum {
                name: _,
                values: _,
                default,
                on_conflict: _,
            } => (*default).clone().into(),
            Self::StringArray { name: _ } => serde_json::json! {[]},
        }
    }
}

impl std::fmt::Display for Field {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            Self::Bool {
                name,
                default,
                on_conflict,
            } => match on_conflict {
                OnConflict::Default => {
                    if *default {
                        write!(f, "{name}: bool = true")?;
                    } else {
                        write!(f, "{name}: bool")?;
                    }
                }
                OnConflict::Agreement => {
                    if *default {
                        write!(f, "{name}: bool @ agreement = true")?;
                    } else {
                        write!(f, "{name}: bool @ agreement")?;
                    }
                }
                OnConflict::LargestValue => {
                    if *default {
                        write!(f, "{name}: bool @ sticky = true")?;
                    } else {
                        write!(f, "{name}: bool @ sticky")?;
                    }
                }
            },
            Self::String {
                name,
                default,
                on_conflict,
            } => match on_conflict {
                OnConflict::Default => {
                    if let Some(default) = default.as_ref() {
                        write!(f, "{name}: string = {default:?}")?;
                    } else {
                        write!(f, "{name}: string")?;
                    }
                }
                OnConflict::Agreement => {
                    if let Some(default) = default.as_ref() {
                        write!(f, "{name}: string @ agreement = {default:?}")?;
                    } else {
                        write!(f, "{name}: string @ agreement")?;
                    }
                }
                OnConflict::LargestValue => {
                    if let Some(default) = default.as_ref() {
                        write!(f, "{name}: string @ last wins = {default:?}")?;
                    } else {
                        write!(f, "{name}: string @ last wins")?;
                    }
                }
            },
            Self::StringEnum {
                name,
                values,
                default,
                on_conflict,
            } => {
                let values = values
                    .iter()
                    .map(|v| format!("{v:?}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                match on_conflict {
                    OnConflict::Default => {
                        if let Some(default) = default.as_ref() {
                            write!(f, "{name}: [{values}] = {default:?}")?;
                        } else {
                            write!(f, "{name}: [{values}]")?;
                        }
                    }
                    OnConflict::Agreement => {
                        if let Some(default) = default.as_ref() {
                            write!(f, "{name}: [{values}] @ agreement = {default:?}")?;
                        } else {
                            write!(f, "{name}: [{values}] @ agreement")?;
                        }
                    }
                    OnConflict::LargestValue => {
                        if let Some(default) = default.as_ref() {
                            write!(f, "{name}: [{values}] @ highest wins = {default:?}")?;
                        } else {
                            write!(f, "{name}: [{values}] @ highest wins")?;
                        }
                    }
                }
            }
            Self::StringArray { name } => {
                write!(f, "{name}: [string]")?;
            }
            Self::Number {
                name,
                default,
                on_conflict,
            } => match on_conflict {
                OnConflict::Default => {
                    if let Some(default) = default.as_ref() {
                        write!(f, "{name}: number = {}", default.0)?;
                    } else {
                        write!(f, "{name}: number")?;
                    }
                }
                OnConflict::Agreement => {
                    if let Some(default) = default.as_ref() {
                        write!(f, "{name}: number @ agreement = {}", default.0)?;
                    } else {
                        write!(f, "{name}: number @ agreement")?;
                    }
                }
                OnConflict::LargestValue => {
                    if let Some(default) = default.as_ref() {
                        write!(f, "{name}: number @ last wins = {}", default.0)?;
                    } else {
                        write!(f, "{name}: number @ last wins")?;
                    }
                }
            },
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_name() {
        let bool_field = Field::Bool {
            name: "is_active".to_string(),
            default: true,
            on_conflict: OnConflict::Default,
        };
        assert_eq!(bool_field.name(), "is_active");

        let string_field = Field::String {
            name: "description".to_string(),
            default: Some("test".to_string()),
            on_conflict: OnConflict::Agreement,
        };
        assert_eq!(string_field.name(), "description");

        let enum_field = Field::StringEnum {
            name: "priority".to_string(),
            values: vec!["low".to_string(), "high".to_string()],
            default: None,
            on_conflict: OnConflict::LargestValue,
        };
        assert_eq!(enum_field.name(), "priority");

        let array_field = Field::StringArray {
            name: "tags".to_string(),
        };
        assert_eq!(array_field.name(), "tags");

        let number_field = Field::Number {
            name: "score".to_string(),
            default: Some(t64(42.0)),
            on_conflict: OnConflict::Default,
        };
        assert_eq!(number_field.name(), "score");
    }

    #[test]
    fn field_default_value() {
        let bool_field = Field::Bool {
            name: "is_active".to_string(),
            default: true,
            on_conflict: OnConflict::Default,
        };
        assert_eq!(bool_field.default_value(), serde_json::json!(true));

        let string_field = Field::String {
            name: "description".to_string(),
            default: Some("test".to_string()),
            on_conflict: OnConflict::Agreement,
        };
        assert_eq!(string_field.default_value(), serde_json::json!("test"));

        let string_field_none = Field::String {
            name: "description".to_string(),
            default: None,
            on_conflict: OnConflict::Agreement,
        };
        assert_eq!(string_field_none.default_value(), serde_json::json!(null));

        let enum_field = Field::StringEnum {
            name: "priority".to_string(),
            values: vec!["low".to_string(), "high".to_string()],
            default: Some("low".to_string()),
            on_conflict: OnConflict::LargestValue,
        };
        assert_eq!(enum_field.default_value(), serde_json::json!("low"));

        let array_field = Field::StringArray {
            name: "tags".to_string(),
        };
        assert_eq!(array_field.default_value(), serde_json::json!([]));

        let number_field = Field::Number {
            name: "score".to_string(),
            default: Some(t64(42.5)),
            on_conflict: OnConflict::Default,
        };
        assert_eq!(number_field.default_value(), serde_json::json!(42.5));
    }

    #[test]
    fn field_display_bool() {
        let field = Field::Bool {
            name: "is_active".to_string(),
            default: true,
            on_conflict: OnConflict::Default,
        };
        assert_eq!(field.to_string(), "is_active: bool = true");

        let field = Field::Bool {
            name: "is_active".to_string(),
            default: false,
            on_conflict: OnConflict::Default,
        };
        assert_eq!(field.to_string(), "is_active: bool");

        let field = Field::Bool {
            name: "is_active".to_string(),
            default: true,
            on_conflict: OnConflict::Agreement,
        };
        assert_eq!(field.to_string(), "is_active: bool @ agreement = true");

        let field = Field::Bool {
            name: "is_active".to_string(),
            default: false,
            on_conflict: OnConflict::LargestValue,
        };
        assert_eq!(field.to_string(), "is_active: bool @ sticky");
    }

    #[test]
    fn field_display_string() {
        let field = Field::String {
            name: "description".to_string(),
            default: Some("default text".to_string()),
            on_conflict: OnConflict::Default,
        };
        assert_eq!(field.to_string(), "description: string = \"default text\"");

        let field = Field::String {
            name: "description".to_string(),
            default: None,
            on_conflict: OnConflict::Agreement,
        };
        assert_eq!(field.to_string(), "description: string @ agreement");

        let field = Field::String {
            name: "description".to_string(),
            default: Some("test".to_string()),
            on_conflict: OnConflict::LargestValue,
        };
        assert_eq!(
            field.to_string(),
            "description: string @ last wins = \"test\""
        );
    }

    #[test]
    fn field_display_string_enum() {
        let field = Field::StringEnum {
            name: "priority".to_string(),
            values: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
            default: Some("medium".to_string()),
            on_conflict: OnConflict::Default,
        };
        assert_eq!(
            field.to_string(),
            "priority: [\"low\", \"medium\", \"high\"] = \"medium\""
        );

        let field = Field::StringEnum {
            name: "priority".to_string(),
            values: vec!["low".to_string(), "high".to_string()],
            default: None,
            on_conflict: OnConflict::LargestValue,
        };
        assert_eq!(
            field.to_string(),
            "priority: [\"low\", \"high\"] @ highest wins"
        );
    }

    #[test]
    fn field_display_string_array() {
        let field = Field::StringArray {
            name: "tags".to_string(),
        };
        assert_eq!(field.to_string(), "tags: [string]");
    }

    #[test]
    fn field_display_number() {
        let field = Field::Number {
            name: "score".to_string(),
            default: Some(t64(42.5)),
            on_conflict: OnConflict::Default,
        };
        assert_eq!(field.to_string(), "score: number = 42.5");

        let field = Field::Number {
            name: "score".to_string(),
            default: None,
            on_conflict: OnConflict::Agreement,
        };
        assert_eq!(field.to_string(), "score: number @ agreement");
    }

    #[test]
    fn field_serialization() {
        let field = Field::Bool {
            name: "is_active".to_string(),
            default: true,
            on_conflict: OnConflict::Default,
        };
        let serialized = serde_json::to_string(&field).unwrap();
        let deserialized: Field = serde_json::from_str(&serialized).unwrap();
        assert_eq!(field, deserialized);

        let field = Field::StringArray {
            name: "tags".to_string(),
        };
        let serialized = serde_json::to_string(&field).unwrap();
        let deserialized: Field = serde_json::from_str(&serialized).unwrap();
        assert_eq!(field, deserialized);
    }
}
