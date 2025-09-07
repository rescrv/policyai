/// Defines how to resolve conflicts when multiple policies set the same field.
///
/// When multiple policies attempt to set the same field to different values,
/// the conflict resolution strategy determines which value wins.
///
/// # Strategies
///
/// - `Default`: Use the field's default value, ignoring policy values
/// - `Agreement`: All policies must agree on the value, or a conflict is reported
/// - `LargestValue`: The largest value wins (true > false for bools, longer strings win, etc.)
///
/// # Example
///
/// ```ignore
/// use policyai::{Field, OnConflict};
///
/// let field = Field::StringEnum {
///     name: "priority".to_string(),
///     values: vec!["low".to_string(), "high".to_string()],
///     default: None,
///     on_conflict: OnConflict::LargestValue, // "high" would win over "low"
/// };
/// ```
#[derive(Copy, Clone, Default, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum OnConflict {
    /// Use the field's default value when conflicts occur
    #[default]
    #[serde(rename = "default")]
    Default,
    /// All policies must agree on the value
    #[serde(rename = "agreement")]
    Agreement,
    /// The largest value wins
    #[serde(rename = "largest")]
    LargestValue,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_conflict_default() {
        let conflict = OnConflict::default();
        assert_eq!(conflict, OnConflict::Default);
    }

    #[test]
    fn on_conflict_equality() {
        assert_eq!(OnConflict::Default, OnConflict::Default);
        assert_eq!(OnConflict::Agreement, OnConflict::Agreement);
        assert_eq!(OnConflict::LargestValue, OnConflict::LargestValue);
        assert_ne!(OnConflict::Default, OnConflict::Agreement);
        assert_ne!(OnConflict::Agreement, OnConflict::LargestValue);
        assert_ne!(OnConflict::Default, OnConflict::LargestValue);
    }

    #[test]
    fn on_conflict_copy() {
        let original = OnConflict::Agreement;
        let copied = original;
        assert_eq!(original, copied);
    }

    #[test]
    fn on_conflict_serialization() {
        let conflict = OnConflict::Default;
        let serialized = serde_json::to_string(&conflict).unwrap();
        assert_eq!(serialized, "\"default\"");
        let deserialized: OnConflict = serde_json::from_str(&serialized).unwrap();
        assert_eq!(conflict, deserialized);

        let conflict = OnConflict::Agreement;
        let serialized = serde_json::to_string(&conflict).unwrap();
        assert_eq!(serialized, "\"agreement\"");
        let deserialized: OnConflict = serde_json::from_str(&serialized).unwrap();
        assert_eq!(conflict, deserialized);

        let conflict = OnConflict::LargestValue;
        let serialized = serde_json::to_string(&conflict).unwrap();
        assert_eq!(serialized, "\"largest\"");
        let deserialized: OnConflict = serde_json::from_str(&serialized).unwrap();
        assert_eq!(conflict, deserialized);
    }

    #[test]
    fn on_conflict_debug() {
        assert_eq!(format!("{:?}", OnConflict::Default), "Default");
        assert_eq!(format!("{:?}", OnConflict::Agreement), "Agreement");
        assert_eq!(format!("{:?}", OnConflict::LargestValue), "LargestValue");
    }
}
