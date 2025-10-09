#![deny(missing_docs)]

//! PolicyAI: A framework for turning unstructured data into structured data via composable policies.
//!
//! PolicyAI provides a mechanism for writing policies that transform unstructured text into
//! structured outputs. Policies are composable, meaning multiple policies can be applied together
//! with configurable conflict resolution strategies.
//!
//! # Core Concepts
//!
//! - **PolicyType**: Defines the structure of data that policies will work with
//! - **Policy**: A semantic injection coupled with structured actions
//! - **Field**: A typed field in a policy with default values and conflict resolution
//! - **Manager**: Coordinates the application of multiple policies to unstructured data
//! - **Report**: The result of applying policies, including the structured output
//!
//! # Example
//!
//! ```
//! use policyai::{PolicyType, Field, OnConflict, Manager};
//!
//! let policy_type = PolicyType {
//!     name: "EmailPolicy".to_string(),
//!     fields: vec![
//!         Field::Bool {
//!             name: "unread".to_string(),
//!             default: Some(true),
//!             on_conflict: OnConflict::Default,
//!         },
//!         Field::StringEnum {
//!             name: "priority".to_string(),
//!             values: vec!["low".to_string(), "high".to_string()],
//!             default: None,
//!             on_conflict: OnConflict::LargestValue,
//!         },
//!     ],
//! };
//! ```

use std::cmp::Ordering;

/// Data structures and utilities for test data
pub mod data;

/// Analysis tools for evaluation metrics
pub mod analysis;

mod errors;
mod field;
mod manager;
mod masks;
mod on_conflict;
mod parser;
mod policy;
mod policy_type;
mod report;
mod report_builder;
mod usage;

pub use errors::{ApplyError, Conflict, PolicyError};
pub use field::Field;
pub use manager::Manager;
pub use masks::{BoolMask, NumberMask, StringArrayMask, StringEnumMask, StringMask};
pub use on_conflict::OnConflict;
pub use parser::ParseError;
pub use policy::Policy;
pub use policy_type::PolicyType;
pub use report::Report;
pub use report_builder::ReportBuilder;
pub use usage::Usage;

//////////////////////////////////////////////// t64 ///////////////////////////////////////////////

/// A totally-ordered 64-bit floating point number.
///
/// This type implements `Ord` and `Eq` for f64 values by using total ordering,
/// which means NaN values are considered equal to themselves and greater than
/// all other values, including positive infinity.
#[derive(Clone, Copy, Debug, Default, serde::Deserialize, serde::Serialize)]
#[allow(non_camel_case_types)]
#[repr(transparent)]
pub struct t64(pub f64);

impl Eq for t64 {}

impl PartialEq for t64 {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Ord for t64 {
    fn cmp(&self, other: &Self) -> Ordering {
        f64::total_cmp(&self.0, &other.0)
    }
}

impl PartialOrd for t64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<t64> for serde_json::Value {
    fn from(x: t64) -> Self {
        serde_json::Number::from_f64(x.0)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)
    }
}

//////////////////////////////////////////// Number Helpers ///////////////////////////////////////

pub(crate) fn number_is_equal(lhs: &serde_json::Number, rhs: &serde_json::Number) -> bool {
    if lhs.is_f64() && rhs.is_f64() {
        lhs.as_f64() == rhs.as_f64()
    } else if lhs.is_u64() && rhs.is_u64() {
        lhs.as_u64() == rhs.as_u64()
    } else if lhs.is_i64() && rhs.is_i64() {
        lhs.as_i64() == rhs.as_i64()
    } else {
        // Compare across different number types by converting to f64
        match (lhs.as_f64(), rhs.as_f64()) {
            (Some(l), Some(r)) => l == r,
            _ => false,
        }
    }
}

pub(crate) fn number_less_than(lhs: &serde_json::Number, rhs: &serde_json::Number) -> bool {
    if lhs.is_f64() && rhs.is_f64() {
        lhs.as_f64() < rhs.as_f64()
    } else if lhs.is_u64() && rhs.is_u64() {
        lhs.as_u64() < rhs.as_u64()
    } else if lhs.is_i64() && rhs.is_i64() {
        lhs.as_i64() < rhs.as_i64()
    } else {
        // Compare across different number types by converting to f64
        match (lhs.as_f64(), rhs.as_f64()) {
            (Some(l), Some(r)) => l < r,
            _ => false,
        }
    }
}

/////////////////////////////////////////////// tests //////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use claudius::{Anthropic, MessageCreateParams};

    use super::*;

    #[test]
    fn t64_equality() {
        assert_eq!(t64(1.0), t64(1.0));
        assert_ne!(t64(1.0), t64(2.0));
        assert_eq!(t64(f64::NAN), t64(f64::NAN)); // NaN equals itself in total ordering
    }

    #[test]
    fn t64_ordering() {
        assert!(t64(1.0) < t64(2.0));
        assert!(t64(2.0) > t64(1.0));
        assert!(t64(1.0) <= t64(1.0));
        assert!(t64(1.0) >= t64(1.0));

        // Test NaN handling
        assert!(t64(f64::NEG_INFINITY) < t64(f64::NAN));
        assert!(t64(f64::NAN) > t64(f64::INFINITY));
    }

    #[test]
    fn t64_serialization() {
        let value = t64(42.5);
        let serialized = serde_json::to_string(&value).unwrap();
        assert_eq!(serialized, "42.5");
        let deserialized: t64 = serde_json::from_str(&serialized).unwrap();
        assert_eq!(value, deserialized);
    }

    #[test]
    fn t64_to_json_value() {
        let value = t64(3.25);
        let json_value: serde_json::Value = value.into();
        assert_eq!(json_value, serde_json::json!(3.25));
    }

    #[test]
    fn t64_whole_number_serialization() {
        let value = t64(42.0);
        let json_value: serde_json::Value = value.into();
        let serialized = serde_json::to_string(&json_value).unwrap();
        let deserialized: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        let as_t64: t64 = serde_json::from_value(deserialized).unwrap();
        assert_eq!(value, as_t64);
    }

    #[test]
    fn t64_integer_deserialization() {
        let json_str = "42";
        let value: t64 = serde_json::from_str(json_str).unwrap();
        assert_eq!(value, t64(42.0));
    }

    #[test]
    fn number_is_equal() {
        let n1 = serde_json::Number::from(42);
        let n2 = serde_json::Number::from(42);
        assert!(super::number_is_equal(&n1, &n2));

        let n1 = serde_json::Number::from_f64(3.25).unwrap();
        let n2 = serde_json::Number::from_f64(3.25).unwrap();
        assert!(super::number_is_equal(&n1, &n2));

        let n1 = serde_json::Number::from(42);
        let n2 = serde_json::Number::from(43);
        assert!(!super::number_is_equal(&n1, &n2));
    }

    #[test]
    fn number_less_than() {
        let n1 = serde_json::Number::from(41);
        let n2 = serde_json::Number::from(42);
        assert!(super::number_less_than(&n1, &n2));
        assert!(!super::number_less_than(&n2, &n1));

        let n1 = serde_json::Number::from_f64(3.24).unwrap();
        let n2 = serde_json::Number::from_f64(3.25).unwrap();
        assert!(super::number_less_than(&n1, &n2));
        assert!(!super::number_less_than(&n2, &n1));
    }

    #[test]
    fn readme() {
        let policy = PolicyType {
            name: "policyai::EmailPolicy".to_string(),
            fields: vec![
                Field::Bool {
                    name: "unread".to_string(),
                    default: Some(true),
                    on_conflict: OnConflict::Default,
                },
                Field::StringEnum {
                    name: "priority".to_string(),
                    values: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
                    default: None,
                    on_conflict: OnConflict::LargestValue,
                },
                Field::StringEnum {
                    name: "category".to_string(),
                    values: vec![
                        "ai".to_string(),
                        "distributed systems".to_string(),
                        "other".to_string(),
                    ],
                    default: Some("other".to_string()),
                    on_conflict: OnConflict::Agreement,
                },
                Field::String {
                    name: "template".to_string(),
                    default: None,
                    on_conflict: OnConflict::Agreement,
                },
                Field::StringArray {
                    name: "labels".to_string(),
                },
            ],
        };
        assert_eq!(
            r#"type policyai::EmailPolicy {
    unread: bool = true,
    priority: ["low", "medium", "high"] @ highest wins,
    category: ["ai", "distributed systems", "other"] @ agreement = "other",
    template: string @ agreement,
    labels: [string],
}"#,
            format!("{policy}")
        );
    }

    #[tokio::test]
    async fn with_semantic_injection() {
        let client = Anthropic::new(None).unwrap();
        let policy = PolicyType {
            name: "policyai::EmailPolicy".to_string(),
            fields: vec![
                Field::Bool {
                    name: "unread".to_string(),
                    default: Some(true),
                    on_conflict: OnConflict::Default,
                },
                Field::StringEnum {
                    name: "priority".to_string(),
                    values: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
                    default: None,
                    on_conflict: OnConflict::LargestValue,
                },
                Field::StringEnum {
                    name: "category".to_string(),
                    values: vec![
                        "ai".to_string(),
                        "distributed systems".to_string(),
                        "other".to_string(),
                    ],
                    default: Some("other".to_string()),
                    on_conflict: OnConflict::Agreement,
                },
                Field::String {
                    name: "template".to_string(),
                    default: None,
                    on_conflict: OnConflict::Agreement,
                },
                Field::StringArray {
                    name: "labels".to_string(),
                },
            ],
        };
        let policy = policy
            .with_semantic_injection(
                &client,
                "If the user talks about Paxos, set \"category\" to \"distributed systems\".",
            )
            .await
            .unwrap();
        assert_eq!(
            serde_json::json! {{
                "category": "distributed systems",
            }},
            policy.action,
        );
    }

    #[tokio::test]
    async fn numeric_semantic_injection() {
        let client = Anthropic::new(None).unwrap();
        let policy = PolicyType {
            name: "policyai::EmailPolicy".to_string(),
            fields: vec![Field::Number {
                name: "weight".to_string(),
                default: None,
                on_conflict: OnConflict::Default,
            }],
        };
        let policy = policy
            .with_semantic_injection(&client, "Assign weight to the email.")
            .await
            .unwrap();
        assert!(matches!(
            policy.action.get("weight"),
            Some(serde_json::Value::Number(_))
        ));
    }

    #[tokio::test]
    async fn apply_readme_policy() {
        let client = Anthropic::new(None).unwrap();
        let policy = PolicyType {
            name: "policyai::EmailPolicy".to_string(),
            fields: vec![
                Field::Bool {
                    name: "unread".to_string(),
                    default: Some(true),
                    on_conflict: OnConflict::Default,
                },
                Field::StringEnum {
                    name: "priority".to_string(),
                    values: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
                    default: None,
                    on_conflict: OnConflict::LargestValue,
                },
                Field::String {
                    name: "template".to_string(),
                    default: None,
                    on_conflict: OnConflict::Agreement,
                },
                Field::StringEnum {
                    name: "category".to_string(),
                    values: vec![
                        "ai".to_string(),
                        "distributed systems".to_string(),
                        "other".to_string(),
                    ],
                    default: Some("other".to_string()),
                    on_conflict: OnConflict::Agreement,
                },
                Field::StringArray {
                    name: "labels".to_string(),
                },
            ],
        };
        let policy = policy
            .with_semantic_injection(
                &client,
                "When the email is about AI:  Set \"priority\" to \"low\" and \"unread\" to \"true\".",
            )
            .await
            .unwrap();
        assert_eq!(
            serde_json::json! {{"priority": "low", "unread": true}},
            policy.action
        );
        let mut manager = Manager::default();
        manager.add(policy);
        let report = manager
            .apply(
                &Anthropic::new(None).unwrap(),
                MessageCreateParams {
                    max_tokens: 2048,
                    ..Default::default()
                },
                r#"From: robert@example.org
To: jeff@example.org

This is an email about AI.
        "#,
                None,
            )
            .await
            .expect("manager should produce a JSON value");
        println!("{report}");
        assert_eq!(
            serde_json::json! {{"category": "other", "priority": "low", "unread": true}},
            report.value()
        );
    }
}
