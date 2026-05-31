use std::collections::HashMap;

use claudius::MessageParam;

use crate::{
    number_is_equal, number_less_than, BoolMask, Conflict, NumberMask, OnConflict, PolicyError,
    StringArrayMask, StringEnumMask, StringMask,
};

fn enum_value_is_greater(existing: &str, new: &str, values: &[String]) -> bool {
    let existing_index = values.iter().position(|value| value == existing);
    let new_index = values.iter().position(|value| value == new);
    matches!((existing_index, new_index), (Some(existing), Some(new)) if new > existing)
}

/// Statistics for a single field in a report.
///
/// Tracks how many times a field was set and the distribution of values
/// that were reported for that field.
#[derive(Clone, Debug, Default, serde::Deserialize, serde::Serialize)]
pub struct FieldStats {
    /// The number of times this field was set.
    pub count: usize,
    /// Distribution of values as a histogram (value -> count).
    pub distribution: HashMap<String, usize>,
}

impl FieldStats {
    /// Record a value being set for this field.
    pub fn record(&mut self, value: &serde_json::Value) {
        self.count += 1;
        let key = match value {
            serde_json::Value::Null => "null".to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(arr) => {
                format!("[{}]", arr.len())
            }
            serde_json::Value::Object(obj) => {
                format!("{{{}}}", obj.len())
            }
        };
        *self.distribution.entry(key).or_insert(0) += 1;
    }

    /// Record an array element being added to a field.
    pub fn record_array_element(&mut self, value: &str) {
        self.count += 1;
        *self.distribution.entry(value.to_string()).or_insert(0) += 1;
    }
}

/// Contains the result of applying policies to unstructured data.
///
/// A Report tracks which rules matched, what values were extracted,
/// and any conflicts or errors that occurred during policy application.
#[derive(Clone, Default, serde::Deserialize, serde::Serialize)]
pub struct Report {
    /// Messages that were used in the LLM conversation
    pub messages: Vec<MessageParam>,
    /// Boolean field masks that were applied during processing
    pub bool_masks: Vec<BoolMask>,
    /// Numeric field masks that were applied during processing
    pub number_masks: Vec<NumberMask>,
    /// String field masks that were applied during processing
    pub string_masks: Vec<StringMask>,
    /// String array field masks that were applied during processing
    pub string_array_masks: Vec<StringArrayMask>,
    /// String enum field masks that were applied during processing
    pub string_enum_masks: Vec<StringEnumMask>,
    /// Mapping of policy indices to their associated field names
    pub masks_by_index: Vec<Vec<String>>,
    /// List of policy rule indices that were matched during processing
    pub rules_matched: Vec<usize>,
    /// The intermediate representation JSON received from the LLM
    pub ir: Option<serde_json::Value>,
    /// Default values for all fields in the report
    pub default: Option<serde_json::Value>,

    value: Option<serde_json::Value>,
    errors: Vec<PolicyError>,
    conflicts: Vec<Conflict>,
    field_stats: HashMap<String, FieldStats>,
}

impl Report {
    /// Create a new Report with the specified masks and configuration.
    ///
    /// # Arguments
    ///
    /// * `messages` - Messages to be included in the LLM conversation
    /// * `bool_masks` - Boolean field masks for policy application
    /// * `number_masks` - Numeric field masks for policy application
    /// * `string_masks` - String field masks for policy application
    /// * `string_array_masks` - String array field masks for policy application
    /// * `string_enum_masks` - String enum field masks for policy application
    /// * `masks_by_index` - Mapping of policy indices to their field names
    ///
    /// # Example
    ///
    /// ```
    /// use policyai::Report;
    /// use claudius::MessageParam;
    /// let report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// ```
    pub fn new(
        messages: Vec<MessageParam>,
        bool_masks: Vec<BoolMask>,
        number_masks: Vec<NumberMask>,
        string_masks: Vec<StringMask>,
        string_array_masks: Vec<StringArrayMask>,
        string_enum_masks: Vec<StringEnumMask>,
        masks_by_index: Vec<Vec<String>>,
    ) -> Self {
        Self {
            messages,
            bool_masks,
            number_masks,
            string_masks,
            string_array_masks,
            string_enum_masks,
            masks_by_index,
            rules_matched: vec![],
            ir: None,
            default: None,
            value: None,
            errors: vec![],
            conflicts: vec![],
            field_stats: HashMap::new(),
        }
    }

    /// Get the final structured output value combining defaults and extracted values.
    ///
    /// Returns a JSON object that merges the default values with any values
    /// that were successfully extracted and reported during policy application.
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::Report;
    /// # use claudius::MessageParam;
    /// let report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// let output = report.value();
    /// assert!(output.is_object());
    /// ```
    pub fn value(&self) -> serde_json::Value {
        let mut value = self.default.clone().unwrap_or(serde_json::json! {{}});
        if let Some(serde_json::Value::Object(obj)) = self.value.as_ref() {
            for (k, v) in obj.iter() {
                value[k.clone()] = v.clone();
            }
        }
        value
    }

    /// Get all policy errors that occurred during processing.
    ///
    /// Returns a slice of PolicyError instances representing issues such as
    /// invariant violations, type check failures, and default conflicts.
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::Report;
    /// # use claudius::MessageParam;
    /// let report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// let errors = report.errors();
    /// assert!(errors.is_empty());
    /// ```
    pub fn errors(&self) -> &[PolicyError] {
        &self.errors
    }

    /// Get all conflicts that occurred during policy value resolution.
    ///
    /// Returns a slice of Conflict instances representing situations where
    /// multiple policies specified different values for the same field.
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::Report;
    /// # use claudius::MessageParam;
    /// let report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// let conflicts = report.conflicts();
    /// assert!(conflicts.is_empty());
    /// ```
    pub fn conflicts(&self) -> &[Conflict] {
        &self.conflicts
    }

    /// Get the statistics for a specific field.
    ///
    /// Returns the FieldStats for the given field name, or None if no values
    /// were reported for that field.
    ///
    /// # Arguments
    ///
    /// * `field` - The name of the field to get statistics for
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::{Report, OnConflict};
    /// # use claudius::MessageParam;
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// report.report_bool(1, "active", true, OnConflict::Default);
    /// let stats = report.field_stats("active").unwrap();
    /// assert_eq!(stats.count, 1);
    /// ```
    pub fn field_stats(&self, field: &str) -> Option<&FieldStats> {
        self.field_stats.get(field)
    }

    /// Get statistics for all fields.
    ///
    /// Returns a reference to the HashMap containing statistics for all fields
    /// that had values reported during policy application.
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::{Report, OnConflict};
    /// # use claudius::MessageParam;
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// report.report_bool(1, "active", true, OnConflict::Default);
    /// let all_stats = report.all_field_stats();
    /// assert!(all_stats.contains_key("active"));
    /// ```
    pub fn all_field_stats(&self) -> &HashMap<String, FieldStats> {
        &self.field_stats
    }

    /// Check if the report contains any errors or conflicts.
    ///
    /// Returns true if there are any policy errors or conflicts that occurred
    /// during processing, indicating potential issues with the policy application.
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::Report;
    /// # use claudius::MessageParam;
    /// let report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// assert!(!report.has_errors());
    /// ```
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty() || !self.conflicts.is_empty()
    }

    /// Report a default boolean value for a field.
    ///
    /// Sets or validates the default value for a boolean field. If a default
    /// already exists and differs, reports a default conflict error.
    ///
    /// # Arguments
    ///
    /// * `field` - The name of the field to set the default for
    /// * `default` - The default boolean value
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::Report;
    /// # use claudius::MessageParam;
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// report.report_bool_default("active", true);
    /// ```
    pub fn report_bool_default(&mut self, field: &str, default: bool) {
        let build = self.default.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(existing) = build.get(field) {
            if *existing != serde_json::Value::Bool(default) {
                self.errors.push(PolicyError::DefaultConflict {
                    field: field.to_string(),
                    existing: Box::new(existing.clone()),
                    new: Box::new(serde_json::Value::Bool(default)),
                    suggestion: "Ensure all policies use the same default value for this field"
                        .to_string(),
                });
            }
        } else {
            build[field.to_string()] = default.into();
        }
    }

    /// Report a boolean value from a policy application.
    ///
    /// Records a boolean value extracted by a policy and handles conflicts
    /// according to the specified conflict resolution strategy.
    ///
    /// # Arguments
    ///
    /// * `policy_index` - The index of the policy reporting this value
    /// * `field` - The name of the field being reported
    /// * `value` - The boolean value to report
    /// * `on_conflict` - Strategy for handling conflicts with existing values
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::{Report, OnConflict};
    /// # use claudius::MessageParam;
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// report.report_bool(1, "urgent", true, OnConflict::Agreement);
    /// ```
    pub fn report_bool(
        &mut self,
        policy_index: usize,
        field: &str,
        value: bool,
        on_conflict: OnConflict,
    ) {
        self.report_policy_index(policy_index);
        self.field_stats
            .entry(field.to_string())
            .or_default()
            .record(&serde_json::Value::Bool(value));
        let build = self.value.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(v) = build.get_mut(field) {
            match v {
                serde_json::Value::Null => *v = value.into(),
                serde_json::Value::Bool(b) => {
                    if *b != value {
                        match on_conflict {
                            OnConflict::Default => {}
                            OnConflict::Agreement => {
                                let b = *b;
                                self.report_bool_conflict(field, b, value);
                            }
                            OnConflict::LargestValue => {
                                if value {
                                    *b = value;
                                }
                            }
                        }
                    }
                }
                serde_json::Value::Number(_) => {
                    self.report_invariant_violation(
                        file!(),
                        line!(),
                        "number found in place of bool",
                    );
                }
                serde_json::Value::String(_) => {
                    self.report_invariant_violation(
                        file!(),
                        line!(),
                        "string found in place of bool",
                    );
                }
                serde_json::Value::Array(_) => {
                    self.report_invariant_violation(
                        file!(),
                        line!(),
                        "array found in place of bool",
                    );
                }
                serde_json::Value::Object(_) => {
                    self.report_invariant_violation(file!(), line!(), "found an object");
                }
            }
        } else {
            build[field] = value.into();
        }
    }

    /// Report a default numeric value for a field.
    ///
    /// Sets or validates the default value for a numeric field. If a default
    /// already exists and differs, reports a default conflict error.
    ///
    /// # Arguments
    ///
    /// * `field` - The name of the field to set the default for
    /// * `default` - The default numeric value
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::Report;
    /// # use claudius::MessageParam;
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// report.report_number_default("score", 0);
    /// ```
    pub fn report_number_default(&mut self, field: &str, default: impl Into<serde_json::Number>) {
        let default = default.into();
        let build = self.default.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(existing) = build.get(field) {
            if *existing != serde_json::Value::Number(default.clone()) {
                self.errors.push(PolicyError::DefaultConflict {
                    field: field.to_string(),
                    existing: Box::new(existing.clone()),
                    new: Box::new(serde_json::Value::Number(default)),
                    suggestion: "Ensure all policies use the same default value for this field"
                        .to_string(),
                });
            }
        } else {
            build[field.to_string()] = default.into();
        }
    }

    /// Report a numeric value from a policy application.
    ///
    /// Records a numeric value extracted by a policy and handles conflicts
    /// according to the specified conflict resolution strategy.
    ///
    /// # Arguments
    ///
    /// * `policy_index` - The index of the policy reporting this value
    /// * `field` - The name of the field being reported
    /// * `value` - The numeric value to report
    /// * `on_conflict` - Strategy for handling conflicts with existing values
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::{Report, OnConflict};
    /// # use claudius::MessageParam;
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// report.report_number(1, "priority", 10, OnConflict::LargestValue);
    /// ```
    pub fn report_number(
        &mut self,
        policy_index: usize,
        field: &str,
        value: impl Into<serde_json::Number>,
        on_conflict: OnConflict,
    ) {
        self.report_policy_index(policy_index);
        let value = value.into();
        self.field_stats
            .entry(field.to_string())
            .or_default()
            .record(&serde_json::Value::Number(value.clone()));

        let mut conflict_to_report = None;
        let mut error_to_report = None;

        let build = self.value.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(v) = build.get_mut(field) {
            match v {
                serde_json::Value::Null => *v = value.into(),
                serde_json::Value::Number(existing) => {
                    if !number_is_equal(existing, &value) {
                        match on_conflict {
                            OnConflict::Default => {}
                            OnConflict::Agreement => {
                                conflict_to_report =
                                    Some((field.to_string(), existing.clone(), value.clone()));
                            }
                            OnConflict::LargestValue => {
                                if number_less_than(existing, &value) {
                                    *existing = value;
                                }
                            }
                        }
                    }
                }
                serde_json::Value::Bool(_) => {
                    error_to_report = Some("bool found in place of number".to_string());
                }
                serde_json::Value::String(_) => {
                    error_to_report = Some("string found in place of number".to_string());
                }
                serde_json::Value::Array(_) => {
                    error_to_report = Some("array found in place of number".to_string());
                }
                serde_json::Value::Object(_) => {
                    error_to_report = Some("found an object".to_string());
                }
            }
        } else {
            build[field] = value.into();
        }

        if let Some((field_name, old_val, new_val)) = conflict_to_report {
            self.report_number_conflict(&field_name, old_val, new_val);
        }
        if let Some(error_msg) = error_to_report {
            self.report_invariant_violation(file!(), line!(), &error_msg);
        }
    }

    /// Report a default string value for a field.
    ///
    /// Sets or validates the default value for a string field. If a default
    /// already exists and differs, reports a default conflict error.
    ///
    /// # Arguments
    ///
    /// * `field` - The name of the field to set the default for
    /// * `default` - The default string value
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::Report;
    /// # use claudius::MessageParam;
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// report.report_string_default("category", "unknown");
    /// ```
    pub fn report_string_default(&mut self, field: &str, default: impl Into<String>) {
        let default = default.into();
        let build = self.default.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(existing) = build.get(field) {
            if *existing != serde_json::Value::String(default.clone()) {
                self.errors.push(PolicyError::DefaultConflict {
                    field: field.to_string(),
                    existing: Box::new(existing.clone()),
                    new: Box::new(serde_json::Value::String(default)),
                    suggestion: "Ensure all policies use the same default value for this field"
                        .to_string(),
                });
            }
        } else {
            build[field.to_string()] = default.into();
        }
    }

    /// Report a string value from a policy application.
    ///
    /// Records a string value extracted by a policy and handles conflicts
    /// according to the specified conflict resolution strategy.
    ///
    /// # Arguments
    ///
    /// * `policy_index` - The index of the policy reporting this value
    /// * `field` - The name of the field being reported
    /// * `value` - The string value to report
    /// * `on_conflict` - Strategy for handling conflicts with existing values
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::{Report, OnConflict};
    /// # use claudius::MessageParam;
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// report.report_string(1, "title", "Important Message".to_string(), OnConflict::Agreement);
    /// ```
    pub fn report_string(
        &mut self,
        policy_index: usize,
        field: &str,
        value: String,
        on_conflict: OnConflict,
    ) {
        self.report_policy_index(policy_index);
        self.field_stats
            .entry(field.to_string())
            .or_default()
            .record(&serde_json::Value::String(value.clone()));

        let mut conflict_to_report = None;
        let mut error_to_report = None;

        let build = self.value.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(v) = build.get_mut(field) {
            match v {
                serde_json::Value::Null => *v = value.into(),
                serde_json::Value::String(existing) => {
                    if *existing != value {
                        match on_conflict {
                            OnConflict::Default => {}
                            OnConflict::Agreement => {
                                conflict_to_report =
                                    Some((field.to_string(), existing.clone(), value.clone()));
                            }
                            OnConflict::LargestValue => {
                                if value.len() > existing.len() {
                                    *v = value.into();
                                }
                            }
                        }
                    }
                }
                serde_json::Value::Bool(_) => {
                    error_to_report = Some("bool found in place of string".to_string());
                }
                serde_json::Value::Number(_) => {
                    error_to_report = Some("number found in place of string".to_string());
                }
                serde_json::Value::Array(_) => {
                    error_to_report = Some("array found in place of string".to_string());
                }
                serde_json::Value::Object(_) => {
                    error_to_report = Some("found an object".to_string());
                }
            }
        } else {
            build[field] = value.into();
        }

        if let Some((field_name, old_val, new_val)) = conflict_to_report {
            self.report_string_conflict(&field_name, old_val, new_val);
        }
        if let Some(error_msg) = error_to_report {
            self.report_invariant_violation(file!(), line!(), &error_msg);
        }
    }

    /// Report a string enum value from a policy application.
    ///
    /// Records a string enum value extracted by a policy and handles conflicts
    /// according to the specified conflict resolution strategy.
    ///
    /// # Arguments
    ///
    /// * `policy_index` - The index of the policy reporting this value
    /// * `field` - The name of the field being reported
    /// * `value` - The enum value to report
    /// * `on_conflict` - Strategy for handling conflicts with existing values
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::{Report, OnConflict};
    /// # use claudius::MessageParam;
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// let values = vec!["inactive".to_string(), "active".to_string()];
    /// report.report_string_enum(1, "status", "active".to_string(), &values, OnConflict::LargestValue);
    /// ```
    pub fn report_string_enum(
        &mut self,
        policy_index: usize,
        field: &str,
        value: String,
        values: &[String],
        on_conflict: OnConflict,
    ) {
        self.report_policy_index(policy_index);
        self.field_stats
            .entry(field.to_string())
            .or_default()
            .record(&serde_json::Value::String(value.clone()));
        let build = self.value.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(v) = build.get_mut(field) {
            match v {
                serde_json::Value::Null => *v = value.into(),
                serde_json::Value::String(s) => {
                    if *s != value {
                        match on_conflict {
                            OnConflict::Default => {}
                            OnConflict::Agreement => {
                                let s = s.clone();
                                self.report_string_conflict(field, s, value);
                            }
                            OnConflict::LargestValue => {
                                if enum_value_is_greater(s, &value, values) {
                                    *v = value.into();
                                }
                            }
                        }
                    }
                }
                serde_json::Value::Bool(_) => {
                    self.report_invariant_violation(
                        file!(),
                        line!(),
                        "bool found in place of string enum",
                    );
                }
                serde_json::Value::Number(_) => {
                    self.report_invariant_violation(
                        file!(),
                        line!(),
                        "number found in place of string enum",
                    );
                }
                serde_json::Value::Array(_) => {
                    self.report_invariant_violation(
                        file!(),
                        line!(),
                        "array found in place of string enum",
                    );
                }
                serde_json::Value::Object(_) => {
                    self.report_invariant_violation(file!(), line!(), "found an object");
                }
            }
        } else {
            build[field] = value.into();
        }
    }

    /// Report a string array element from a policy application.
    ///
    /// Adds a string value to an array field. If the field doesn't exist,
    /// creates a new array. Duplicates are automatically filtered out.
    ///
    /// # Arguments
    ///
    /// * `policy_index` - The index of the policy reporting this value
    /// * `field` - The name of the array field being reported to
    /// * `value` - The string value to add to the array
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::Report;
    /// # use claudius::MessageParam;
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// report.report_string_array(1, "tags", "urgent".to_string());
    /// report.report_string_array(1, "tags", "important".to_string());
    /// ```
    pub fn report_string_array(&mut self, policy_index: usize, field: &str, value: String) {
        self.report_policy_index(policy_index);
        self.field_stats
            .entry(field.to_string())
            .or_default()
            .record_array_element(&value);
        let build = self.value.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(serde_json::Value::Array(arr)) = build.get_mut(field) {
            let value: serde_json::Value = value.into();
            if !arr.contains(&value) {
                arr.push(value);
            }
        } else {
            build[field.to_string()] = vec![value].into();
        }
    }

    /// Initialize an empty string array for a field in the report.
    pub fn init_empty_string_array(&mut self, policy_index: usize, field: &str) {
        self.report_policy_index(policy_index);
        let build = self.value.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        build
            .as_object_mut()
            .unwrap()
            .entry(field)
            .or_insert_with(|| serde_json::Value::Array(vec![]));
    }

    /// Record that a policy was matched.
    ///
    /// This is called internally when a mask is applied and matches the input data,
    /// tracking which policies contributed to the report.
    ///
    /// # Arguments
    ///
    /// * `policy_index` - The index of the policy that was matched
    pub fn report_policy_index(&mut self, policy_index: usize) {
        self.rules_matched.push(policy_index);
    }

    /// Report an invariant violation error.
    ///
    /// Records a programming error where an internal assumption was violated,
    /// typically indicating a bug in the policy application logic.
    ///
    /// # Arguments
    ///
    /// * `file` - The source file where the violation occurred
    /// * `line` - The line number where the violation occurred
    /// * `message` - A description of the invariant that was violated
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::Report;
    /// # use claudius::MessageParam;
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// report.report_invariant_violation(file!(), line!(), "unexpected null value");
    /// ```
    pub fn report_invariant_violation(&mut self, file: &str, line: u32, message: &str) {
        self.errors.push(PolicyError::InvariantViolation {
            file: file.to_string(),
            line,
            message: message.to_string(),
        });
    }

    /// Report a type check failure error.
    ///
    /// Records an error where the LLM provided data that doesn't match the
    /// expected type for a field, such as a string where a number was expected.
    ///
    /// # Arguments
    ///
    /// * `file` - The source file where the failure occurred
    /// * `line` - The line number where the failure occurred
    /// * `message` - A description of the type check failure
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::Report;
    /// # use claudius::MessageParam;
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// report.report_type_check_failure(file!(), line!(), "expected boolean, got string");
    /// ```
    pub fn report_type_check_failure(&mut self, file: &str, line: u32, message: &str) {
        self.errors.push(PolicyError::TypeCheckFailure {
            file: file.to_string(),
            line,
            message: message.to_string(),
            expected: "valid JSON matching policy type".to_string(),
            actual: "invalid or mismatched JSON".to_string(),
        });
    }

    fn report_bool_conflict(&mut self, field: &str, val1: bool, val2: bool) {
        self.conflicts.push(Conflict::BoolConflict {
            field: field.to_string(),
            val1,
            val2,
        });
    }

    /// Report a conflict between two numeric values for the same field.
    ///
    /// Records a conflict when multiple policies specify different numeric values
    /// for the same field, enabling conflict detection and resolution reporting.
    ///
    /// # Arguments
    ///
    /// * `field` - The name of the field experiencing the conflict
    /// * `val1` - The first conflicting numeric value
    /// * `val2` - The second conflicting numeric value
    pub fn report_number_conflict(
        &mut self,
        field: &str,
        val1: serde_json::Number,
        val2: serde_json::Number,
    ) {
        self.conflicts.push(Conflict::NumberConflict {
            field: field.to_string(),
            val1,
            val2,
        });
    }

    /// Report a conflict between two string values for the same field.
    ///
    /// Records a conflict when multiple policies specify different string values
    /// for the same field, enabling conflict detection and resolution reporting.
    ///
    /// # Arguments
    ///
    /// * `field` - The name of the field experiencing the conflict
    /// * `val1` - The first conflicting string value
    /// * `val2` - The second conflicting string value
    pub fn report_string_conflict(&mut self, field: &str, val1: String, val2: String) {
        self.conflicts.push(Conflict::StringConflict {
            field: field.to_string(),
            val1,
            val2,
        });
    }

    /// Report a conflict between a boolean flag and expected enum value.
    ///
    /// Records a conflict when a string enum field receives a boolean value
    /// that doesn't match the expected enum value, enabling conflict detection
    /// and resolution reporting.
    ///
    /// # Arguments
    ///
    /// * `field` - The name of the field experiencing the conflict
    /// * `val1` - The existing string value from the report
    /// * `val2` - The expected enum string value
    pub fn report_string_enum_conflict(&mut self, field: &str, val1: String, val2: String) {
        self.conflicts.push(Conflict::StringConflict {
            field: field.to_string(),
            val1,
            val2,
        });
    }
}

impl std::fmt::Display for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(
            f,
            "ir: {}",
            serde_json::to_string_pretty(self.ir.as_ref().unwrap_or(&serde_json::json! {{}}))
                .unwrap()
        )?;
        writeln!(
            f,
            "so: {}",
            serde_json::to_string_pretty(self.value.as_ref().unwrap_or(&serde_json::json! {{}}))
                .unwrap()
        )
    }
}

impl std::fmt::Debug for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_struct("Report").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_stats_record_bool() {
        let mut stats = FieldStats::default();
        stats.record(&serde_json::Value::Bool(true));
        stats.record(&serde_json::Value::Bool(true));
        stats.record(&serde_json::Value::Bool(false));

        assert_eq!(stats.count, 3);
        assert_eq!(stats.distribution.get("true"), Some(&2));
        assert_eq!(stats.distribution.get("false"), Some(&1));
        // Debug output for future inspection
        println!(
            "field_stats_record_bool: count={}, distribution={:?}",
            stats.count, stats.distribution
        );
    }

    #[test]
    fn field_stats_record_number() {
        let mut stats = FieldStats::default();
        stats.record(&serde_json::json!(42));
        stats.record(&serde_json::json!(42));
        stats.record(&serde_json::json!(100));

        assert_eq!(stats.count, 3);
        assert_eq!(stats.distribution.get("42"), Some(&2));
        assert_eq!(stats.distribution.get("100"), Some(&1));
        println!(
            "field_stats_record_number: count={}, distribution={:?}",
            stats.count, stats.distribution
        );
    }

    #[test]
    fn field_stats_record_string() {
        let mut stats = FieldStats::default();
        stats.record(&serde_json::json!("high"));
        stats.record(&serde_json::json!("low"));
        stats.record(&serde_json::json!("high"));

        assert_eq!(stats.count, 3);
        assert_eq!(stats.distribution.get("high"), Some(&2));
        assert_eq!(stats.distribution.get("low"), Some(&1));
        println!(
            "field_stats_record_string: count={}, distribution={:?}",
            stats.count, stats.distribution
        );
    }

    #[test]
    fn field_stats_record_array_element() {
        let mut stats = FieldStats::default();
        stats.record_array_element("tag1");
        stats.record_array_element("tag2");
        stats.record_array_element("tag1");

        assert_eq!(stats.count, 3);
        assert_eq!(stats.distribution.get("tag1"), Some(&2));
        assert_eq!(stats.distribution.get("tag2"), Some(&1));
        println!(
            "field_stats_record_array_element: count={}, distribution={:?}",
            stats.count, stats.distribution
        );
    }

    #[test]
    fn report_tracks_bool_stats() {
        let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        report.report_bool(1, "active", true, OnConflict::Default);
        report.report_bool(2, "active", true, OnConflict::Default);
        report.report_bool(3, "active", false, OnConflict::Default);

        let stats = report
            .field_stats("active")
            .expect("active field should have stats");
        assert_eq!(stats.count, 3);
        assert_eq!(stats.distribution.get("true"), Some(&2));
        assert_eq!(stats.distribution.get("false"), Some(&1));
        println!(
            "report_tracks_bool_stats: count={}, distribution={:?}",
            stats.count, stats.distribution
        );
    }

    #[test]
    fn report_tracks_number_stats() {
        let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        report.report_number(1, "score", 10, OnConflict::Default);
        report.report_number(2, "score", 20, OnConflict::Default);
        report.report_number(3, "score", 10, OnConflict::Default);

        let stats = report
            .field_stats("score")
            .expect("score field should have stats");
        assert_eq!(stats.count, 3);
        assert_eq!(stats.distribution.get("10"), Some(&2));
        assert_eq!(stats.distribution.get("20"), Some(&1));
        println!(
            "report_tracks_number_stats: count={}, distribution={:?}",
            stats.count, stats.distribution
        );
    }

    #[test]
    fn report_tracks_string_stats() {
        let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        report.report_string(1, "title", "Hello".to_string(), OnConflict::Default);
        report.report_string(2, "title", "World".to_string(), OnConflict::Default);

        let stats = report
            .field_stats("title")
            .expect("title field should have stats");
        assert_eq!(stats.count, 2);
        assert_eq!(stats.distribution.get("Hello"), Some(&1));
        assert_eq!(stats.distribution.get("World"), Some(&1));
        println!(
            "report_tracks_string_stats: count={}, distribution={:?}",
            stats.count, stats.distribution
        );
    }

    #[test]
    fn report_tracks_string_enum_stats() {
        let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        let values = vec!["low".to_string(), "medium".to_string(), "high".to_string()];
        report.report_string_enum(
            1,
            "priority",
            "high".to_string(),
            &values,
            OnConflict::Default,
        );
        report.report_string_enum(
            2,
            "priority",
            "low".to_string(),
            &values,
            OnConflict::Default,
        );
        report.report_string_enum(
            3,
            "priority",
            "high".to_string(),
            &values,
            OnConflict::Default,
        );

        let stats = report
            .field_stats("priority")
            .expect("priority field should have stats");
        assert_eq!(stats.count, 3);
        assert_eq!(stats.distribution.get("high"), Some(&2));
        assert_eq!(stats.distribution.get("low"), Some(&1));
        println!(
            "report_tracks_string_enum_stats: count={}, distribution={:?}",
            stats.count, stats.distribution
        );
    }

    #[test]
    fn report_tracks_string_array_stats() {
        let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        report.report_string_array(1, "tags", "urgent".to_string());
        report.report_string_array(2, "tags", "important".to_string());
        report.report_string_array(3, "tags", "urgent".to_string());

        let stats = report
            .field_stats("tags")
            .expect("tags field should have stats");
        assert_eq!(stats.count, 3);
        assert_eq!(stats.distribution.get("urgent"), Some(&2));
        assert_eq!(stats.distribution.get("important"), Some(&1));
        println!(
            "report_tracks_string_array_stats: count={}, distribution={:?}",
            stats.count, stats.distribution
        );
    }

    #[test]
    fn report_number_largest_value_keeps_largest_without_conflict() {
        let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        report.report_number(1, "score", 10, OnConflict::LargestValue);
        report.report_number(2, "score", 20, OnConflict::LargestValue);
        report.report_number(3, "score", 5, OnConflict::LargestValue);

        assert_eq!(serde_json::json!({"score": 20}), report.value());
        assert!(report.conflicts().is_empty());
    }

    #[test]
    fn report_string_enum_largest_value_uses_declared_order() {
        let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        let values = vec!["low".to_string(), "medium".to_string(), "high".to_string()];
        report.report_string_enum(
            1,
            "priority",
            "medium".to_string(),
            &values,
            OnConflict::LargestValue,
        );
        report.report_string_enum(
            2,
            "priority",
            "high".to_string(),
            &values,
            OnConflict::LargestValue,
        );
        report.report_string_enum(
            3,
            "priority",
            "medium".to_string(),
            &values,
            OnConflict::LargestValue,
        );

        assert_eq!(serde_json::json!({"priority": "high"}), report.value());
        assert!(report.conflicts().is_empty());
    }

    #[test]
    fn report_all_field_stats() {
        let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        report.report_bool(1, "active", true, OnConflict::Default);
        report.report_number(2, "score", 42, OnConflict::Default);

        let all_stats = report.all_field_stats();
        assert!(all_stats.contains_key("active"));
        assert!(all_stats.contains_key("score"));
        assert_eq!(all_stats.len(), 2);
        println!("report_all_field_stats: {:?}", all_stats);
    }

    #[test]
    fn report_field_stats_none_for_unreported() {
        let report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
        assert!(report.field_stats("nonexistent").is_none());
        println!("report_field_stats_none_for_unreported: field_stats returned None as expected");
    }
}
