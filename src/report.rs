use claudius::MessageParam;

use crate::{
    number_is_equal, number_less_than, BoolMask, Conflict, NumberMask, OnConflict, PolicyError,
    StringArrayMask, StringEnumMask, StringMask,
};

/// Contains the result of applying policies to unstructured data.
///
/// A Report tracks which rules matched, what values were extracted,
/// and any conflicts or errors that occurred during policy application.
#[derive(serde::Deserialize, serde::Serialize)]
pub struct Report {
    pub messages: Vec<MessageParam>,
    pub bool_masks: Vec<BoolMask>,
    pub number_masks: Vec<NumberMask>,
    pub string_masks: Vec<StringMask>,
    pub string_array_masks: Vec<StringArrayMask>,
    pub string_enum_masks: Vec<StringEnumMask>,
    pub masks_by_index: Vec<Vec<String>>,
    pub rules_matched: Vec<usize>,
    pub ir: Option<serde_json::Value>,

    default: Option<serde_json::Value>,
    value: Option<serde_json::Value>,
    errors: Vec<PolicyError>,
    conflicts: Vec<Conflict>,
}

impl Report {
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
        }
    }

    pub fn value(&self) -> serde_json::Value {
        let mut value = self.default.clone().unwrap_or(serde_json::json! {{}});
        if let Some(serde_json::Value::Object(obj)) = self.value.as_ref() {
            for (k, v) in obj.iter() {
                value[k.clone()] = v.clone();
            }
        }
        value
    }

    pub fn errors(&self) -> &[PolicyError] {
        &self.errors
    }

    pub fn conflicts(&self) -> &[Conflict] {
        &self.conflicts
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty() || !self.conflicts.is_empty()
    }

    pub fn report_bool_default(&mut self, field: &str, default: bool) {
        let build = self.default.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(existing) = build.get(field) {
            if *existing != serde_json::Value::Bool(default) {
                self.errors.push(PolicyError::DefaultConflict {
                    field: field.to_string(),
                    existing: existing.clone(),
                    new: serde_json::Value::Bool(default),
                    suggestion: "Ensure all policies use the same default value for this field"
                        .to_string(),
                });
            }
        } else {
            build[field.to_string()] = default.into();
        }
    }

    pub fn report_bool(
        &mut self,
        policy_index: usize,
        field: &str,
        value: bool,
        on_conflict: OnConflict,
    ) {
        self.report_policy_index(policy_index);
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

    pub fn report_number_default(&mut self, field: &str, default: impl Into<serde_json::Number>) {
        let default = default.into();
        let build = self.default.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(existing) = build.get(field) {
            if *existing != serde_json::Value::Number(default.clone()) {
                self.errors.push(PolicyError::DefaultConflict {
                    field: field.to_string(),
                    existing: existing.clone(),
                    new: serde_json::Value::Number(default),
                    suggestion: "Ensure all policies use the same default value for this field"
                        .to_string(),
                });
            }
        } else {
            build[field.to_string()] = default.into();
        }
    }

    pub fn report_number(
        &mut self,
        policy_index: usize,
        field: &str,
        value: impl Into<serde_json::Number>,
        on_conflict: OnConflict,
    ) {
        self.report_policy_index(policy_index);
        let value = value.into();

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
                                } else {
                                    conflict_to_report =
                                        Some((field.to_string(), existing.clone(), value.clone()));
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

    pub fn report_string_default(&mut self, field: &str, default: impl Into<String>) {
        let default = default.into();
        let build = self.default.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(existing) = build.get(field) {
            if *existing != serde_json::Value::String(default.clone()) {
                self.errors.push(PolicyError::DefaultConflict {
                    field: field.to_string(),
                    existing: existing.clone(),
                    new: serde_json::Value::String(default),
                    suggestion: "Ensure all policies use the same default value for this field"
                        .to_string(),
                });
            }
        } else {
            build[field.to_string()] = default.into();
        }
    }

    pub fn report_string(
        &mut self,
        policy_index: usize,
        field: &str,
        value: String,
        on_conflict: OnConflict,
    ) {
        self.report_policy_index(policy_index);

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

    pub fn report_string_enum(
        &mut self,
        policy_index: usize,
        field: &str,
        value: String,
        on_conflict: OnConflict,
    ) {
        self.report_policy_index(policy_index);
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
                                if value.len() > s.len() {
                                    *v = value.into();
                                } else {
                                    let s = s.clone();
                                    self.report_string_conflict(field, s, value);
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

    pub fn report_string_array(&mut self, policy_index: usize, field: &str, value: String) {
        self.report_policy_index(policy_index);
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

    fn report_policy_index(&mut self, policy_index: usize) {
        self.rules_matched.push(policy_index);
    }

    pub fn report_invariant_violation(&mut self, file: &str, line: u32, message: &str) {
        self.errors.push(PolicyError::InvariantViolation {
            file: file.to_string(),
            line,
            message: message.to_string(),
        });
    }

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

    fn report_number_conflict(
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

    fn report_string_conflict(&mut self, field: &str, val1: String, val2: String) {
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
