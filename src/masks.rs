use crate::{number_is_equal, t64, OnConflict, Report};

///////////////////////////////////////////// BoolMask /////////////////////////////////////////////

/// Represents a boolean field mask for policy application.
///
/// A BoolMask handles the extraction and conflict resolution of boolean values
/// from unstructured data based on policy rules.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct BoolMask {
    /// Index of the policy this mask belongs to
    pub policy_index: usize,
    /// Original field name from the policy definition
    pub name: String,
    /// Masked field name unlikely to be in LLM training data
    pub mask: String,
    /// Default value when the field is not present
    pub default: Option<bool>,
    /// Strategy for resolving conflicts when multiple policies set different values
    pub on_conflict: OnConflict,
}

impl BoolMask {
    /// Create a new BoolMask with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `policy_index` - The index of the policy this mask belongs to
    /// * `name` - The original field name from the policy definition
    /// * `mask` - The masked field name unlikely to be in LLM training data
    /// * `default` - The default boolean value when field is absent
    /// * `on_conflict` - Strategy for resolving conflicts between policies
    ///
    /// # Example
    ///
    /// ```
    /// use policyai::{BoolMask, OnConflict};
    /// let mask = BoolMask::new(
    ///     1,
    ///     "urgent".to_string(),
    ///     "field_abc123".to_string(),
    ///     None,
    ///     true,
    ///     OnConflict::Agreement
    /// );
    /// ```
    pub fn new(
        policy_index: usize,
        name: String,
        mask: String,
        default: Option<bool>,
        on_conflict: OnConflict,
    ) -> Self {
        Self {
            policy_index,
            name,
            mask,
            default,
            on_conflict,
        }
    }

    /// Apply this boolean mask to intermediate representation data.
    ///
    /// Extracts the boolean value from the IR and reports it to the given Report
    /// if it matches the expected value, otherwise reports the default.
    ///
    /// # Arguments
    ///
    /// * `ir` - The intermediate representation JSON from the LLM
    /// * `report` - The report to write results and errors to
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::{BoolMask, OnConflict, Report};
    /// let mask = BoolMask::new(1, "urgent".to_string(), "field_abc".to_string(), None, true, OnConflict::Default);
    /// let ir = serde_json::json!({"field_abc": true});
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// mask.apply_to(&ir, &mut report);
    /// ```
    pub fn apply_to(&self, ir: &serde_json::Value, report: &mut Report) {
        match ir.get(&self.mask) {
            Some(serde_json::Value::Bool(ret)) => {
                report.report_bool(self.policy_index, &self.name, *ret, self.on_conflict);
            }
            Some(_) => {
                report.report_type_check_failure(
                    file!(),
                    line!(),
                    &format!("expected boolean for {}", self.name),
                );
            }
            None => {
                if let Some(v) = self.default {
                    report.report_bool_default(&self.name, v);
                }
            }
        }
    }
}

//////////////////////////////////////////// NumberMask ////////////////////////////////////////////

/// Represents a numeric field mask for policy application.
///
/// A NumberMask handles the extraction and conflict resolution of numeric values
/// from unstructured data based on policy rules.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct NumberMask {
    /// Index of the policy this mask belongs to
    pub policy_index: usize,
    /// Original field name from the policy definition
    pub name: String,
    /// Masked field name unlikely to be in LLM training data
    pub mask: String,
    /// Default value when the field is not present
    pub default: Option<t64>,
    /// Expected numeric value for this policy rule
    pub value: Option<serde_json::Number>,
    /// Strategy for resolving conflicts when multiple policies set different values
    pub on_conflict: OnConflict,
}

impl NumberMask {
    /// Create a new NumberMask with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `policy_index` - The index of the policy this mask belongs to
    /// * `name` - The original field name from the policy definition
    /// * `mask` - The masked field name unlikely to be in LLM training data
    /// * `default` - The default numeric value when field is absent
    /// * `value` - The expected numeric value for this mask
    /// * `on_conflict` - Strategy for resolving conflicts between policies
    ///
    /// # Example
    ///
    /// ```
    /// use policyai::{NumberMask, OnConflict, t64};
    /// let mask = NumberMask::new(
    ///     1,
    ///     "priority".to_string(),
    ///     "field_xyz789".to_string(),
    ///     Some(t64(0.0)),
    ///     Some(serde_json::Number::from(42)),
    ///     OnConflict::LargestValue
    /// );
    /// ```
    pub fn new(
        policy_index: usize,
        name: String,
        mask: String,
        default: Option<t64>,
        value: Option<serde_json::Number>,
        on_conflict: OnConflict,
    ) -> Self {
        Self {
            policy_index,
            name,
            mask,
            default,
            value,
            on_conflict,
        }
    }

    /// Apply this numeric mask to intermediate representation data.
    ///
    /// Extracts the numeric value from the IR and reports it to the given Report,
    /// applying conflict resolution strategies as needed.
    ///
    /// # Arguments
    ///
    /// * `ir` - The intermediate representation JSON from the LLM
    /// * `report` - The report to write results and errors to
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::{NumberMask, OnConflict, Report, t64};
    /// # use claudius::MessageParam;
    /// let mask = NumberMask::new(1, "score".to_string(), "field_num".to_string(), Some(t64(0.0)), Some(serde_json::Number::from(42)), OnConflict::Default);
    /// let ir = serde_json::json!({"field_num": 42});
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// mask.apply_to(&ir, &mut report);
    /// ```
    pub fn apply_to(&self, ir: &serde_json::Value, report: &mut Report) {
        match ir.get(&self.mask) {
            Some(serde_json::Value::Number(value)) => {
                if let Some(expected_value) = &self.value {
                    if number_is_equal(value, expected_value) {
                        report.report_number(
                            self.policy_index,
                            &self.name,
                            value.clone(),
                            self.on_conflict,
                        );
                    } else {
                        report.report_policy_index(self.policy_index);
                        report.report_number_conflict(
                            &self.name,
                            value.clone(),
                            expected_value.clone(),
                        );
                    }
                } else {
                    report.report_number(
                        self.policy_index,
                        &self.name,
                        value.clone(),
                        self.on_conflict,
                    );
                }
            }
            Some(_) => {
                report.report_type_check_failure(
                    file!(),
                    line!(),
                    &format!("expected number for {}", self.name),
                );
            }
            None => {
                if let Some(default) = self.default.as_ref() {
                    if let Some(default) = serde_json::Number::from_f64(default.0) {
                        report.report_number_default(&self.name, default);
                    } else {
                        report.report_invariant_violation(
                            file!(),
                            line!(),
                            "cannot cast to number",
                        );
                    }
                }
            }
        }
    }
}

//////////////////////////////////////////// StringMask ////////////////////////////////////////////

/// Represents a string field mask for policy application.
///
/// A StringMask handles the extraction and conflict resolution of string values
/// from unstructured data based on policy rules.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct StringMask {
    /// Index of the policy this mask belongs to
    pub policy_index: usize,
    /// Original field name from the policy definition
    pub name: String,
    /// Masked field name unlikely to be in LLM training data
    pub mask: String,
    /// Default value when the field is not present
    pub default: Option<String>,
    /// Expected string value for this policy rule
    pub value: Option<String>,
    /// Strategy for resolving conflicts when multiple policies set different values
    pub on_conflict: OnConflict,
}

impl StringMask {
    /// Create a new StringMask with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `policy_index` - The index of the policy this mask belongs to
    /// * `name` - The original field name from the policy definition
    /// * `mask` - The masked field name unlikely to be in LLM training data
    /// * `default` - The default string value when field is absent
    /// * `value` - The expected string value for this mask
    /// * `on_conflict` - Strategy for resolving conflicts between policies
    ///
    /// # Example
    ///
    /// ```
    /// use policyai::{StringMask, OnConflict};
    /// let mask = StringMask::new(
    ///     1,
    ///     "category".to_string(),
    ///     "field_str456".to_string(),
    ///     Some("default".to_string()),
    ///     Some("urgent".to_string()),
    ///     OnConflict::Agreement
    /// );
    /// ```
    pub fn new(
        policy_index: usize,
        name: String,
        mask: String,
        default: Option<String>,
        value: Option<String>,
        on_conflict: OnConflict,
    ) -> Self {
        Self {
            policy_index,
            name,
            mask,
            default,
            value,
            on_conflict,
        }
    }

    /// Apply this string mask to intermediate representation data.
    ///
    /// Extracts the string value from the IR and reports it to the given Report,
    /// applying conflict resolution strategies as needed.
    ///
    /// # Arguments
    ///
    /// * `ir` - The intermediate representation JSON from the LLM
    /// * `report` - The report to write results and errors to
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::{StringMask, OnConflict, Report};
    /// # use claudius::MessageParam;
    /// let mask = StringMask::new(1, "title".to_string(), "field_str".to_string(), None, Some("important".to_string()), OnConflict::Default);
    /// let ir = serde_json::json!({"field_str": "important"});
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// mask.apply_to(&ir, &mut report);
    /// ```
    pub fn apply_to(&self, ir: &serde_json::Value, report: &mut Report) {
        match ir.get(&self.mask) {
            Some(serde_json::Value::String(value)) => {
                if let Some(expected_value) = &self.value {
                    if value == expected_value {
                        report.report_string(
                            self.policy_index,
                            &self.name,
                            value.clone(),
                            self.on_conflict,
                        );
                    } else {
                        report.report_policy_index(self.policy_index);
                        report.report_string_conflict(
                            &self.name,
                            value.clone(),
                            expected_value.clone(),
                        );
                    }
                } else {
                    report.report_string(
                        self.policy_index,
                        &self.name,
                        value.clone(),
                        self.on_conflict,
                    );
                }
            }
            Some(_) => {
                report.report_type_check_failure(
                    file!(),
                    line!(),
                    &format!("expected string for {}", self.name),
                );
            }
            _ => {
                if let Some(default) = self.default.as_ref() {
                    report.report_string_default(&self.name, default);
                }
            }
        }
    }
}

////////////////////////////////////////// StringArrayMask /////////////////////////////////////////

/// Represents a string array field mask for policy application.
///
/// A StringArrayMask handles the extraction of arrays of strings from
/// unstructured data, collecting all matching string values into an array.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct StringArrayMask {
    /// Index of the policy this mask belongs to
    pub policy_index: usize,
    /// Original field name from the policy definition
    pub name: String,
    /// Masked field name unlikely to be in LLM training data
    pub mask: String,
}

impl StringArrayMask {
    /// Create a new StringArrayMask with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `policy_index` - The index of the policy this mask belongs to
    /// * `name` - The original field name from the policy definition
    /// * `mask` - The masked field name unlikely to be in LLM training data
    /// * `_value` - The expected string values (currently unused)
    ///
    /// # Example
    ///
    /// ```
    /// use policyai::StringArrayMask;
    /// let mask = StringArrayMask::new(
    ///     1,
    ///     "tags".to_string(),
    ///     "field_arr789".to_string(),
    ///     vec!["tag1".to_string(), "tag2".to_string()]
    /// );
    /// ```
    pub fn new(policy_index: usize, name: String, mask: String, _value: Vec<String>) -> Self {
        Self {
            policy_index,
            name,
            mask,
        }
    }

    /// Apply this string array mask to intermediate representation data.
    ///
    /// Extracts string arrays from the IR (supporting nested arrays) and reports
    /// each individual string to the given Report.
    ///
    /// # Arguments
    ///
    /// * `ir` - The intermediate representation JSON from the LLM
    /// * `report` - The report to write results and errors to
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::{StringArrayMask, Report};
    /// # use claudius::MessageParam;
    /// let mask = StringArrayMask::new(1, "tags".to_string(), "field_arr".to_string(), vec![]);
    /// let ir = serde_json::json!({"field_arr": ["tag1", "tag2"]});
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// mask.apply_to(&ir, &mut report);
    /// ```
    pub fn apply_to(&self, ir: &serde_json::Value, report: &mut Report) {
        fn extract_strings(value: &serde_json::Value, depth: usize) -> Option<Vec<String>> {
            if depth == 0 {
                None
            } else if let serde_json::Value::String(s) = value {
                Some(vec![s.clone()])
            } else if let serde_json::Value::Array(a) = value {
                let mut all = vec![];
                for v in a {
                    all.extend(extract_strings(v, depth - 1)?.into_iter());
                }
                Some(all)
            } else {
                None
            }
        }
        if let Some(reported) = ir.get(&self.mask) {
            match extract_strings(reported, 128) {
                Some(strings) => {
                    if strings.is_empty() {
                        report.init_empty_string_array(self.policy_index, &self.name);
                    } else {
                        for s in strings {
                            report.report_string_array(self.policy_index, &self.name, s);
                        }
                    }
                }
                None => {
                    report.report_type_check_failure(
                        file!(),
                        line!(),
                        &format!("expected [string] for {}", self.name),
                    );
                }
            }
        }
    }
}

////////////////////////////////////////// StringEnumMask //////////////////////////////////////////

/// Represents a string enumeration field mask for policy application.
///
/// A StringEnumMask handles boolean-style enumeration values where the presence
/// of a specific enum value is indicated by a boolean flag in the intermediate representation.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct StringEnumMask {
    /// Index of the policy this mask belongs to
    pub policy_index: usize,
    /// Original field name from the policy definition
    pub name: String,
    /// Masked field name unlikely to be in LLM training data
    pub mask: String,
    /// The specific enum value this mask represents
    pub value: Option<String>,
    /// Default enum value when the field is not present
    pub default: Option<String>,
    /// Strategy for resolving conflicts when multiple policies set different values
    pub on_conflict: OnConflict,
}

impl StringEnumMask {
    /// Create a new StringEnumMask with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `policy_index` - The index of the policy this mask belongs to
    /// * `name` - The original field name from the policy definition
    /// * `mask` - The masked field name unlikely to be in LLM training data
    /// * `value` - The specific enum value this mask represents
    /// * `default` - The default enum value when field is absent
    /// * `on_conflict` - Strategy for resolving conflicts between policies
    ///
    /// # Example
    ///
    /// ```
    /// use policyai::{StringEnumMask, OnConflict};
    /// let mask = StringEnumMask::new(
    ///     1,
    ///     "status".to_string(),
    ///     "field_enum456".to_string(),
    ///     Some("active".to_string()),
    ///     Some("inactive".to_string()),
    ///     OnConflict::LargestValue
    /// );
    /// ```
    pub fn new(
        policy_index: usize,
        name: String,
        mask: String,
        value: Option<String>,
        default: Option<String>,
        on_conflict: OnConflict,
    ) -> Self {
        Self {
            policy_index,
            name,
            mask,
            value,
            default,
            on_conflict,
        }
    }

    /// Apply this string enum mask to intermediate representation data.
    ///
    /// Checks for a boolean flag in the IR and if true, reports the associated
    /// enum value. This supports enum fields where each possible value is
    /// represented as a separate boolean flag.
    ///
    /// # Arguments
    ///
    /// * `ir` - The intermediate representation JSON from the LLM
    /// * `report` - The report to write results and errors to
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::{StringEnumMask, OnConflict, Report};
    /// # use claudius::MessageParam;
    /// let mask = StringEnumMask::new(1, "priority".to_string(), "field_enum".to_string(), Some("high".to_string()), None, OnConflict::Default);
    /// let ir = serde_json::json!({"field_enum": true});
    /// let mut report = Report::new(vec![], vec![], vec![], vec![], vec![], vec![], vec![]);
    /// mask.apply_to(&ir, &mut report);
    /// ```
    pub fn apply_to(&self, ir: &serde_json::Value, report: &mut Report) {
        match ir.get(&self.mask) {
            Some(serde_json::Value::Bool(value)) => {
                if *value {
                    if let Some(enum_value) = &self.value {
                        report.report_string_enum(
                            self.policy_index,
                            &self.name,
                            enum_value.clone(),
                            self.on_conflict,
                        );
                    } else {
                        report.report_policy_index(self.policy_index);
                        report.report_string_enum_conflict(
                            &self.name,
                            value.to_string(),
                            "null".to_string(),
                        );
                    }
                } else if let Some(default) = self.default.as_ref() {
                    report.report_string_default(&self.name, default);
                }
            }
            Some(_) => {
                report.report_type_check_failure(
                    file!(),
                    line!(),
                    &format!("expected string for {}", self.name),
                );
            }
            _ => {
                if let Some(default) = self.default.as_ref() {
                    report.report_string_default(&self.name, default);
                }
            }
        }
    }
}
