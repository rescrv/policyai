use crate::{t64, OnConflict, Report};

///////////////////////////////////////////// BoolMask /////////////////////////////////////////////

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct BoolMask {
    pub policy_index: usize,
    pub name: String,
    pub mask: String,
    pub default: bool,
    pub is_true: bool,
    pub on_conflict: OnConflict,
}

impl BoolMask {
    pub fn new(
        policy_index: usize,
        name: String,
        mask: String,
        default: bool,
        is_true: bool,
        on_conflict: OnConflict,
    ) -> Self {
        Self {
            policy_index,
            name,
            mask,
            default,
            is_true,
            on_conflict,
        }
    }

    pub fn apply_to(&self, ir: &serde_json::Value, report: &mut Report) {
        match ir.get(&self.mask) {
            Some(serde_json::Value::Bool(ret)) => {
                if *ret == self.is_true {
                    report.report_bool(self.policy_index, &self.name, *ret, self.on_conflict);
                } else {
                    report.report_bool_default(&self.name, self.default);
                }
            }
            Some(_) => {
                report.report_type_check_failure(
                    file!(),
                    line!(),
                    &format!("expected boolean for {}", self.name),
                );
            }
            None => {
                report.report_bool_default(&self.name, self.default);
            }
        }
    }
}

//////////////////////////////////////////// NumberMask ////////////////////////////////////////////

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct NumberMask {
    pub policy_index: usize,
    pub name: String,
    pub mask: String,
    pub default: Option<t64>,
    pub value: serde_json::Number,
    pub on_conflict: OnConflict,
}

impl NumberMask {
    pub fn new(
        policy_index: usize,
        name: String,
        mask: String,
        default: Option<t64>,
        value: serde_json::Number,
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

    pub fn apply_to(&self, ir: &serde_json::Value, report: &mut Report) {
        match ir.get(&self.mask) {
            Some(serde_json::Value::Number(value)) => {
                report.report_number(
                    self.policy_index,
                    &self.name,
                    value.clone(),
                    self.on_conflict,
                );
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

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct StringMask {
    pub policy_index: usize,
    pub name: String,
    pub mask: String,
    pub default: Option<String>,
    pub value: String,
    pub on_conflict: OnConflict,
}

impl StringMask {
    pub fn new(
        policy_index: usize,
        name: String,
        mask: String,
        default: Option<String>,
        value: String,
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

    pub fn apply_to(&self, ir: &serde_json::Value, report: &mut Report) {
        match ir.get(&self.mask) {
            Some(serde_json::Value::String(value)) => {
                report.report_string(
                    self.policy_index,
                    &self.name,
                    value.clone(),
                    self.on_conflict,
                );
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

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct StringArrayMask {
    pub policy_index: usize,
    pub name: String,
    pub mask: String,
}

impl StringArrayMask {
    pub fn new(policy_index: usize, name: String, mask: String, _value: Vec<String>) -> Self {
        Self {
            policy_index,
            name,
            mask,
        }
    }

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
                    for s in strings {
                        report.report_string_array(self.policy_index, &self.name, s);
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

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct StringEnumMask {
    pub policy_index: usize,
    pub name: String,
    pub mask: String,
    pub value: String,
    pub default: Option<String>,
    pub on_conflict: OnConflict,
}

impl StringEnumMask {
    pub fn new(
        policy_index: usize,
        name: String,
        mask: String,
        value: String,
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

    pub fn apply_to(&self, ir: &serde_json::Value, report: &mut Report) {
        match ir.get(&self.mask) {
            Some(serde_json::Value::Bool(value)) => {
                if *value {
                    report.report_string_enum(
                        self.policy_index,
                        &self.name,
                        self.value.clone(),
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
