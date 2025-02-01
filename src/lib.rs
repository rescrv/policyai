use std::cmp::Ordering;
use std::collections::HashMap;

use yammer::{ollama_host, ChatMessage, JsonSchema};

pub mod data;

mod masking;
mod parser;

pub use masking::MaskGenerator;
pub use parser::ParseError;

//////////////////////////////////////////////// t64 ///////////////////////////////////////////////

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
        x.0.into()
    }
}

//////////////////////////////////////////// PolicyType ////////////////////////////////////////////

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct PolicyType {
    pub name: String,
    pub fields: Vec<Field>,
}

impl PolicyType {
    pub fn parse(input: &str) -> Result<Self, ParseError> {
        parser::parse_all(parser::policy_type)(input)
    }

    pub async fn with_semantic_injection(&self, injection: &str) -> Result<Policy, yammer::Error> {
        let mut schema = serde_json::json! {{}};
        let mut properties = serde_json::json! {{}};
        for field in self.fields.iter() {
            let (name, schema) = match field {
                Field::Bool {
                    name,
                    default: _,
                    on_conflict: _,
                } => (name.clone(), bool::json_schema()),
                Field::Number {
                    name,
                    default: _,
                    on_conflict: _,
                } => (name.clone(), f64::json_schema()),
                Field::String {
                    name,
                    default: _,
                    on_conflict: _,
                } => (name.clone(), String::json_schema()),
                Field::StringEnum {
                    name,
                    values,
                    default: _,
                    on_conflict: _,
                } => {
                    let mut schema = String::json_schema();
                    schema["enum"] = values.clone().into();
                    (name.clone(), schema)
                }
                Field::StringArray { name } => (name.clone(), Vec::<String>::json_schema()),
            };
            properties[name] = schema;
        }
        schema["required"] = serde_json::json! {[]};
        schema["type"] = "object".into();
        schema["properties"] = properties;
        let system = r#"
Your task is to assume that the following message's conditional ask is true.

Assume it is true and generate a sample response.  Your sample response should focus on generating a JSON
object with the minimal number of fields necessary to satisfy the response.

Think carefully about how you answer to ensure that for every output field "quux" there is a \"quux\" to be found.
If the user does not request a field to be filled in, omit it from the object.

Example Input: Extract the hashtags to field \"foo\" and set \"bar\" to true.
Example Output: {"foo": "\#HashTag\", "bar": true}

Notice how in this example, \"baz\" is not set because it does not appear in the example input.
"#
        .to_string();
        let req = yammer::GenerateRequest {
            model: "phi4".to_string(),
            prompt: injection.to_string(),
            system: Some(system),
            format: Some(schema),
            stream: Some(false),
            suffix: None,
            template: None,
            images: None,
            keep_alive: None,
            raw: None,
            options: Some(serde_json::json! {{
                "temperature": 0.1,
            }}),
        };
        let resp = req
            .make_request(&ollama_host(None))
            .send()
            .await?
            .error_for_status()?
            .json::<yammer::GenerateResponse>()
            .await?;
        let prompt = injection.to_string();
        let action = serde_json::from_str(&resp.response)?;
        Ok(Policy {
            r#type: self.clone(),
            prompt,
            action,
        })
    }
}

impl std::fmt::Display for PolicyType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        writeln!(f, "type {} {{", self.name)?;
        for field in self.fields.iter() {
            writeln!(f, "    {},", field)?;
        }
        write!(f, "}}")
    }
}

/////////////////////////////////////////////// Field //////////////////////////////////////////////

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Field {
    #[serde(rename = "bool")]
    Bool {
        name: String,
        default: bool,
        on_conflict: OnConflict,
    },
    #[serde(rename = "string")]
    String {
        name: String,
        default: Option<String>,
        on_conflict: OnConflict,
    },
    #[serde(rename = "enum")]
    StringEnum {
        name: String,
        values: Vec<String>,
        default: Option<String>,
        on_conflict: OnConflict,
    },
    #[serde(rename = "array")]
    StringArray { name: String },
    #[serde(rename = "number")]
    Number {
        name: String,
        default: Option<t64>,
        on_conflict: OnConflict,
    },
}

impl Field {
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
                        write!(f, "{}: bool = true", name)?;
                    } else {
                        write!(f, "{}: bool", name)?;
                    }
                }
                OnConflict::Agreement => {
                    if *default {
                        write!(f, "{}: bool @ agreement = true", name)?;
                    } else {
                        write!(f, "{}: bool @ agreement", name)?;
                    }
                }
                OnConflict::LargestValue => {
                    if *default {
                        write!(f, "{}: bool @ sticky = true", name)?;
                    } else {
                        write!(f, "{}: bool @ sticky", name)?;
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
                        write!(f, "{}: string = {default:?}", name)?;
                    } else {
                        write!(f, "{}: string", name)?;
                    }
                }
                OnConflict::Agreement => {
                    if let Some(default) = default.as_ref() {
                        write!(f, "{}: string @ agreement = {default:?}", name)?;
                    } else {
                        write!(f, "{}: string @ agreement", name)?;
                    }
                }
                OnConflict::LargestValue => {
                    if let Some(default) = default.as_ref() {
                        write!(f, "{}: string @ last wins = {default:?}", name)?;
                    } else {
                        write!(f, "{}: string @ last wins", name)?;
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
                    .map(|v| format!("{:?}", v))
                    .collect::<Vec<_>>()
                    .join(", ");
                match on_conflict {
                    OnConflict::Default => {
                        if let Some(default) = default.as_ref() {
                            write!(f, "{}: [{}] = {default:?}", name, values)?;
                        } else {
                            write!(f, "{}: [{}]", name, values)?;
                        }
                    }
                    OnConflict::Agreement => {
                        if let Some(default) = default.as_ref() {
                            write!(f, "{}: [{}] @ agreement = {default:?}", name, values)?;
                        } else {
                            write!(f, "{}: [{}] @ agreement", name, values)?;
                        }
                    }
                    OnConflict::LargestValue => {
                        if let Some(default) = default.as_ref() {
                            write!(f, "{}: [{}] @ highest wins = {default:?}", name, values)?;
                        } else {
                            write!(f, "{}: [{}] @ highest wins", name, values)?;
                        }
                    }
                }
            }
            Self::StringArray { name } => {
                write!(f, "{}: [string]", name)?;
            }
            Self::Number {
                name,
                default,
                on_conflict,
            } => match on_conflict {
                OnConflict::Default => {
                    if let Some(default) = default.as_ref() {
                        write!(f, "{}: number = {default:?}", name)?;
                    } else {
                        write!(f, "{}: number", name)?;
                    }
                }
                OnConflict::Agreement => {
                    if let Some(default) = default.as_ref() {
                        write!(f, "{}: number @ agreement = {default:?}", name)?;
                    } else {
                        write!(f, "{}: number @ agreement", name)?;
                    }
                }
                OnConflict::LargestValue => {
                    if let Some(default) = default.as_ref() {
                        write!(f, "{}: number @ last wins = {default:?}", name)?;
                    } else {
                        write!(f, "{}: number @ last wins", name)?;
                    }
                }
            },
        }
        Ok(())
    }
}

//////////////////////////////////////////// OnConflict ////////////////////////////////////////////

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum OnConflict {
    #[default]
    #[serde(rename = "default")]
    Default,
    #[serde(rename = "agreement")]
    Agreement,
    #[serde(rename = "largest")]
    LargestValue,
}

////////////////////////////////////////////// Policy //////////////////////////////////////////////

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Policy {
    pub r#type: PolicyType,
    pub prompt: String,
    pub action: serde_json::Value,
}

//////////////////////////////////////////// PolicyError ///////////////////////////////////////////

#[derive(Clone, Debug)]
pub enum PolicyError {
    Inconsistent {
        field: String,
        expected: Option<serde_json::Value>,
        returned: Option<serde_json::Value>,
    },
    ActionOmitted {
        field_name: String,
        action: serde_json::Value,
    },
    ExpectedBool {
        field_name: String,
    },
    ExpectedNumber {
        field_name: String,
    },
    ExpectedString {
        field_name: String,
    },
}

////////////////////////////////////////////// Report //////////////////////////////////////////////

#[derive(serde::Deserialize, serde::Serialize)]
pub struct Report {
    messages: Vec<ChatMessage>,
    bool_mask: HashMap<String, String>,
    number_mask: HashMap<String, String>,
    string_mask: HashMap<String, String>,
    string_array_mask: HashMap<String, String>,
    string_enum_mask: HashMap<String, String>,
    ir: Option<serde_json::Value>,
    default: Option<serde_json::Value>,
    value: Option<serde_json::Value>,
}

impl Report {
    pub fn value(&self) -> serde_json::Value {
        let mut value = self.default.clone().unwrap_or(serde_json::json! {{}});
        if let Some(serde_json::Value::Object(obj)) = self.value.as_ref() {
            for (k, v) in obj.iter() {
                value[k.clone()] = v.clone();
            }
        }
        value
    }

    pub fn report_bool_default(&mut self, field: &str, default: bool) {
        let build = self.default.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if build.get(field).is_none() {
            build[field.to_string()] = default.into();
        }
    }

    pub fn report_bool(&mut self, field: &str, value: bool, on_conflict: OnConflict) {
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
                                } else {
                                    let b = *b;
                                    self.report_bool_conflict(field, b, value);
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
            if *existing != serde_json::Value::Number(default) {
                todo!();
            }
        } else {
            build[field.to_string()] = default.into();
        }
    }

    pub fn report_number(
        &mut self,
        field: &str,
        value: impl Into<serde_json::Number>,
        on_conflict: OnConflict,
    ) {
        let value = value.into();
        let build = self.value.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(v) = build.get_mut(field) {
            match v {
                serde_json::Value::Null => *v = value.into(),
                serde_json::Value::Number(n) => {
                    fn is_equal(lhs: &serde_json::Number, rhs: &serde_json::Number) -> bool {
                        if lhs.is_f64() && rhs.is_f64() {
                            lhs.as_f64() == rhs.as_f64()
                        } else if lhs.is_u64() && rhs.is_u64() {
                            lhs.as_u64() == rhs.as_u64()
                        } else if lhs.is_i64() && rhs.is_i64() {
                            lhs.as_i64() == rhs.as_i64()
                        } else {
                            false
                        }
                    }
                    fn less_than(lhs: &serde_json::Number, rhs: &serde_json::Number) -> bool {
                        if lhs.is_f64() && rhs.is_f64() {
                            lhs.as_f64() < rhs.as_f64()
                        } else if lhs.is_u64() && rhs.is_u64() {
                            lhs.as_u64() < rhs.as_u64()
                        } else if lhs.is_i64() && rhs.is_i64() {
                            lhs.as_i64() < rhs.as_i64()
                        } else {
                            false
                        }
                    }
                    if is_equal(&*n, &value) {
                        match on_conflict {
                            OnConflict::Default => {}
                            OnConflict::Agreement => {
                                let n = n.clone();
                                self.report_number_conflict(field, n, value);
                            }
                            OnConflict::LargestValue => {
                                if less_than(&*n, &value) {
                                    *n = value;
                                } else {
                                    let n = n.clone();
                                    self.report_number_conflict(field, n, value);
                                }
                            }
                        }
                    }
                }
                serde_json::Value::Bool(_) => {
                    self.report_invariant_violation(
                        file!(),
                        line!(),
                        "bool found in place of string",
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
                        "array found in place of string",
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

    pub fn report_string_default(&mut self, field: &str, default: impl Into<String>) {
        let default = default.into();
        let build = self.default.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(existing) = build.get(field) {
            if *existing != serde_json::Value::String(default) {
                todo!();
            }
        } else {
            build[field.to_string()] = default.into();
        }
    }

    pub fn report_string(&mut self, field: &str, value: String, on_conflict: OnConflict) {
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
                        "bool found in place of string",
                    );
                }
                serde_json::Value::Number(_) => {
                    self.report_invariant_violation(
                        file!(),
                        line!(),
                        "number found in place of string",
                    );
                }
                serde_json::Value::Array(_) => {
                    self.report_invariant_violation(
                        file!(),
                        line!(),
                        "array found in place of string",
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

    pub fn report_string_enum(
        &mut self,
        field: &str,
        value: String,
        values: &[String],
        on_conflict: OnConflict,
    ) {
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
                        "bool found in place of string",
                    );
                }
                serde_json::Value::Number(_) => {
                    self.report_invariant_violation(
                        file!(),
                        line!(),
                        "number found in place of string",
                    );
                }
                serde_json::Value::Array(_) => {
                    self.report_invariant_violation(
                        file!(),
                        line!(),
                        "array found in place of string",
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

    pub fn report_string_array(&mut self, field: &str, value: String) {
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

    fn report_invariant_violation(&mut self, file: &str, line: u32, message: &str) {}

    fn report_type_check_failure(&mut self, file: &str, line: u32, message: &str) {}

    fn report_bool_conflict(&mut self, field: &str, val1: bool, val2: bool) {}
    fn report_string_conflict(&mut self, field: &str, val1: String, val2: String) {}
    fn report_number_conflict(
        &mut self,
        field: &str,
        val1: serde_json::Number,
        val2: serde_json::Number,
    ) {
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

/////////////////////////////////////////// ReportBuilder //////////////////////////////////////////

#[derive(Default)]
pub struct ReportBuilder {
    mask_gen: MaskGenerator,
    bool_mask: HashMap<(String, bool), String>,
    number_mask: HashMap<String, String>,
    string_mask: HashMap<String, String>,
    string_array_mask: HashMap<String, String>,
    string_enum_mask: HashMap<(String, String), String>,
    #[allow(clippy::type_complexity)]
    masks: Vec<(String, Box<dyn Fn(&mut Report, serde_json::Value)>)>,
    messages: Vec<ChatMessage>,
    policy_index: usize,
    required: Vec<String>,
    properties: serde_json::Value,
}

impl ReportBuilder {
    pub fn add_policy(&mut self, policy: &Policy) -> Result<(), PolicyError> {
        // Assume default=0, so we increment and take, but we increment at the end.
        let mut content = format!("Rule #{}:\n{}", self.policy_index, policy.prompt);
        let mut sample = serde_json::Map::default();
        #[allow(clippy::type_complexity)]
        let mut masks: Vec<(String, Box<dyn Fn(&mut Report, serde_json::Value)>)> = vec![];
        let mut exemplar = serde_json::json! {{}};
        let mut required = self.required.clone();
        let mut properties = self.properties.clone();
        for field in policy.r#type.fields.iter() {
            let Some(value) = policy.action.get(field.name()) else {
                continue;
            };
            match field {
                Field::Bool {
                    name,
                    default,
                    on_conflict,
                } => {
                    let serde_json::Value::Bool(v) = value else {
                        return Err(PolicyError::ExpectedBool {
                            field_name: name.to_string(),
                        });
                    };
                    let mask = self
                        .bool_mask
                        .entry((name.to_string(), *v))
                        .or_insert_with(|| {
                            let mask = self.mask_gen.generate();
                            self.messages.push(ChatMessage {
                                role: "system".to_string(),
                                content: format!(
                                    "Unless specified by a numbered rule, set {mask:?} to false."
                                ),
                                images: None,
                                tool_calls: None,
                            });
                            if *v {
                                sample.insert(name.to_string(), true.into());
                            } else {
                                sample.insert(name.to_string(), false.into());
                            }
                            mask
                        });
                    if *v {
                        content += &format!(
                            r#"  If the rule applies, indicate {name:?} is true by outputting {{{mask:?}: true}} instead."#
                        );
                    } else {
                        content += &format!(
                            r#"  If the rule applies, indicate {name:?} is false by outputting {{{mask:?}: true}} instead."#
                        );
                    }
                    exemplar[mask.clone()] = true.into();
                    required.push(mask.clone());
                    properties[mask.clone()] = bool::json_schema();
                    let name = name.to_string();
                    let default = *default;
                    let is_true = *v;
                    let on_conflict = *on_conflict;
                    masks.push((
                        mask.clone(),
                        Box::new(move |report, reported| {
                            if let serde_json::Value::Bool(true) = reported {
                                if is_true {
                                    report.report_bool(&name, true, on_conflict);
                                } else {
                                    report.report_bool(&name, false, on_conflict);
                                }
                            } else {
                                report.report_type_check_failure(
                                    file!(),
                                    line!(),
                                    &format!("expected boolean for {name}"),
                                );
                            }
                            report.report_bool_default(&name, default);
                        }),
                    ));
                }
                Field::Number {
                    name,
                    default,
                    on_conflict,
                } => {
                    let mask = self
                        .number_mask
                        .entry(name.to_string())
                        .or_insert_with(|| self.mask_gen.generate());
                    content = content.replace(&format!("{name:?}"), &format!("{mask:?}"));
                    sample.insert(name.to_string(), value.clone());
                    exemplar[mask.clone()] = serde_json::Value::Null;
                    required.push(mask.clone());
                    properties[mask.clone()] = f64::json_schema();
                    let name = name.clone();
                    let default = *default;
                    let on_conflict = *on_conflict;
                    masks.push((
                        mask.clone(),
                        Box::new(move |report, reported| {
                            if let serde_json::Value::Number(n) = reported {
                                report.report_number(&name, n, on_conflict);
                            } else {
                                report.report_type_check_failure(
                                    file!(),
                                    line!(),
                                    &format!("expected number for {name}"),
                                );
                            }
                            if let Some(default) = default {
                                if let Some(default) = serde_json::Number::from_f64(default.0) {
                                    report.report_number_default(&name, default);
                                } else {
                                    report.report_invariant_violation(
                                        file!(),
                                        line!(),
                                        "cannot cast to number",
                                    );
                                }
                            }
                        }),
                    ));
                }
                Field::String {
                    name,
                    default,
                    on_conflict,
                } => {
                    let mask = self
                        .string_mask
                        .entry(name.to_string())
                        .or_insert_with(|| self.mask_gen.generate());
                    content = content.replace(&format!("{name:?}"), &format!("{mask:?}"));
                    sample.insert(name.to_string(), value.clone());
                    exemplar[mask.clone()] = serde_json::Value::Null;
                    required.push(mask.clone());
                    properties[mask.clone()] = String::json_schema();
                    let name = name.clone();
                    let default = default.clone();
                    let on_conflict = *on_conflict;
                    masks.push((
                        mask.clone(),
                        Box::new(move |report, reported| {
                            if let serde_json::Value::String(s) = reported {
                                report.report_string(&name, s, on_conflict);
                            } else {
                                report.report_type_check_failure(
                                    file!(),
                                    line!(),
                                    &format!("expected string for {name}"),
                                );
                            }
                            if let Some(default) = &default {
                                report.report_string_default(&name, default);
                            }
                        }),
                    ));
                }
                Field::StringEnum {
                    name,
                    values,
                    default,
                    on_conflict,
                } => {
                    for v in values {
                        if v != value {
                            continue;
                        }
                        let mask = self
                            .string_enum_mask
                            .entry((name.to_string(), v.to_string()))
                            .or_insert_with(|| {
                                let mask = self.mask_gen.generate();
                                self.messages.push(ChatMessage {
                                    role: "system".to_string(),
                                    content: format!("Unless specified by a numbered rule, output {{{mask:?}: false}}."),
                                    images: None,
                                    tool_calls: None,
                                });
                                mask
                            });
                        exemplar[mask.clone()] = true.into();
                        required.push(mask.clone());
                        properties[mask.clone()] = bool::json_schema();
                        let name = name.clone();
                        let default = default.clone();
                        let values = values.clone();
                        let on_conflict = *on_conflict;
                        let v = v.clone();
                        masks.push((
                            mask.to_string(),
                            Box::new(move |report, reported| {
                                if let serde_json::Value::Bool(true) = reported {
                                    report.report_string_enum(
                                        &name,
                                        v.clone(),
                                        &values,
                                        on_conflict,
                                    );
                                } else {
                                    report.report_type_check_failure(
                                        file!(),
                                        line!(),
                                        &format!("expected string for {name}"),
                                    );
                                }
                                if let Some(default) = &default {
                                    report.report_string_default(&name, default);
                                }
                            }),
                        ));
                    }
                }
                Field::StringArray { name } => {
                    let mask = self
                        .string_array_mask
                        .entry(name.to_string())
                        .or_insert_with(|| self.mask_gen.generate());
                    content = content.replace(&format!("{name:?}"), &format!("{mask:?}"));
                    sample.insert(name.to_string(), value.clone());
                    exemplar[mask.clone()] = value.clone();
                    required.push(mask.clone());
                    properties[mask.clone()] = Vec::<String>::json_schema();
                    let name = name.clone();
                    masks.push((
                        mask.clone(),
                        Box::new(move |report, reported| {
                            fn extract_strings(
                                value: &serde_json::Value,
                                depth: usize,
                            ) -> Option<Vec<String>> {
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
                            match extract_strings(&reported, 128) {
                                Some(strings) => {
                                    for s in strings {
                                        report.report_string_array(&name, s);
                                    }
                                }
                                None => {
                                    report.report_type_check_failure(
                                        file!(),
                                        line!(),
                                        &format!("expected [string] for {name}"),
                                    );
                                }
                            }
                        }),
                    ));
                }
            }
        }
        for (key, returned) in sample.iter() {
            let expected = policy.action.get(key);
            if expected != Some(returned) {
                let field = key.clone();
                let expected = expected.cloned();
                let returned = Some(returned.clone());
                return Err(PolicyError::Inconsistent {
                    field,
                    expected,
                    returned,
                });
            }
        }
        self.policy_index += 1;
        self.masks.extend(masks);
        content += &format!(
            "  Output the specific JSON (removing null placeholders and inserting values): {}",
            serde_json::to_string(&exemplar).unwrap()
        );
        self.messages.push(ChatMessage {
            role: "system".to_string(),
            content: content.clone(),
            images: None,
            tool_calls: None,
        });
        required.push("rules".into());
        required.push("justification".into());
        properties["rules"] = Vec::<String>::json_schema();
        properties["justification"] = String::json_schema();
        self.required = required;
        self.properties = properties;
        Ok(())
    }

    fn consume_ir(self, ir: serde_json::Value) -> Result<Report, ApplyError> {
        let mut report = Report {
            messages: self.messages,
            bool_mask: self
                .bool_mask
                .into_iter()
                .map(|(k, v)| (v, format!("{}::{}", k.0, k.1)))
                .collect(),
            number_mask: self.number_mask.into_iter().map(|(k, v)| (v, k)).collect(),
            string_mask: self.string_mask.into_iter().map(|(k, v)| (v, k)).collect(),
            string_array_mask: self
                .string_array_mask
                .into_iter()
                .map(|(k, v)| (v, k))
                .collect(),
            string_enum_mask: self
                .string_enum_mask
                .into_iter()
                .map(|(k, v)| (v, format!("{}::{}", k.0, k.1)))
                .collect(),
            ir: Some(ir.clone()),
            default: None,
            value: None,
        };
        for (mask, action) in self.masks.iter() {
            let Some(value) = ir.get(mask) else {
                return Err(ApplyError::Conflict(Conflict::TODO));
            };
            action(&mut report, value.clone());
        }
        Ok(report)
    }

    fn messages(&self) -> Vec<ChatMessage> {
        self.messages.clone()
    }

    fn schema(&self) -> serde_json::Value {
        let mut schema = serde_json::json! {{}};
        schema["type"] = "object".into();
        schema["required"] = self.required.clone().into();
        schema["properties"] = self.properties.clone();
        schema
    }
}

impl std::fmt::Debug for Report {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        f.debug_struct("Report").finish_non_exhaustive()
    }
}

//////////////////////////////////////////// ApplyError ////////////////////////////////////////////

#[derive(Debug)]
pub enum ApplyError {
    Policy(PolicyError),
    Yammer(yammer::Error),
    Conflict(Conflict),
}

impl From<PolicyError> for ApplyError {
    fn from(err: PolicyError) -> Self {
        Self::Policy(err)
    }
}

impl<T: Into<yammer::Error>> From<T> for ApplyError {
    fn from(err: T) -> Self {
        Self::Yammer(err.into())
    }
}

///////////////////////////////////////////// Conflict /////////////////////////////////////////////

#[derive(Debug)]
pub enum Conflict {
    TODO,
    Disagree {
        name: String,
        value1: serde_json::Value,
        value2: serde_json::Value,
    },
}

////////////////////////////////////////////// Manager /////////////////////////////////////////////

#[derive(Debug, Default)]
pub struct Manager {
    policies: Vec<Policy>,
}

impl Manager {
    pub fn add(&mut self, policy: Policy) {
        if let Some(last) = self.policies.last() {
            assert_eq!(last.r#type, policy.r#type);
        }
        self.policies.push(policy);
    }

    pub async fn apply(
        &mut self,
        host: Option<String>,
        template: yammer::ChatRequest,
        prompt: &str,
    ) -> Result<Report, ApplyError> {
        let (report, req) = self.request_for(template, prompt).await?;
        let resp = req
            .make_request(&ollama_host(host))
            .send()
            .await?
            .error_for_status()?
            .json::<yammer::ChatResponse>()
            .await?;
        let ir: serde_json::Value = serde_json::from_str(&resp.message.content)?;
        report.consume_ir(ir)
    }

    pub async fn request_for(
        &mut self,
        template: yammer::ChatRequest,
        prompt: &str,
    ) -> Result<(ReportBuilder, yammer::ChatRequest), ApplyError> {
        let mut req = template;
        req.messages = vec![ChatMessage {
            role: "system".to_string(),
            content: r#"
You are a rule application machine.

Your task is to use a numbered list of policies that describe how to extract JSON structure from
unstructured data.  Each rule makes a statement about the nature of the input that matches and
what is expected of the output.

Because you are extracting JSON structure for the user, you will respond in JSON.
"#
            .to_string(),
            images: None,
            tool_calls: None,
        }];
        let mut report = ReportBuilder::default();
        for policy in self.policies.iter() {
            report.add_policy(policy)?;
        }
        req.messages.extend(report.messages());
        req.messages.push(ChatMessage {
            role: "user".to_string(),
            content: format!("Unstructured Data: {prompt}"),
            images: None,
            tool_calls: None,
        });
        req.format = Some(report.schema());
        req.stream = Some(false);
        req.tools = None;
        Ok((report, req))
    }
}

/////////////////////////////////////////////// tests //////////////////////////////////////////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readme() {
        let policy = PolicyType {
            name: "policyai::EmailPolicy".to_string(),
            fields: vec![
                Field::Bool {
                    name: "unread".to_string(),
                    default: true,
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
            format!("{}", policy)
        );
    }

    #[tokio::test]
    async fn with_semantic_injection() {
        let policy = PolicyType {
            name: "policyai::EmailPolicy".to_string(),
            fields: vec![
                Field::Bool {
                    name: "unread".to_string(),
                    default: true,
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
        let policy = PolicyType {
            name: "policyai::EmailPolicy".to_string(),
            fields: vec![Field::Number {
                name: "weight".to_string(),
                default: None,
                on_conflict: OnConflict::Default,
            }],
        };
        let policy = policy
            .with_semantic_injection("Assign weight to the email.")
            .await
            .unwrap();
        assert!(matches!(
            policy.action.get("weight"),
            Some(serde_json::Value::Number(_))
        ));
    }

    #[tokio::test]
    async fn apply_readme_policy() {
        let policy = PolicyType {
            name: "policyai::EmailPolicy".to_string(),
            fields: vec![
                Field::Bool {
                    name: "unread".to_string(),
                    default: true,
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
            .with_semantic_injection("Set \"priority\" to low and \"unread\" to true.")
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
                None,
                yammer::ChatRequest {
                    model: "phi4".to_string(),
                    format: None,
                    keep_alive: None,
                    messages: vec![],
                    tools: None,
                    stream: None,
                    options: serde_json::json! {{
                        "temperature": 0.1,
                    }},
                },
                r#"From: robert@example.org
To: jeff@example.org

This is an email about AI.
"#,
            )
            .await
            .expect("manager should produce a JSON value");
        println!("{report}");
        assert_eq!(
            serde_json::json! {{"priority": "low", "unread": true}},
            report.value()
        );
    }
}
