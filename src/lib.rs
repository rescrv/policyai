use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;

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
            model: "phi4:14b-fp16".to_string(),
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
                "num_ctx": 16_000,
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
    Inconsistent { field: Field },
    ExpectedBool { field_name: String },
    ExpectedNumber { field_name: String },
    ExpectedString { field_name: String },
}

/////////////////////////////////////////// SemanticMask ///////////////////////////////////////////

pub trait SemanticMask {}

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
        /*
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
        */
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
        /*
        match ir.get(&self.mask) {
            Some(serde_json::Value::String(value)) => {
                report.report_string(self.policy_index, &self.name, value, self.on_conflict);
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
        */
    }
}

/*
                           if let serde_json::Value::String(s) = reported {
                           } else {
                           }
                           if let Some(default) = &default {
                           }
*/

////////////////////////////////////////// StringArrayMask /////////////////////////////////////////

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct StringArrayMask {
    pub policy_index: usize,
    pub name: String,
    pub mask: String,
}

impl StringArrayMask {
    pub fn new(policy_index: usize, name: String, mask: String, value: Vec<String>) -> Self {
        Self {
            policy_index,
            name,
            mask,
        }
    }

    pub fn apply_to(&self, report: &mut Report) {
        todo!();
    }
}

/*
                   masks.push((
                       mask.clone(),
                       Arc::new(move |report, reported| {
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
                                       report.report_string_array(policy_index, &name, s);
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
*/

////////////////////////////////////////// StringEnumMask //////////////////////////////////////////

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct StringEnumMask {
    pub policy_index: usize,
    pub name: String,
    pub mask: String,
}

impl StringEnumMask {
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
        }
    }

    pub fn apply_to(&self, report: &mut Report) {
        todo!();
    }
}

/*
    masks.push((
        mask.to_string(),
        Arc::new(move |report, reported| {
            if let serde_json::Value::Bool(true) = reported {
                report.report_string_enum(
                    policy_index,
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
*/

////////////////////////////////////////////// Report //////////////////////////////////////////////

#[derive(serde::Deserialize, serde::Serialize)]
pub struct Report {
    pub messages: Vec<ChatMessage>,
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

    fn report_bool_default(&mut self, field: &str, default: bool) {
        let build = self.default.get_or_insert_with(|| {
            serde_json::json! {{}}
        });
        if let Some(existing) = build.get(field) {
            if *existing != serde_json::Value::Bool(default) {
                todo!();
            }
        } else {
            build[field.to_string()] = default.into();
        }
    }

    fn report_bool(
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

    fn report_number_default(&mut self, field: &str, default: impl Into<serde_json::Number>) {
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

    fn report_number(
        &mut self,
        policy_index: usize,
        field: &str,
        value: impl Into<serde_json::Number>,
        on_conflict: OnConflict,
    ) {
        self.report_policy_index(policy_index);
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
                            // TODO(rescrv):  We can do better by considering all possibilities.
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
                            // TODO(rescrv):  We can do better by considering all possibilities.
                            false
                        }
                    }
                    if !is_equal(&*n, &value) {
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
                        "bool found in place of number",
                    );
                }
                serde_json::Value::String(_) => {
                    self.report_invariant_violation(
                        file!(),
                        line!(),
                        "string found in place of number",
                    );
                }
                serde_json::Value::Array(_) => {
                    self.report_invariant_violation(
                        file!(),
                        line!(),
                        "array found in place of number",
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

    fn report_string_default(&mut self, field: &str, default: impl Into<String>) {
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

    fn report_string(
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

    fn report_string_enum(
        &mut self,
        policy_index: usize,
        field: &str,
        value: String,
        values: &[String],
        on_conflict: OnConflict,
    ) {
        self.report_policy_index(policy_index);
        if !values.contains(&value) {
            todo!();
        }
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

    fn report_string_array(&mut self, policy_index: usize, field: &str, value: String) {
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

    fn report_invariant_violation(&mut self, _file: &str, _line: u32, _message: &str) {
        panic!();
    }

    fn report_type_check_failure(&mut self, _file: &str, _line: u32, _message: &str) {
        panic!();
    }

    fn report_bool_conflict(&mut self, _field: &str, _val1: bool, _val2: bool) {
        panic!();
    }

    fn report_string_conflict(&mut self, _field: &str, _val1: String, _val2: String) {
        panic!();
    }

    fn report_number_conflict(
        &mut self,
        _field: &str,
        _val1: serde_json::Number,
        _val2: serde_json::Number,
    ) {
        panic!();
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

#[derive(Clone)]
pub struct ReportBuilder {
    mask_index: usize,
    mask_gen: MaskGenerator,
    bool_masks: Vec<BoolMask>,
    number_masks: Vec<NumberMask>,
    string_masks: Vec<StringMask>,
    string_array_masks: Vec<StringArrayMask>,
    string_enum_masks: Vec<StringEnumMask>,
    masks_by_index: Vec<Vec<String>>,
    default_return: serde_json::Value,
    messages: Vec<ChatMessage>,
    policy_index: usize,
    required: Vec<String>,
    properties: serde_json::Value,
}

impl ReportBuilder {
    pub fn add_policy(&mut self, policy: &Policy) -> Result<(), PolicyError> {
        // Assume default=0, so we increment mask_index here (in case we throw out parts of it) and
        // increment policy_index at the end when we "commit".
        self.mask_index += 1;
        let mut content = format!(
            "Rule #{}:
Criteria: {}",
            self.policy_index, policy.prompt
        );
        #[allow(clippy::type_complexity)]
        let mut required = self.required.clone();
        let mut properties = self.properties.clone();
        let mut bool_masks = self.bool_masks.clone();
        let mut number_masks = self.number_masks.clone();
        let mut string_masks = self.string_masks.clone();
        let mut string_array_masks = self.string_array_masks.clone();
        let mut string_enum_masks = self.string_enum_masks.clone();
        let mut masks_by_index = self.masks_by_index.clone();
        masks_by_index.push(vec![]);
        // SAFETY(rescrv):  We push, we pop, we never stop.
        let masks = masks_by_index.last_mut().unwrap();
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
                            field_name: name.clone(),
                        });
                    };
                    let mask = self.mask_gen.generate();
                    masks.push(mask.clone());
                    bool_masks.push(BoolMask::new(
                        self.policy_index,
                        name.clone(),
                        mask.clone(),
                        *default,
                        *v,
                        *on_conflict,
                    ));
                    self.default_return[mask.clone()] = (!*v).into();
                    content = content.replace(&format!("{name:?}"), &format!("{mask:?}"));
                    required.push(mask.clone());
                    properties[mask] = bool::json_schema();
                }
                Field::Number {
                    name,
                    default,
                    on_conflict,
                } => {
                    let serde_json::Value::Number(v) = value else {
                        return Err(PolicyError::ExpectedNumber {
                            field_name: name.clone(),
                        });
                    };
                    let mask = self.mask_gen.generate();
                    masks.push(mask.clone());
                    number_masks.push(NumberMask::new(
                        self.policy_index,
                        name.clone(),
                        mask.clone(),
                        *default,
                        v.clone(),
                        *on_conflict,
                    ));
                    self.default_return[mask.clone()] = serde_json::Value::Number(v.clone());
                    content = content.replace(&format!("{name:?}"), &format!("{mask:?}"));
                    required.push(mask.clone());
                    properties[mask] = f64::json_schema();
                }
                Field::String {
                    name,
                    default,
                    on_conflict,
                } => {
                    let serde_json::Value::String(v) = value else {
                        return Err(PolicyError::ExpectedString {
                            field_name: name.clone(),
                        });
                    };
                    let mask = self.mask_gen.generate();
                    masks.push(mask.clone());
                    string_masks.push(StringMask::new(
                        self.policy_index,
                        name.clone(),
                        mask.clone(),
                        default.clone(),
                        v.clone(),
                        *on_conflict,
                    ));
                    self.default_return[mask.clone()] = serde_json::Value::String(v.clone());
                    content = content.replace(&format!("{name:?}"), &format!("{mask:?}"));
                    required.push(mask.clone());
                    properties[mask] = f64::json_schema();
                }
                Field::StringArray { name } => {
                    let serde_json::Value::Array(v) = value else {
                        return Err(PolicyError::ExpectedString {
                            field_name: name.clone(),
                        });
                    };
                    let mut strings = vec![];
                    for v in v {
                        if let serde_json::Value::String(v) = v {
                            strings.push(v.clone());
                        } else {
                            todo!();
                        }
                    }
                    let mask = self.mask_gen.generate();
                    masks.push(mask.clone());
                    string_array_masks.push(StringArrayMask::new(
                        self.policy_index,
                        name.clone(),
                        mask.clone(),
                        strings,
                    ));
                    self.default_return[mask.clone()] = serde_json::Value::Array(vec![]);
                    content = content.replace(&format!("{name:?}"), &format!("{mask:?}"));
                    required.push(mask.clone());
                    properties[mask] = Vec::<String>::json_schema();
                }
                Field::StringEnum {
                    name,
                    values,
                    default,
                    on_conflict,
                } => {
                    let Some(v) = values.iter().find(|x| *x == value) else {
                        todo!();
                    };
                    let mask = self.mask_gen.generate();
                    masks.push(mask.clone());
                    string_enum_masks.push(StringEnumMask::new(
                        self.policy_index,
                        name.clone(),
                        mask.clone(),
                        default.clone(),
                        v.clone(),
                        *on_conflict,
                    ));
                    self.default_return[mask.clone()] = false.into();
                    content = content.replace(&format!("{name:?}"), &format!("{mask:?}"));
                    content = content.replace(&format!("{v:?}"), "true");
                    required.push(mask.clone());
                    properties[mask] = bool::json_schema();
                }
            }
        }
        self.messages.push(ChatMessage {
            role: "system".to_string(),
            content: content.clone(),
            images: None,
            tool_calls: None,
        });
        self.policy_index += 1;
        self.required = required;
        self.properties = properties;
        self.bool_masks = bool_masks;
        self.number_masks = number_masks;
        self.string_masks = string_masks;
        self.string_array_masks = string_array_masks;
        self.string_enum_masks = string_enum_masks;
        self.masks_by_index = masks_by_index;
        Ok(())
    }

    fn consume_ir(self, ir: serde_json::Value) -> Result<Report, ApplyError> {
        let mut report = Report {
            messages: self.messages,
            bool_masks: self.bool_masks,
            number_masks: self.number_masks,
            string_masks: self.string_masks,
            string_array_masks: self.string_array_masks,
            string_enum_masks: self.string_enum_masks,
            masks_by_index: self.masks_by_index,
            rules_matched: vec![],
            ir: Some(ir.clone()),
            default: None,
            value: None,
        };
        for m in report.bool_masks.clone().into_iter() {
            m.apply_to(&ir, &mut report);
        }
        for m in report.number_masks.clone().into_iter() {
            m.apply_to(&ir, &mut report);
        }
        for m in report.string_masks.clone().into_iter() {
            m.apply_to(&ir, &mut report);
        }
        for m in report.string_array_masks.clone().into_iter() {
            m.apply_to(&mut report);
        }
        for m in report.string_enum_masks.clone().into_iter() {
            m.apply_to(&mut report);
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

impl Default for ReportBuilder {
    fn default() -> ReportBuilder {
        ReportBuilder {
            mask_index: 1,
            mask_gen: MaskGenerator::default(),
            bool_masks: vec![],
            number_masks: vec![],
            string_masks: vec![],
            string_array_masks: vec![],
            string_enum_masks: vec![],
            masks_by_index: vec![],
            default_return: serde_json::json! {{}},
            messages: vec![],
            policy_index: 1,
            required: vec![
                "__rule_numbers__".to_string(),
                "__justification__".to_string(),
            ],
            properties: serde_json::json! {{
                "__rule_numbers__": Vec::<f64>::json_schema(),
                "__justification__": String::json_schema(),
            }},
        }
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
    TooManyIterations,
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
        unstructured_data: &str,
    ) -> Result<Report, ApplyError> {
        let (report, mut req) = self.request_for(template, unstructured_data).await?;
        for _ in 0..3 {
            let resp = req
                .make_request(&ollama_host(host.clone()))
                .send()
                .await?
                .error_for_status()?
                .json::<yammer::ChatResponse>()
                .await?;
            let mut ir: serde_json::Value = serde_json::from_str(&resp.message.content)?;
            let Some(reportedly_matched) = ir.get("__rule_numbers__").cloned() else {
                continue;
            };
            let Some(mut reportedly_matched): Option<Vec<usize>> =
                serde_json::from_value(reportedly_matched).ok()
            else {
                continue;
            };
            let report = report.clone().consume_ir(ir.clone())?;
            let mut empirically_matched = report.rules_matched.clone();
            empirically_matched.sort();
            empirically_matched.dedup();
            reportedly_matched.sort();
            reportedly_matched.dedup();
            if *empirically_matched == reportedly_matched {
                return Ok(report);
            }
            let empirical_but_not_reported = empirically_matched
                .iter()
                .filter(|x| !reportedly_matched.iter().any(|y| **x == *y))
                .cloned()
                .collect::<Vec<_>>();
            let reported_but_not_empirical = reportedly_matched
                .iter()
                .filter(|x| !empirically_matched.iter().any(|y| **x == *y))
                .cloned()
                .collect::<Vec<_>>();
            ir["__justification__"] = "<omitted>".to_string().into();
            req.messages.push(ChatMessage {
                role: "assistant".to_string(),
                content: format!("```json\n{}\n```\n", serde_json::to_string(&ir).unwrap()),
                images: None,
                tool_calls: None,
            });
            let mut content =
                "Your output is inconsistent and I reject it with a request for you to try again."
                    .to_string();
            if !empirical_but_not_reported.is_empty() {
                content += "\n\nYou took action on the following rules but did not report them in \"__rule_numbers__\":\n";
                for rule_number in empirical_but_not_reported.into_iter() {
                    if rule_number > 0 && rule_number <= report.masks_by_index.len() {
                        content += &format!(
                            "- Rule {}: Either change {:?} to a different value or append {} to \"__rule_numbers__\".\n",
                            rule_number,
                            report.masks_by_index[rule_number - 1],
                            rule_number
                        );
                    } else {
                        content += "- Rule number {} doesn't exist.\n";
                    }
                }
            }
            if !reported_but_not_empirical.is_empty() {
                content += "\n\nYou reported the following rules but did not perform their associated actions:\n";
                for rule_number in reported_but_not_empirical.into_iter() {
                    if rule_number > 0 && rule_number <= report.masks_by_index.len() {
                        content += &format!(
                            "- Rule {}: Either change {:?} or remove {} from \"__rule_numbers__\".\n",
                            rule_number,
                            report.masks_by_index[rule_number - 1],
                            rule_number
                        );
                    } else {
                        content += "- Rule number {} doesn't exist.\n";
                    }
                }
            }
            content += "
Please correct all mistakes and output the entire, corrected object as JSON.
Do not simply return your previous answer because I will reject it and ask you to try again.";
            req.messages.push(ChatMessage {
                role: "user".to_string(),
                content,
                images: None,
                tool_calls: None,
            });
        }
        Err(ApplyError::TooManyIterations)
    }

    pub async fn request_for(
        &mut self,
        template: yammer::ChatRequest,
        prompt: &str,
    ) -> Result<(ReportBuilder, yammer::ChatRequest), ApplyError> {
        let mut report = ReportBuilder::default();
        for policy in self.policies.iter() {
            report.add_policy(policy)?;
        }
        let mut req = template;
        req.messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: r#"
You are tasked with extracting structure from unstructured data.

You will be provided a series of rules specifying criteria about UNSTRUCTURED DATA.
For each rule, there are zero or more associated outputs.

Respond in JSON.

Detailed Instructions:
1.  Locate all default instructions and prepare to follow them.
2.  For each instruction below, consider it carefully.
    a.  For Rules:  Check that the rule's criteria describes UNSTRUCTURED DATA.
        i.  If the rule describes UNSTRUCTURED DATA, decide how to output the fact that the rule
            matches.  The instructions and instructions alone portray this information.
            Output the associated output.  Add the output to the __rule_numbers__.
        ii. If the rule does not describe UNSTRUCTURED DATA, do not follow any instructions
            pertaning to the rule.
3.  Multiple rules may match.  Repeat instruction 2 until no further changes.
4.  It's possible to miss rules that apply.  Double check your work by following steps 1-3 again.
5.  Prepare the Justification field.  This should include a justification for each rule of why it
    was or was not matched.
6.  Output the final result as JSON.
"#
                .to_string(),
                images: None,
                tool_calls: None,
            },
            ChatMessage {
                role: "system".to_string(),
                content: format!(
                    "Unless overridden, output {}",
                    serde_json::to_string(&report.default_return).unwrap()
                ),
                images: None,
                tool_calls: None,
            },
        ];
        req.messages.extend(report.messages());
        req.messages.push(ChatMessage {
            role: "system".to_string(),
            content: "Consider each of the above rules' criteria three times.
For each rule, consider whether the criteria applies in isolation and without consider of other rules.
" 
                .to_string(),
            images: None,
            tool_calls: None,
        });
        req.messages.push(ChatMessage {
            role: "user".to_string(),
            content: format!("UNSTRUCTURED DATA:\n{prompt}"),
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
            .with_semantic_injection(
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
                None,
                yammer::ChatRequest {
                    model: "phi4:14b-fp16".to_string(),
                    format: None,
                    keep_alive: None,
                    messages: vec![],
                    tools: None,
                    stream: None,
                    options: serde_json::json! {{
                        "num_ctx": 16_000,
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
