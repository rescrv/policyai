use claudius::{push_or_merge_message, JsonSchema, MessageParam, MessageRole};

use crate::{
    ApplyError, BoolMask, Field, MaskGenerator, NumberMask, Policy, PolicyError, Report,
    StringArrayMask, StringEnumMask, StringMask,
};

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
    messages: Vec<MessageParam>,
    policy_index: usize,
    required: Vec<String>,
    properties: serde_json::Value,
}

impl ReportBuilder {
    pub fn add_policy(&mut self, policy: &Policy) -> Result<(), PolicyError> {
        // Assume default=0, so we increment mask_index here (in case we throw out parts of it) and
        // increment policy_index at the end when we "commit".
        self.mask_index += 1;
        let mut content = format!("Rule #{}:\nCriteria: {}", self.policy_index, policy.prompt);

        // Collect all changes first before applying them
        let mut new_bool_masks = Vec::new();
        let mut new_number_masks = Vec::new();
        let mut new_string_masks = Vec::new();
        let mut new_string_array_masks = Vec::new();
        let mut new_string_enum_masks = Vec::new();
        let mut new_required = Vec::new();
        let mut new_properties = serde_json::Map::new();
        let mut new_masks = Vec::new();
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
                        return Err(PolicyError::expected_bool(name.clone(), value));
                    };
                    let mask = self.mask_gen.generate();
                    new_masks.push(mask.clone());
                    new_bool_masks.push(BoolMask::new(
                        self.policy_index,
                        name.clone(),
                        mask.clone(),
                        *default,
                        *v,
                        *on_conflict,
                    ));
                    self.default_return[&mask] = (!*v).into();
                    content = content.replace(&format!("{name:?}"), &format!("{mask:?}"));
                    new_required.push(mask.clone());
                    new_properties.insert(mask, bool::json_schema());
                }
                Field::Number {
                    name,
                    default,
                    on_conflict,
                } => {
                    let serde_json::Value::Number(v) = value else {
                        return Err(PolicyError::expected_number(name.clone(), value));
                    };
                    let mask = self.mask_gen.generate();
                    new_masks.push(mask.clone());
                    new_number_masks.push(NumberMask::new(
                        self.policy_index,
                        name.clone(),
                        mask.clone(),
                        *default,
                        v.clone(),
                        *on_conflict,
                    ));
                    self.default_return[&mask] = serde_json::Value::Number(v.clone());
                    content = content.replace(&format!("{name:?}"), &format!("{mask:?}"));
                    if default.is_some() {
                        new_required.push(mask.clone());
                    }
                    new_properties.insert(mask, f64::json_schema());
                }
                Field::String {
                    name,
                    default,
                    on_conflict,
                } => {
                    let serde_json::Value::String(v) = value else {
                        return Err(PolicyError::expected_string(name.clone(), value));
                    };
                    let mask = self.mask_gen.generate();
                    new_masks.push(mask.clone());
                    new_string_masks.push(StringMask::new(
                        self.policy_index,
                        name.clone(),
                        mask.clone(),
                        default.clone(),
                        v.clone(),
                        *on_conflict,
                    ));
                    self.default_return[&mask] = serde_json::Value::String(v.clone());
                    content = content.replace(&format!("{name:?}"), &format!("{mask:?}"));
                    if default.is_some() {
                        new_required.push(mask.clone());
                    }
                    new_properties.insert(mask, String::json_schema());
                }
                Field::StringArray { name } => {
                    let serde_json::Value::Array(v) = value else {
                        return Err(PolicyError::expected_string(name.clone(), value));
                    };
                    let mut strings = vec![];
                    for v in v {
                        if let serde_json::Value::String(v) = v {
                            strings.push(v.clone());
                        } else {
                            return Err(PolicyError::expected_string(name.clone(), v));
                        }
                    }
                    let mask = self.mask_gen.generate();
                    new_masks.push(mask.clone());
                    new_string_array_masks.push(StringArrayMask::new(
                        self.policy_index,
                        name.clone(),
                        mask.clone(),
                        strings,
                    ));
                    self.default_return[&mask] = serde_json::Value::Array(vec![]);
                    content = content.replace(&format!("{name:?}"), &format!("{mask:?}"));
                    new_properties.insert(mask, Vec::<String>::json_schema());
                }
                Field::StringEnum {
                    name,
                    values,
                    default,
                    on_conflict,
                } => {
                    let Some(v) = values.iter().find(|x| *x == value) else {
                        return Err(PolicyError::expected_string(name.clone(), value));
                    };
                    let mask = self.mask_gen.generate();
                    new_masks.push(mask.clone());
                    new_string_enum_masks.push(StringEnumMask::new(
                        self.policy_index,
                        name.clone(),
                        mask.clone(),
                        v.clone(),
                        default.clone(),
                        *on_conflict,
                    ));
                    self.default_return[&mask] = false.into();
                    content = content.replace(&format!("{name:?}"), &format!("{mask:?}"));
                    content = content.replace(&format!("{v:?}"), "true");
                    if default.is_some() {
                        new_required.push(mask.clone());
                    }
                    new_properties.insert(mask, bool::json_schema());
                }
            }
        }
        // Commit all changes atomically
        push_or_merge_message(
            &mut self.messages,
            MessageParam {
                role: MessageRole::User,
                content: format!("<rule>{content}</rule>").into(),
            },
        );

        // Extend collections instead of replacing
        self.required.extend(new_required);
        if let serde_json::Value::Object(props) = &mut self.properties {
            props.extend(new_properties);
        }
        self.bool_masks.extend(new_bool_masks);
        self.number_masks.extend(new_number_masks);
        self.string_masks.extend(new_string_masks);
        self.string_array_masks.extend(new_string_array_masks);
        self.string_enum_masks.extend(new_string_enum_masks);
        self.masks_by_index.push(new_masks);

        self.policy_index += 1;
        Ok(())
    }

    pub fn consume_ir(self, ir: serde_json::Value) -> Result<Report, ApplyError> {
        let mut report = Report::new(
            self.messages,
            self.bool_masks,
            self.number_masks,
            self.string_masks,
            self.string_array_masks,
            self.string_enum_masks,
            self.masks_by_index,
        );
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
            m.apply_to(&ir, &mut report);
        }
        for m in report.string_enum_masks.clone().into_iter() {
            m.apply_to(&ir, &mut report);
        }
        Ok(report)
    }

    pub fn default_return(&self) -> &serde_json::Value {
        &self.default_return
    }

    pub fn messages(&self) -> Vec<MessageParam> {
        self.messages.clone()
    }

    pub fn schema(&self) -> serde_json::Value {
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
            required: vec!["__rule_numbers__".to_string()],
            properties: serde_json::json! {{
                "__rule_numbers__": Vec::<f64>::json_schema(),
            }},
        }
    }
}
