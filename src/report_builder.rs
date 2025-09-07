use claudius::{push_or_merge_message, JsonSchema, MessageParam, MessageRole};

use crate::{
    ApplyError, BoolMask, Field, MaskGenerator, NumberMask, Policy, PolicyError, Report,
    StringArrayMask, StringEnumMask, StringMask,
};

/// Builder for constructing Reports from policy definitions.
///
/// A ReportBuilder accumulates policy configurations and creates the necessary
/// masks and infrastructure for applying those policies to unstructured data.
/// It handles field obfuscation, schema generation, and intermediate representation
/// processing.
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
    /// Add a policy to this report builder.
    ///
    /// Processes the policy definition and creates the necessary masks for each field
    /// that has a value specified in the policy's action. Handles field name obfuscation
    /// for secure communication with LLMs.
    ///
    /// # Arguments
    ///
    /// * `policy` - The policy to add to this builder
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the policy was successfully added, or a `PolicyError`
    /// if there were issues with the policy definition or field values.
    ///
    /// # Errors
    ///
    /// Returns `PolicyError` if:
    /// - Field values don't match their expected types
    /// - Enum values are not in the allowed set
    /// - Array fields contain non-string values
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::{ReportBuilder, Policy, PolicyType};
    /// let mut builder = ReportBuilder::default();
    /// # let policy_type = PolicyType::parse("type Test { active: bool = true }").unwrap();
    /// # let policy = Policy {
    /// #     r#type: policy_type,
    /// #     prompt: "test".to_string(),
    /// #     action: serde_json::json!({"active": true}),
    /// # };
    /// builder.add_policy(&policy)?;
    /// # Ok::<(), policyai::PolicyError>(())
    /// ```
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

    /// Convert intermediate representation into a final Report.
    ///
    /// Takes the JSON output from an LLM and applies all configured masks to extract
    /// structured data according to the policies that were added to this builder.
    ///
    /// # Arguments
    ///
    /// * `ir` - The intermediate representation JSON from the LLM
    ///
    /// # Returns
    ///
    /// Returns a `Report` containing the extracted structured data, or an `ApplyError`
    /// if there were issues processing the intermediate representation.
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::ReportBuilder;
    /// let builder = ReportBuilder::default();
    /// let ir = serde_json::json!({"field_abc": true});
    /// let report = builder.consume_ir(ir)?;
    /// # Ok::<(), policyai::ApplyError>(())
    /// ```
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

    /// Get the default return value structure.
    ///
    /// Returns the JSON object that represents the default values for all fields,
    /// which is used when the LLM doesn't provide specific values.
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::ReportBuilder;
    /// let builder = ReportBuilder::default();
    /// let defaults = builder.default_return();
    /// assert!(defaults.is_object());
    /// ```
    pub fn default_return(&self) -> &serde_json::Value {
        &self.default_return
    }

    /// Get the messages that should be included in LLM requests.
    ///
    /// Returns a vector of message parameters containing the formatted policy
    /// rules that will be sent to the LLM as part of the conversation.
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::ReportBuilder;
    /// let builder = ReportBuilder::default();
    /// let messages = builder.messages();
    /// // Messages will be empty for a default builder with no policies
    /// ```
    pub fn messages(&self) -> Vec<MessageParam> {
        self.messages.clone()
    }

    /// Get the JSON schema for the expected LLM output.
    ///
    /// Returns a JSON schema object that describes the structure and types
    /// that the LLM should use when providing its response.
    ///
    /// # Example
    ///
    /// ```
    /// # use policyai::ReportBuilder;
    /// let builder = ReportBuilder::default();
    /// let schema = builder.schema();
    /// assert_eq!(schema["type"], "object");
    /// assert!(schema["properties"].is_object());
    /// ```
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
