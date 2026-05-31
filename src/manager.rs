use std::time::Instant;

use claudius::{
    push_or_merge_message, Anthropic, ContentBlock, MessageCreateParams, MessageParam,
    MessageParamContent, MessageRole, OutputFormat, SystemPrompt, TextBlock, ToolChoice,
    ToolResultBlock,
};

use crate::{ApplyError, Policy, PolicyError, Report, ReportBuilder, Usage};

/// Selects how PolicyAI requests structured output from the model.
///
/// This can be configured on [`Manager`] before inference, or passed directly
/// to [`Manager::apply_with_inference_config`] or
/// [`Manager::request_for_with_inference_config`] for a single request.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InferenceConfig {
    /// Use the `output_json` tool without strict structured-output validation.
    #[default]
    ToolUse,
    /// Use the `output_json` tool with `strict: true`.
    StrictToolUse,
    /// Use [`OutputFormat::JsonSchema`] instead of tool use.
    OutputFormatJsonSchema,
}

impl InferenceConfig {
    fn uses_tools(self) -> bool {
        matches!(self, Self::ToolUse | Self::StrictToolUse)
    }
}

/// Manages a collection of policies and applies them to unstructured data.
///
/// The Manager ensures all policies have the same type and coordinates
/// their application to extract structured data from unstructured text.
///
/// # Example
///
/// ```no_run
/// use policyai::Manager;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// # use claudius::{Anthropic, MessageCreateParams, MessageParam, MessageRole};
/// # use policyai::{PolicyType, Policy, Usage, DEFAULT_MODEL};
///
/// let mut manager = Manager::default();
/// # let client = Anthropic::new(None)?;
/// # let policy_type = PolicyType::parse("type TestPolicy { active: bool = true }")?;
/// # let policy = Policy {
/// #     r#type: policy_type,
/// #     prompt: "Test policy".to_string(),
/// #     action: serde_json::json!({}),
/// # };
/// manager.add(policy);
///
/// # let template = MessageCreateParams {
/// #     max_tokens: 1024,
/// #     model: DEFAULT_MODEL,
/// #     messages: vec![],
/// #     ..Default::default()
/// # };
/// # let mut usage = Some(Usage::default());
/// let report = manager.apply(
///     &client,
///     template,
///     "unstructured text data",
///     usage.as_mut()
/// ).await?;
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Default)]
pub struct Manager {
    policies: Vec<Policy>,
    inference_config: InferenceConfig,
}

impl Manager {
    /// Configure how this manager requests structured model output.
    ///
    /// This builder-style method is useful when selecting the inference config
    /// before the final call to [`Manager::apply`].
    pub fn with_inference_config(mut self, inference_config: InferenceConfig) -> Self {
        self.inference_config = inference_config;
        self
    }

    /// Set how this manager requests structured model output.
    pub fn set_inference_config(&mut self, inference_config: InferenceConfig) {
        self.inference_config = inference_config;
    }

    /// Return the currently configured inference mode.
    pub fn inference_config(&self) -> InferenceConfig {
        self.inference_config
    }

    /// Add a policy to the manager.
    ///
    /// # Panics
    ///
    /// Panics if the policy type doesn't match existing policies in the manager.
    pub fn add(&mut self, policy: Policy) {
        self.try_add(policy)
            .expect("policy type doesn't match existing policies");
    }

    /// Add a policy to the manager.
    ///
    /// # Errors
    ///
    /// Returns [`PolicyError::PolicyTypeMismatch`] if the policy type doesn't
    /// match existing policies in the manager.
    #[allow(clippy::result_large_err)]
    pub fn try_add(&mut self, policy: Policy) -> Result<(), PolicyError> {
        if let Some(last) = self.policies.last() {
            if last.r#type != policy.r#type {
                return Err(PolicyError::PolicyTypeMismatch {
                    expected: Box::new(last.r#type.clone()),
                    actual: Box::new(policy.r#type),
                });
            }
        }
        self.policies.push(policy);
        Ok(())
    }

    /// Get the number of policies managed.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.policies.len()
    }

    /// Check if the manager has no policies.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.policies.is_empty()
    }

    /// Apply all managed policies to unstructured data.
    ///
    /// This method sends the unstructured data to an LLM along with all policies,
    /// and attempts to extract structured data according to the policy rules.
    /// It will retry up to 3 times if the LLM's output is inconsistent.
    ///
    /// # Arguments
    ///
    /// * `client` - The Anthropic client for LLM communication
    /// * `template` - Message parameters template for the LLM request
    /// * `unstructured_data` - The text to apply policies to
    /// * `usage` - Optional mutable reference to track usage metrics
    ///
    /// # Returns
    ///
    /// A `Report` containing the structured output, or an `ApplyError` if processing fails.
    pub async fn apply(
        &mut self,
        client: &Anthropic,
        template: MessageCreateParams,
        unstructured_data: &str,
        usage: Option<&mut Usage>,
    ) -> Result<Report, ApplyError> {
        self.apply_with_inference_config(
            client,
            template,
            unstructured_data,
            usage,
            self.inference_config,
        )
        .await
    }

    /// Apply all managed policies using a specific inference configuration.
    ///
    /// This is the per-call override for selecting between non-strict tool use,
    /// strict tool use, and JSON-schema [`OutputFormat`] structured output.
    pub async fn apply_with_inference_config(
        &mut self,
        client: &Anthropic,
        template: MessageCreateParams,
        unstructured_data: &str,
        mut usage: Option<&mut Usage>,
        inference_config: InferenceConfig,
    ) -> Result<Report, ApplyError> {
        let start_time = Instant::now();
        let (report, mut req) = self
            .request_for_with_inference_config(template, unstructured_data, inference_config)
            .await?;
        let max_attempts = 5;
        let mut last_error = String::new();

        // Initialize usage tracking if provided
        if let Some(usage) = &mut usage {
            **usage = Usage::new();
        }

        for attempt in 1..=max_attempts {
            let resp = client.send(req.clone()).await?;

            // Track usage if provided
            if let Some(usage) = &mut usage {
                usage.add_claudius_usage(resp.usage);
                usage.increment_iterations();
            }
            let extracted = extract_ir(&resp.content, inference_config)?;
            let ir = extracted.ir;
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
                // Set final wall clock time
                if let Some(usage) = &mut usage {
                    usage.set_wall_clock_time(start_time.elapsed());
                }
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
            let mut content =
                "<instruction>The reported rule numbers do not match the fields that were output.  Re-evaluate your output to resolve the following inconsistencies.</instruction>"
                    .to_string();
            if !empirical_but_not_reported.is_empty() {
                for rule_number in empirical_but_not_reported.into_iter() {
                    if rule_number > 0 && rule_number <= report.masks_by_index.len() {
                        for mask in report.masks_by_index[rule_number - 1].iter() {
                            content += &format!("<inconsistency>{rule_number} was not present in rule numbers, but \"{mask}\" was set.<resolution>Unset \"{mask}\" if the context doesn't match or add {rule_number} to \"__rule_numbers__\" if the rule matches.</resolution></inconsistency>");
                        }
                    } else {
                        content += &format!("<inconsistency>Rule number {rule_number} present in __rule_numbers__, but it doesn't exist in the reported rules.</inconsistency>");
                    }
                }
            }
            if !reported_but_not_empirical.is_empty() {
                content += "\n\nYou reported the following rules but did not output their JSON:\n";
                for rule_number in reported_but_not_empirical.into_iter() {
                    if rule_number > 0 && rule_number <= report.masks_by_index.len() {
                        for mask in report.masks_by_index[rule_number - 1].iter() {
                            content += &format!("<inconsistency>{rule_number} was present in rule numbers, but \"{mask}\" was not set.<resolution>Set \"{mask}\" if the context matches or remove {rule_number} from \"__rule_numbers__\" if the rule does not match.</resolution></inconsistency>");
                        }
                    } else {
                        content += &format!("<inconsistency>Rule number {rule_number} present in __rule_numbers__, but it doesn't exist in the reported rules.</inconsistency>");
                    }
                }
            }
            last_error = format!("Attempt {attempt}/{max_attempts}: Rule mismatch - empirically matched {empirically_matched:?} but reportedly matched {reportedly_matched:?}");
            push_or_merge_message(
                &mut req.messages,
                MessageParam {
                    role: MessageRole::Assistant,
                    content: MessageParamContent::Array(resp.content.clone()),
                },
            );
            match extracted.feedback {
                FeedbackTarget::Tool { tool_use_id } => {
                    push_or_merge_message(
                        &mut req.messages,
                        MessageParam {
                            role: MessageRole::User,
                            content: MessageParamContent::Array(vec![ContentBlock::ToolResult(
                                ToolResultBlock {
                                    tool_use_id,
                                    cache_control: None,
                                    is_error: Some(true),
                                    content: Some(
                                        format!("<error-message>{content}</error-message>").into(),
                                    ),
                                },
                            )]),
                        },
                    );
                }
                FeedbackTarget::Text => {
                    push_or_merge_message(
                        &mut req.messages,
                        MessageParam::new_with_string(
                            format!("<error-message>{content}</error-message>"),
                            MessageRole::User,
                        ),
                    );
                }
            }
        }
        // Set final wall clock time even on error
        if let Some(usage) = &mut usage {
            usage.set_wall_clock_time(start_time.elapsed());
        }
        Err(ApplyError::too_many_iterations(max_attempts, last_error))
    }

    /// Prepare a request for LLM processing by building the necessary context.
    ///
    /// This method constructs the complete request that will be sent to the LLM,
    /// including system prompts, policy rules, and the input text. It returns
    /// both a ReportBuilder for processing the response and the configured request.
    ///
    /// # Arguments
    ///
    /// * `template` - Base message parameters to use for the LLM request
    /// * `text` - The unstructured text data to analyze
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - `ReportBuilder`: Used to process the LLM's structured response
    /// - `MessageCreateParams`: The complete request ready to send to the LLM
    ///
    /// # Errors
    ///
    /// Returns `ApplyError` if policy addition to the report builder fails.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use policyai::Manager;
    /// # use claudius::MessageCreateParams;
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut manager = Manager::default();
    /// let template = MessageCreateParams::default();
    /// let (report_builder, request) = manager.request_for(template, "analyze this text").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn request_for(
        &mut self,
        template: MessageCreateParams,
        text: &str,
    ) -> Result<(ReportBuilder, MessageCreateParams), ApplyError> {
        self.request_for_with_inference_config(template, text, self.inference_config)
            .await
    }

    /// Prepare a request using a specific inference configuration.
    ///
    /// This is the per-call request-builder override for selecting between
    /// non-strict tool use, strict tool use, and JSON-schema [`OutputFormat`]
    /// structured output.
    pub async fn request_for_with_inference_config(
        &mut self,
        template: MessageCreateParams,
        text: &str,
        inference_config: InferenceConfig,
    ) -> Result<(ReportBuilder, MessageCreateParams), ApplyError> {
        let mut report = ReportBuilder::default();
        for policy in self.policies.iter() {
            report.add_policy(policy)?;
        }
        let mut req = template;
        req.system = Some(SystemPrompt::from_blocks(vec![TextBlock {
            text: include_str!("../prompts/manager.md").to_string(),
            cache_control: None,
            citations: None,
        }]));

        push_or_merge_message(
            &mut req.messages,
            MessageParam::new_with_string(
                format!(
                    "<default>Unless specified otherwise, output {}</default>",
                    serde_json::to_string(report.default_return()).unwrap()
                ),
                MessageRole::User,
            ),
        );
        for message in report.messages() {
            push_or_merge_message(&mut req.messages, message)
        }
        push_or_merge_message(
            &mut req.messages,
            MessageParam::new_with_string(format!("<text>{text}</text>"), MessageRole::User),
        );
        push_or_merge_message(
            &mut req.messages,
            MessageParam::new_with_string(
                include_str!("../prompts/manager_suffix.md").to_string(),
                MessageRole::User,
            ),
        );
        configure_structured_output(&mut req, &report, inference_config);
        Ok((report, req))
    }
}

struct ExtractedIr {
    ir: serde_json::Value,
    feedback: FeedbackTarget,
}

enum FeedbackTarget {
    Tool { tool_use_id: String },
    Text,
}

#[allow(clippy::result_large_err)]
fn extract_ir(
    content: &[ContentBlock],
    inference_config: InferenceConfig,
) -> Result<ExtractedIr, ApplyError> {
    if content.len() != 1 {
        return Err(ApplyError::invalid_response(
            format!("Expected exactly 1 content block, got {}", content.len()),
            "Check that the LLM is configured correctly for the selected inference config",
        ));
    }
    match inference_config {
        InferenceConfig::ToolUse | InferenceConfig::StrictToolUse => {
            let ContentBlock::ToolUse(t) = &content[0] else {
                return Err(ApplyError::invalid_response(
                    "Expected ToolUse content block",
                    "The LLM should be using the output_json tool to provide structured output",
                ));
            };
            Ok(ExtractedIr {
                ir: t.input.clone(),
                feedback: FeedbackTarget::Tool {
                    tool_use_id: t.id.clone(),
                },
            })
        }
        InferenceConfig::OutputFormatJsonSchema => {
            let ContentBlock::Text(t) = &content[0] else {
                return Err(ApplyError::invalid_response(
                    "Expected Text content block",
                    "The LLM should return JSON text when OutputFormatJsonSchema is selected",
                ));
            };
            let ir = serde_json::from_str(t.text.trim()).map_err(|err| {
                ApplyError::invalid_response(
                    format!("Could not parse JSON response: {err}"),
                    "Check that the model response is valid JSON for the configured schema",
                )
            })?;
            Ok(ExtractedIr {
                ir,
                feedback: FeedbackTarget::Text,
            })
        }
    }
}

fn configure_structured_output(
    req: &mut MessageCreateParams,
    report: &ReportBuilder,
    inference_config: InferenceConfig,
) {
    clear_output_format_config(req);
    if inference_config.uses_tools() {
        req.tool_choice = Some(ToolChoice::tool("output_json"));
        req.tools = Some(vec![claudius::ToolUnionParam::CustomTool(
            claudius::ToolParam {
                name: "output_json".to_string(),
                description: Some("output JSON".to_string()),
                input_schema: report.schema(),
                cache_control: None,
                strict: (inference_config == InferenceConfig::StrictToolUse).then_some(true),
            },
        )]);
    } else {
        req.tool_choice = None;
        req.tools = None;
        req.output_format = Some(OutputFormat::json_schema(report.schema()));
    }
}

fn clear_output_format_config(req: &mut MessageCreateParams) {
    req.output_format = None;
    if let Some(output_config) = &mut req.output_config {
        output_config.format = None;
        if output_config.effort.is_none() {
            req.output_config = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Field, PolicyType};
    use claudius::SystemPrompt;

    fn create_test_policy_type() -> PolicyType {
        PolicyType {
            name: "TestPolicy".to_string(),
            fields: vec![
                Field::Bool {
                    name: "is_active".to_string(),
                    default: Some(false),
                    on_conflict: crate::OnConflict::Default,
                },
                Field::String {
                    name: "message".to_string(),
                    default: Some("default".to_string()),
                    on_conflict: crate::OnConflict::Agreement,
                },
                Field::Number {
                    name: "count".to_string(),
                    default: Some(crate::t64(0.0)),
                    on_conflict: crate::OnConflict::LargestValue,
                },
            ],
        }
    }

    fn create_test_policy(r#type: PolicyType, prompt: &str, action: serde_json::Value) -> Policy {
        Policy {
            r#type,
            prompt: prompt.to_string(),
            action,
        }
    }

    #[test]
    fn manager_default_is_empty() {
        let manager = Manager::default();
        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
        assert_eq!(manager.inference_config(), InferenceConfig::ToolUse);
    }

    #[test]
    fn manager_inference_config_can_be_set_before_apply() {
        let mut manager = Manager::default().with_inference_config(InferenceConfig::StrictToolUse);
        assert_eq!(manager.inference_config(), InferenceConfig::StrictToolUse);

        manager.set_inference_config(InferenceConfig::OutputFormatJsonSchema);
        assert_eq!(
            manager.inference_config(),
            InferenceConfig::OutputFormatJsonSchema
        );
    }

    #[test]
    fn manager_add_single_policy() {
        let mut manager = Manager::default();
        let policy_type = create_test_policy_type();
        let policy = create_test_policy(
            policy_type,
            "test prompt",
            serde_json::json!({"is_active": true}),
        );

        manager.add(policy);
        assert!(!manager.is_empty());
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn manager_add_multiple_policies_same_type() {
        let mut manager = Manager::default();
        let policy_type = create_test_policy_type();

        let policy1 = create_test_policy(
            policy_type.clone(),
            "first prompt",
            serde_json::json!({"is_active": true}),
        );
        let policy2 = create_test_policy(
            policy_type.clone(),
            "second prompt",
            serde_json::json!({"message": "hello"}),
        );
        let policy3 = create_test_policy(
            policy_type,
            "third prompt",
            serde_json::json!({"count": 42}),
        );

        manager.add(policy1);
        manager.add(policy2);
        manager.add(policy3);

        assert_eq!(manager.len(), 3);
    }

    #[test]
    fn manager_try_add_policy_different_type_returns_error() {
        let mut manager = Manager::default();

        let type1 = create_test_policy_type();
        let type2 = PolicyType {
            name: "DifferentPolicy".to_string(),
            fields: vec![Field::Bool {
                name: "enabled".to_string(),
                default: Some(true),
                on_conflict: crate::OnConflict::Default,
            }],
        };

        let policy1 = create_test_policy(type1, "first", serde_json::json!({"is_active": true}));
        let policy2 = create_test_policy(type2, "second", serde_json::json!({"enabled": false}));

        manager.add(policy1);
        let err = manager.try_add(policy2).unwrap_err();
        assert!(matches!(err, crate::PolicyError::PolicyTypeMismatch { .. }));
    }

    #[tokio::test]
    async fn manager_request_for_empty_manager() {
        let mut manager = Manager::default();
        let template = MessageCreateParams::default();
        let text = "test text";

        let result = manager.request_for(template, text).await;
        assert!(result.is_ok());

        let (_report, req) = result.unwrap();
        assert!(!req.messages.is_empty());
        assert!(req.system.is_some());
        assert_eq!(req.tool_choice, Some(ToolChoice::tool("output_json")));
        assert!(!req.requires_structured_outputs_beta());
    }

    #[tokio::test]
    async fn manager_request_for_strict_tool_use() {
        let mut manager = Manager::default();
        let template = MessageCreateParams::default();

        let result = manager
            .request_for_with_inference_config(
                template,
                "test text",
                InferenceConfig::StrictToolUse,
            )
            .await;
        assert!(result.is_ok());

        let (_, req) = result.unwrap();
        assert_eq!(req.tool_choice, Some(ToolChoice::tool("output_json")));
        assert!(req.output_format.is_none());
        assert!(req.requires_structured_outputs_beta());

        let tools = req.tools.as_ref().expect("expected tools");
        assert_eq!(tools.len(), 1);
        match &tools[0] {
            claudius::ToolUnionParam::CustomTool(tool) => {
                assert_eq!(tool.name, "output_json");
                assert_eq!(tool.strict, Some(true));
            }
            _ => panic!("expected custom tool"),
        }
    }

    #[tokio::test]
    async fn manager_request_for_output_format_json_schema() {
        let mut manager = Manager::default();
        let template = MessageCreateParams::default();

        let result = manager
            .request_for_with_inference_config(
                template,
                "test text",
                InferenceConfig::OutputFormatJsonSchema,
            )
            .await;
        assert!(result.is_ok());

        let (_, req) = result.unwrap();
        assert!(req.tool_choice.is_none());
        assert!(req.tools.is_none());
        assert!(req.requires_structured_outputs_beta());

        match req.output_format.expect("expected output format") {
            OutputFormat::JsonSchema { schema } => {
                assert_eq!(schema["type"], "object");
                assert!(schema["properties"].as_object().is_some());
            }
        }
    }

    #[tokio::test]
    async fn manager_request_for_with_policies() {
        let mut manager = Manager::default();
        let policy_type = create_test_policy_type();

        let policy1 = create_test_policy(
            policy_type.clone(),
            "if urgent then",
            serde_json::json!({"is_active": true, "count": 10}),
        );
        let policy2 = create_test_policy(
            policy_type,
            "if contains hello then",
            serde_json::json!({"message": "greeting"}),
        );

        manager.add(policy1);
        manager.add(policy2);

        let template = MessageCreateParams::default();
        let text = "urgent hello world";

        let result = manager.request_for(template, text).await;
        assert!(result.is_ok());

        let (report, req) = result.unwrap();
        assert!(!req.messages.is_empty()); // At least one message
        assert!(req.system.is_some());
        assert!(req.tools.is_some());

        // Verify the schema includes masked fields and special fields
        let schema = report.schema();
        assert!(schema["properties"].as_object().is_some());
        let properties = schema["properties"].as_object().unwrap();

        // Should have __rule_numbers__ special fields
        assert!(properties.contains_key("__rule_numbers__"));

        // Should have 3 masked fields (is_active, message, count)
        // The masked fields will have obfuscated names but correct types
        let masked_fields = properties.keys().filter(|k| !k.starts_with("__")).count();
        assert_eq!(masked_fields, 3, "Expected 3 masked fields");

        // Verify the types of the masked fields
        let mut has_boolean = false;
        let mut has_string = false;
        let mut has_number = false;

        for (key, value) in properties.iter() {
            if !key.starts_with("__") {
                if let Some(type_val) = value.get("type") {
                    match type_val.as_str() {
                        Some("boolean") => has_boolean = true,
                        Some("string") => has_string = true,
                        Some("number") | Some("integer") => has_number = true,
                        _ => {}
                    }
                }
            }
        }

        assert!(has_boolean, "Should have a boolean field (is_active)");
        assert!(has_string, "Should have a string field (message)");
        assert!(has_number, "Should have a number field (count)");
    }

    #[tokio::test]
    async fn manager_request_for_system_prompt() {
        let mut manager = Manager::default();
        let template = MessageCreateParams::default();
        let text = "test";

        let result = manager.request_for(template, text).await;
        assert!(result.is_ok());

        let (_, req) = result.unwrap();
        let system = req.system.unwrap();
        let system_str = match system {
            SystemPrompt::String(s) => s,
            SystemPrompt::Blocks(blocks) => {
                // Extract text from the first SystemTextBlock
                if let Some(text_block) = blocks.first() {
                    text_block.block.text.clone()
                } else {
                    panic!("Expected text block in system prompt")
                }
            }
        };

        // Verify key parts of the system prompt
        assert!(system_str.contains("Output JSON"));
        assert!(system_str.contains("if and only if a rule matches"));
    }

    #[test]
    fn manager_debug_format() {
        let manager = Manager::default();
        let debug_str = format!("{manager:?}");
        assert!(debug_str.contains("Manager"));
        assert!(debug_str.contains("policies"));
    }

    #[tokio::test]
    async fn manager_request_includes_text_message() {
        let mut manager = Manager::default();
        let template = MessageCreateParams::default();
        let test_text = "This is my special test text";

        let result = manager.request_for(template, test_text).await;
        assert!(result.is_ok());

        let (_, req) = result.unwrap();

        // Find the message containing our text
        let mut found_text = false;
        for message in &req.messages {
            if let MessageParamContent::String(content) = &message.content {
                if content.contains(test_text) {
                    found_text = true;
                    break;
                }
            }
        }
        assert!(found_text, "Request should include the input text");
    }
}
