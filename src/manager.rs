use std::time::Instant;

use claudius::{
    push_or_merge_message, Anthropic, ContentBlock, MessageCreateParams, MessageParam,
    MessageParamContent, MessageRole, SystemPrompt, TextBlock, ToolChoice, ToolResultBlock,
};

use crate::{ApplyError, Policy, Report, ReportBuilder, Usage};

/// Manages a collection of policies and applies them to unstructured data.
///
/// The Manager ensures all policies have the same type and coordinates
/// their application to extract structured data from unstructured text.
///
/// # Example
///
/// ```ignore
/// use policyai::{Manager, Policy};
///
/// let mut manager = Manager::default();
/// manager.add(policy1);
/// manager.add(policy2);
///
/// let report = manager.apply(
///     &client,
///     template,
///     "unstructured text data"
/// ).await?;
/// ```
#[derive(Debug, Default)]
pub struct Manager {
    policies: Vec<Policy>,
}

impl Manager {
    /// Add a policy to the manager.
    ///
    /// # Panics
    ///
    /// Panics if the policy type doesn't match existing policies in the manager.
    pub fn add(&mut self, policy: Policy) {
        if let Some(last) = self.policies.last() {
            assert_eq!(last.r#type, policy.r#type);
        }
        self.policies.push(policy);
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
        usage: &mut Option<Usage>,
    ) -> Result<Report, ApplyError> {
        let start_time = Instant::now();
        let (report, mut req) = self.request_for(template, unstructured_data).await?;
        let max_attempts = 5;
        let mut last_error = String::new();

        // Initialize usage tracking if provided
        if let Some(ref mut u) = usage {
            *u = Usage::new();
        }

        for attempt in 1..=max_attempts {
            let resp = client.send(req.clone()).await?;

            // Track usage if provided
            if let Some(ref mut u) = usage {
                u.add_claudius_usage(resp.usage);
                u.increment_iterations();
            }
            if resp.content.len() != 1 {
                return Err(ApplyError::invalid_response(
                    format!(
                        "Expected exactly 1 content block, got {}",
                        resp.content.len()
                    ),
                    "Check that the LLM is configured correctly and the tool definition is valid",
                ));
            }
            let ContentBlock::ToolUse(t) = &resp.content[0] else {
                return Err(ApplyError::invalid_response(
                    "Expected ToolUse content block",
                    "The LLM should be using the output_json tool to provide structured output",
                ));
            };
            let ir = t.input.clone();
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
                if let Some(ref mut u) = usage {
                    u.set_wall_clock_time(start_time.elapsed());
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
                "Your output is inconsistent and I reject it with a request for you to try again."
                    .to_string();
            if !empirical_but_not_reported.is_empty() {
                content += "\n\nYou output the JSON corresponding to the following rules but did not report them in \"__rule_numbers__\":\n";
                for rule_number in empirical_but_not_reported.into_iter() {
                    if rule_number > 0 && rule_number <= report.masks_by_index.len() {
                        for mask in report.masks_by_index[rule_number - 1].iter() {
                            content += &format!(
                            "- Rule {rule_number}: Either set \"{mask}\" to its default or append {rule_number} to \"__rule_numbers__\".\n"
                        );
                        }
                    } else {
                        content += &format!("- Rule number {rule_number} doesn't exist.\n");
                    }
                }
            }
            if !reported_but_not_empirical.is_empty() {
                content += "\n\nYou reported the following rules but did not output their JSON:\n";
                for rule_number in reported_but_not_empirical.into_iter() {
                    if rule_number > 0 && rule_number <= report.masks_by_index.len() {
                        for mask in report.masks_by_index[rule_number - 1].iter() {
                            content += &format!(
                            "- Rule {rule_number}: Either set \"{mask}\" to a non-default value or remove {rule_number} from \"__rule_numbers__\".\n"
                        );
                        }
                    } else {
                        content += &format!("- Rule number {rule_number} doesn't exist.\n");
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
            push_or_merge_message(
                &mut req.messages,
                MessageParam {
                    role: MessageRole::User,
                    content: MessageParamContent::Array(vec![ContentBlock::ToolResult(
                        ToolResultBlock {
                            tool_use_id: t.id.clone(),
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
        // Set final wall clock time even on error
        if let Some(ref mut u) = usage {
            u.set_wall_clock_time(start_time.elapsed());
        }
        Err(ApplyError::too_many_iterations(max_attempts, last_error))
    }

    pub async fn request_for(
        &mut self,
        template: MessageCreateParams,
        text: &str,
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
        req.tool_choice = Some(ToolChoice::tool("output_json"));
        req.tools = Some(vec![claudius::ToolUnionParam::CustomTool(
            claudius::ToolParam {
                name: "output_json".to_string(),
                description: Some("output JSON".to_string()),
                input_schema: report.schema(),
                cache_control: None,
            },
        )]);
        Ok((report, req))
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
                    default: false,
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
    #[should_panic]
    fn manager_add_policy_different_type_panics() {
        let mut manager = Manager::default();

        let type1 = create_test_policy_type();
        let type2 = PolicyType {
            name: "DifferentPolicy".to_string(),
            fields: vec![Field::Bool {
                name: "enabled".to_string(),
                default: true,
                on_conflict: crate::OnConflict::Default,
            }],
        };

        let policy1 = create_test_policy(type1, "first", serde_json::json!({"is_active": true}));
        let policy2 = create_test_policy(type2, "second", serde_json::json!({"enabled": false}));

        manager.add(policy1);
        manager.add(policy2); // This should panic
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
        println!("Number of messages: {}", req.messages.len()); // Debug output
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
                        Some("number") => has_number = true,
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
