use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use claudius::{
    push_or_merge_message, Anthropic, ContentBlock, JsonSchema, KnownModel, MessageCreateParams,
    MessageParam, MessageRole, Metadata, Model, SystemPrompt, TextBlock, ToolChoice,
};

use policyai::data::{EvaluationReport, Metrics, TestDataPoint};
use policyai::{ApplyError, Field, Manager, Policy, Usage};

pub async fn naive_apply(
    client: &Anthropic,
    policies: &[Policy],
    template: &MessageCreateParams,
    text: &str,
    usage: &mut Option<Usage>,
) -> Result<serde_json::Value, ApplyError> {
    let mut req = template.clone();
    req.metadata = Some(Metadata {
        user_id: Some("baseline".into()),
    });
    req.system = Some(SystemPrompt::from_blocks(vec![TextBlock {
        text: include_str!("../../prompts/manager.md").to_string(),
        cache_control: None,
        citations: None,
    }]));
    let mut properties = serde_json::json! {{}};
    for policy in policies.iter() {
        let content = policy.prompt.clone();
        for field in policy.r#type.fields.iter() {
            match field {
                Field::Bool {
                    name,
                    default: _,
                    on_conflict: _,
                } => {
                    properties[name.clone()] = bool::json_schema();
                }
                Field::Number {
                    name,
                    default: _,
                    on_conflict: _,
                } => {
                    properties[name.clone()] = f64::json_schema();
                }
                Field::String {
                    name,
                    default: _,
                    on_conflict: _,
                } => {
                    properties[name.clone()] = String::json_schema();
                }
                Field::StringEnum {
                    name,
                    values,
                    default: _,
                    on_conflict: _,
                } => {
                    let mut schema = String::json_schema();
                    if let serde_json::Value::Object(object) = &mut schema {
                        object.insert("enum".to_string(), values.clone().into());
                    }
                    properties[name.clone()] = schema;
                }
                Field::StringArray { name } => {
                    properties[name.clone()] = Vec::<String>::json_schema();
                }
            }
        }
        push_or_merge_message(
            &mut req.messages,
            MessageParam {
                role: MessageRole::User,
                content: format!("<rule>{content}</rule>").into(),
            },
        );
    }
    push_or_merge_message(
        &mut req.messages,
        MessageParam::new_with_string(format!("<text>{text}</text>"), MessageRole::User),
    );
    let mut schema = serde_json::json! {{}};
    schema["type"] = "object".into();
    schema["required"] = serde_json::Value::Array(vec![]);
    schema["properties"] = properties;
    req.tool_choice = Some(ToolChoice::tool("output_json"));
    req.tools = Some(vec![claudius::ToolUnionParam::CustomTool(
        claudius::ToolParam {
            name: "output_json".to_string(),
            description: Some("output JSON according to policy".to_string()),
            input_schema: schema,
            cache_control: None,
        },
    )]);
    let start_time = Instant::now();
    let resp = client.send(req).await?;

    // Track usage if provided
    if let Some(ref mut u) = usage {
        *u = Usage::new();
        u.add_claudius_usage(resp.usage);
        u.increment_iterations();
        u.set_wall_clock_time(start_time.elapsed());
    }

    if resp.content.len() != 1 {
        todo!();
    }
    let ContentBlock::ToolUse(t) = &resp.content[0] else {
        todo!();
    };
    Ok(t.input.clone())
}

fn values_match(expected: &serde_json::Value, actual: &serde_json::Value) -> bool {
    // Direct equality check first
    if expected == actual {
        return true;
    }

    // Check if both are numbers and compare with tolerance
    match (expected, actual) {
        (serde_json::Value::Number(n1), serde_json::Value::Number(n2)) => {
            // Convert both to f64 and compare with tolerance
            let v1 = if let Some(f) = n1.as_f64() {
                f
            } else if let Some(i) = n1.as_i64() {
                i as f64
            } else if let Some(u) = n1.as_u64() {
                u as f64
            } else {
                return false;
            };

            let v2 = if let Some(f) = n2.as_f64() {
                f
            } else if let Some(i) = n2.as_i64() {
                i as f64
            } else if let Some(u) = n2.as_u64() {
                u as f64
            } else {
                return false;
            };

            // Check if within 0.001% tolerance
            if v1 == 0.0 && v2 == 0.0 {
                true
            } else if v1 == 0.0 || v2 == 0.0 {
                // One is zero, other is not
                false
            } else {
                let relative_diff = ((v1 - v2) / v1).abs();
                relative_diff <= 0.00001 // 0.001% = 0.00001
            }
        }
        _ => false,
    }
}

fn clean_baseline(baseline: &serde_json::Value) -> serde_json::Value {
    // Remove __rule_numbers__ field from baseline if it exists
    if let serde_json::Value::Object(mut obj) = baseline.clone() {
        obj.remove("__rule_numbers__");
        serde_json::Value::Object(obj)
    } else {
        baseline.clone()
    }
}

fn calculate_field_metrics(
    expected: &serde_json::Map<String, serde_json::Value>,
    actual: &serde_json::Value,
) -> (usize, usize, usize, usize) {
    let mut matched = 0;
    let mut wrong_value = 0;
    let mut missing = 0;
    let mut extra = 0;

    let actual_map = actual.as_object();

    for (k, expected_val) in expected {
        if let Some(actual_obj) = actual_map {
            if let Some(actual_val) = actual_obj.get(k) {
                if values_match(expected_val, actual_val) {
                    matched += 1;
                } else {
                    wrong_value += 1;
                }
            } else {
                missing += 1;
            }
        } else {
            missing += 1;
        }
    }

    // Count extra fields (ignoring __rule_numbers__)
    if let Some(actual_obj) = actual_map {
        for k in actual_obj.keys() {
            if k != "__rule_numbers__" && !expected.contains_key(k) {
                extra += 1;
            }
        }
    }

    (matched, wrong_value, missing, extra)
}

#[tokio::main]
async fn main() {
    let client = Anthropic::new(None).unwrap();
    for file in std::env::args().skip(1) {
        let file = OpenOptions::new()
            .read(true)
            .open(file)
            .expect("could not read input");
        let file = BufReader::new(file);
        for line in file.lines() {
            let line = line.expect("could not read data");
            let point: TestDataPoint = match serde_json::from_str(&line) {
                Ok(point) => point,
                Err(err) => {
                    eprintln!("error parsing policy {line}: {err}");
                    continue;
                }
            };
            let mut manager = Manager::default();
            for policy in point.policies.iter() {
                manager.add(policy.clone());
            }
            let expected = match &point.expected {
                Some(serde_json::Value::Object(obj)) => obj.clone(),
                _ => {
                    eprintln!("error parsing expected as object on line {line}");
                    continue;
                }
            };
            let mut metrics = Metrics::default();

            // Run baseline
            let mut baseline_usage = Some(Usage::new());
            let start = Instant::now();
            let baseline = match naive_apply(
                &client,
                &point.policies,
                &MessageCreateParams {
                    max_tokens: 2048,
                    ..Default::default()
                },
                &point.text,
                &mut baseline_usage,
            )
            .await
            {
                Ok(baseline) => Some(baseline),
                Err(err) => {
                    metrics.baseline_error = Some(format!("{err:?}"));
                    None
                }
            };
            metrics.baseline_apply_duration_ms = start.elapsed().as_millis() as u32;
            metrics.baseline_usage = baseline_usage;

            // Calculate baseline metrics if we have a result
            if let Some(ref baseline_val) = baseline {
                let cleaned_baseline = clean_baseline(baseline_val);
                let (matched, wrong, missing, extra) =
                    calculate_field_metrics(&expected, &cleaned_baseline);
                metrics.baseline_fields_matched = matched;
                metrics.baseline_fields_with_wrong_value = wrong;
                metrics.baseline_fields_missing = missing;
                metrics.baseline_extra_fields = extra;
            }
            // Run policyai
            let mut policyai_usage = Some(Usage::new());
            let start = Instant::now();
            let output = match manager
                .apply(
                    &client,
                    MessageCreateParams {
                        max_tokens: 2048,
                        model: Model::Known(KnownModel::ClaudeSonnet40),
                        ..Default::default()
                    },
                    &point.text,
                    &mut policyai_usage,
                )
                .await
            {
                Ok(returned) => returned.value().clone(),
                Err(err) => {
                    metrics.policyai_error = Some(format!("{err:?}"));
                    serde_json::json! {{}}
                }
            };
            metrics.policyai_apply_duration_ms = start.elapsed().as_millis() as u32;
            metrics.policyai_usage = policyai_usage;

            // Calculate policyai metrics if we have a result
            let (matched, wrong, missing, extra) = calculate_field_metrics(&expected, &output);
            metrics.policyai_fields_matched = matched;
            metrics.policyai_fields_with_wrong_value = wrong;
            metrics.policyai_fields_missing = missing;
            metrics.policyai_extra_fields = extra;

            // Create and output the report
            let report = EvaluationReport {
                input: point,
                metrics,
                output,
                baseline,
            };

            // Output JSON report to stdout
            println!("{}", serde_json::to_string(&report).unwrap());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluation_report_minimal() {
        let report = EvaluationReport {
            input: TestDataPoint {
                text: "test".to_string(),
                policies: vec![],
                expected: None,
                conflicts: None,
            },
            metrics: Metrics::default(),
            output: serde_json::Value::Null,
            baseline: None,
        };

        let serialized = serde_json::to_string(&report).unwrap();
        assert!(serialized.contains("input"));
        assert!(serialized.contains("metrics"));
        assert!(serialized.contains("output"));
        assert!(serialized.contains("baseline"));
    }

    #[test]
    fn metrics_default_all_zeros() {
        let metrics = Metrics::default();
        assert_eq!(metrics.policyai_fields_matched, 0);
        assert_eq!(metrics.policyai_fields_with_wrong_value, 0);
        assert_eq!(metrics.policyai_fields_missing, 0);
        assert_eq!(metrics.policyai_extra_fields, 0);
        assert_eq!(metrics.baseline_fields_matched, 0);
        assert_eq!(metrics.baseline_fields_with_wrong_value, 0);
        assert_eq!(metrics.baseline_fields_missing, 0);
        assert_eq!(metrics.baseline_extra_fields, 0);
        assert!(metrics.policyai_error.is_none());
        assert!(metrics.baseline_error.is_none());
        assert_eq!(metrics.policyai_apply_duration_ms, 0);
        assert_eq!(metrics.baseline_apply_duration_ms, 0);
        assert!(metrics.policyai_usage.is_none());
        assert!(metrics.baseline_usage.is_none());
    }

    #[test]
    fn metrics_with_values() {
        let metrics = Metrics {
            policyai_fields_matched: 3,
            policyai_fields_with_wrong_value: 1,
            policyai_fields_missing: 2,
            policyai_extra_fields: 1,
            baseline_fields_matched: 2,
            baseline_fields_with_wrong_value: 2,
            baseline_fields_missing: 3,
            baseline_extra_fields: 0,
            policyai_error: Some("error1".to_string()),
            baseline_error: Some("error2".to_string()),
            policyai_apply_duration_ms: 100,
            baseline_apply_duration_ms: 200,
            policyai_usage: None,
            baseline_usage: None,
        };

        assert_eq!(metrics.policyai_fields_matched, 3);
        assert_eq!(metrics.policyai_fields_with_wrong_value, 1);
        assert_eq!(metrics.policyai_fields_missing, 2);
        assert_eq!(metrics.policyai_extra_fields, 1);
        // TODO(claude): baseline_*
        // TODO(claude): policyai_error
        // TODO(claude): baseline_error
        assert_eq!(metrics.policyai_apply_duration_ms, 100);
        assert_eq!(metrics.baseline_apply_duration_ms, 200);
    }

    #[test]
    fn clean_baseline_removes_rule_numbers() {
        let baseline = serde_json::json!({
            "field1": "value1",
            "field2": 42,
            "__rule_numbers__": [1, 2, 3]
        });

        let cleaned = clean_baseline(&baseline);
        let cleaned_obj = cleaned.as_object().unwrap();

        assert!(cleaned_obj.contains_key("field1"));
        assert!(cleaned_obj.contains_key("field2"));
        assert!(!cleaned_obj.contains_key("__rule_numbers__"));
        assert_eq!(cleaned_obj.len(), 2);
    }

    #[test]
    fn clean_baseline_handles_missing_rule_numbers() {
        let baseline = serde_json::json!({
            "field1": "value1",
            "field2": 42
        });

        let cleaned = clean_baseline(&baseline);
        assert_eq!(cleaned, baseline);
    }

    #[test]
    fn clean_baseline_handles_non_object() {
        let baseline = serde_json::json!("not an object");
        let cleaned = clean_baseline(&baseline);
        assert_eq!(cleaned, baseline);
    }

    #[test]
    fn calculate_field_metrics_ignores_rule_numbers() {
        let expected = serde_json::json!({
            "field1": "value1",
            "field2": 42
        });
        let expected_map = expected.as_object().unwrap();

        let actual = serde_json::json!({
            "field1": "value1",
            "field2": 42,
            "__rule_numbers__": [1, 2]
        });

        let (matched, wrong, missing, extra) = calculate_field_metrics(expected_map, &actual);
        assert_eq!(matched, 2);
        assert_eq!(wrong, 0);
        assert_eq!(missing, 0);
        assert_eq!(extra, 0); // __rule_numbers__ should not count as extra
    }

    #[test]
    fn values_match_identical_numbers() {
        assert!(values_match(&serde_json::json!(42), &serde_json::json!(42)));
        assert!(values_match(
            &serde_json::json!(2.71),
            &serde_json::json!(2.71)
        ));
        assert!(values_match(&serde_json::json!(0), &serde_json::json!(0)));
        assert!(values_match(
            &serde_json::json!(0.0),
            &serde_json::json!(0.0)
        ));
    }

    #[test]
    fn values_match_zero_equivalence() {
        // 0.0 as float and 0 as u64 should match
        assert!(values_match(&serde_json::json!(0.0), &serde_json::json!(0)));
        assert!(values_match(&serde_json::json!(0), &serde_json::json!(0.0)));

        // Different representations of zero
        let zero_float = serde_json::Number::from_f64(0.0).unwrap();
        let zero_int = serde_json::json!(0);
        assert!(values_match(
            &serde_json::Value::Number(zero_float),
            &zero_int
        ));
    }

    #[test]
    fn values_match_with_tolerance() {
        // Within 0.001% tolerance
        assert!(values_match(
            &serde_json::json!(1000.0),
            &serde_json::json!(1000.009)
        ));
        assert!(values_match(
            &serde_json::json!(1000.0),
            &serde_json::json!(999.991)
        ));

        // Just outside 0.001% tolerance
        assert!(!values_match(
            &serde_json::json!(1000.0),
            &serde_json::json!(1000.011)
        ));
        assert!(!values_match(
            &serde_json::json!(1000.0),
            &serde_json::json!(999.989)
        ));
    }

    #[test]
    fn values_match_different_types() {
        assert!(!values_match(
            &serde_json::json!("42"),
            &serde_json::json!(42)
        ));
        assert!(!values_match(
            &serde_json::json!(true),
            &serde_json::json!(1)
        ));
        assert!(!values_match(
            &serde_json::json!(null),
            &serde_json::json!(0)
        ));
    }

    #[test]
    fn calculate_field_metrics_all_match() {
        let expected = serde_json::json!({
            "field1": "value1",
            "field2": 42,
            "field3": true
        });
        let expected_map = expected.as_object().unwrap();

        let actual = serde_json::json!({
            "field1": "value1",
            "field2": 42,
            "field3": true
        });

        let (matched, wrong, missing, extra) = calculate_field_metrics(expected_map, &actual);
        assert_eq!(matched, 3);
        assert_eq!(wrong, 0);
        assert_eq!(missing, 0);
        assert_eq!(extra, 0);
    }

    #[test]
    fn calculate_field_metrics_numeric_tolerance() {
        let expected = serde_json::json!({
            "count": 1000.0,
            "zero_float": 0.0,
            "value": 42
        });
        let expected_map = expected.as_object().unwrap();

        let actual = serde_json::json!({
            "count": 1000.009,  // Within tolerance
            "zero_float": 0,     // 0 as integer should match 0.0
            "value": 42.0        // 42.0 as float should match 42 as int
        });

        let (matched, wrong, missing, extra) = calculate_field_metrics(expected_map, &actual);
        assert_eq!(matched, 3);
        assert_eq!(wrong, 0);
        assert_eq!(missing, 0);
        assert_eq!(extra, 0);
    }

    #[test]
    fn calculate_field_metrics_with_wrong_values() {
        let expected = serde_json::json!({
            "field1": "value1",
            "field2": 42,
            "field3": true
        });
        let expected_map = expected.as_object().unwrap();

        let actual = serde_json::json!({
            "field1": "different",
            "field2": 99,
            "field3": true
        });

        let (matched, wrong, missing, extra) = calculate_field_metrics(expected_map, &actual);
        assert_eq!(matched, 1); // Only field3 matches
        assert_eq!(wrong, 2); // field1 and field2 are wrong
        assert_eq!(missing, 0);
        assert_eq!(extra, 0);
    }

    #[test]
    fn calculate_field_metrics_with_missing_fields() {
        let expected = serde_json::json!({
            "field1": "value1",
            "field2": 42,
            "field3": true
        });
        let expected_map = expected.as_object().unwrap();

        let actual = serde_json::json!({
            "field1": "value1"
        });

        let (matched, wrong, missing, extra) = calculate_field_metrics(expected_map, &actual);
        assert_eq!(matched, 1); // Only field1 matches
        assert_eq!(wrong, 0);
        assert_eq!(missing, 2); // field2 and field3 are missing
        assert_eq!(extra, 0);
    }

    #[test]
    fn calculate_field_metrics_with_extra_fields() {
        let expected = serde_json::json!({
            "field1": "value1"
        });
        let expected_map = expected.as_object().unwrap();

        let actual = serde_json::json!({
            "field1": "value1",
            "field2": 42,
            "field3": true
        });

        let (matched, wrong, missing, extra) = calculate_field_metrics(expected_map, &actual);
        assert_eq!(matched, 1); // field1 matches
        assert_eq!(wrong, 0);
        assert_eq!(missing, 0);
        assert_eq!(extra, 2); // field2 and field3 are extra
    }

    #[test]
    fn calculate_field_metrics_empty_expected() {
        let expected = serde_json::json!({});
        let expected_map = expected.as_object().unwrap();

        let actual = serde_json::json!({
            "field1": "value1"
        });

        let (matched, wrong, missing, extra) = calculate_field_metrics(expected_map, &actual);
        assert_eq!(matched, 0);
        assert_eq!(wrong, 0);
        assert_eq!(missing, 0);
        assert_eq!(extra, 1);
    }

    #[test]
    fn calculate_field_metrics_empty_actual() {
        let expected = serde_json::json!({
            "field1": "value1"
        });
        let expected_map = expected.as_object().unwrap();

        let actual = serde_json::json!({});

        let (matched, wrong, missing, extra) = calculate_field_metrics(expected_map, &actual);
        assert_eq!(matched, 0);
        assert_eq!(wrong, 0);
        assert_eq!(missing, 1);
        assert_eq!(extra, 0);
    }

    #[test]
    fn calculate_field_metrics_both_empty() {
        let expected = serde_json::json!({});
        let expected_map = expected.as_object().unwrap();

        let actual = serde_json::json!({});

        let (matched, wrong, missing, extra) = calculate_field_metrics(expected_map, &actual);
        assert_eq!(matched, 0);
        assert_eq!(wrong, 0);
        assert_eq!(missing, 0);
        assert_eq!(extra, 0);
    }

    #[test]
    fn calculate_field_metrics_actual_not_object() {
        let expected = serde_json::json!({
            "field1": "value1"
        });
        let expected_map = expected.as_object().unwrap();

        let actual = serde_json::json!("not an object");

        let (matched, wrong, missing, extra) = calculate_field_metrics(expected_map, &actual);
        assert_eq!(matched, 0);
        assert_eq!(wrong, 0);
        assert_eq!(missing, 1); // field1 is missing since actual is not an object
        assert_eq!(extra, 0);
    }

    #[test]
    fn evaluation_report_serialization() {
        use policyai::{Field, Policy, PolicyType};

        let policy_type = PolicyType {
            name: "TestPolicy".to_string(),
            fields: vec![Field::Bool {
                name: "enabled".to_string(),
                default: false,
                on_conflict: policyai::OnConflict::Default,
            }],
        };

        let report = EvaluationReport {
            input: TestDataPoint {
                text: "test text".to_string(),
                policies: vec![Policy {
                    r#type: policy_type,
                    prompt: "test".to_string(),
                    action: serde_json::json!({"enabled": true}),
                }],
                expected: Some(serde_json::json!({"enabled": true})),
                conflicts: None,
            },
            metrics: Metrics {
                policyai_fields_matched: 1,
                policyai_fields_with_wrong_value: 0,
                policyai_fields_missing: 0,
                policyai_extra_fields: 0,
                baseline_fields_matched: 1,
                baseline_fields_with_wrong_value: 0,
                baseline_fields_missing: 0,
                baseline_extra_fields: 0,
                policyai_error: None,
                baseline_error: None,
                policyai_apply_duration_ms: 50,
                baseline_apply_duration_ms: 100,
                policyai_usage: None,
                baseline_usage: None,
            },
            output: serde_json::json!({"enabled": true}),
            baseline: Some(serde_json::json!({"enabled": true})),
        };

        let serialized = serde_json::to_string(&report).unwrap();
        let deserialized: EvaluationReport = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            report.metrics.policyai_fields_matched,
            deserialized.metrics.policyai_fields_matched
        );
        assert_eq!(
            report.metrics.policyai_apply_duration_ms,
            deserialized.metrics.policyai_apply_duration_ms
        );
        assert_eq!(report.output, deserialized.output);
        assert_eq!(report.baseline, deserialized.baseline);
    }

    #[test]
    fn metrics_clone() {
        let original = Metrics {
            policyai_fields_matched: 5,
            policyai_fields_with_wrong_value: 2,
            policyai_fields_missing: 1,
            policyai_extra_fields: 3,
            baseline_fields_matched: 4,
            baseline_fields_with_wrong_value: 1,
            baseline_fields_missing: 2,
            baseline_extra_fields: 1,
            policyai_error: Some("error".to_string()),
            baseline_error: None,
            policyai_apply_duration_ms: 150,
            baseline_apply_duration_ms: 250,
            policyai_usage: None,
            baseline_usage: None,
        };

        let cloned = original.clone();
        assert_eq!(
            original.policyai_fields_matched,
            cloned.policyai_fields_matched
        );
        assert_eq!(
            original.policyai_fields_with_wrong_value,
            cloned.policyai_fields_with_wrong_value
        );
        assert_eq!(
            original.policyai_fields_missing,
            cloned.policyai_fields_missing
        );
        assert_eq!(original.policyai_extra_fields, cloned.policyai_extra_fields);
        assert_eq!(original.policyai_error, cloned.policyai_error);
        assert_eq!(
            original.policyai_apply_duration_ms,
            cloned.policyai_apply_duration_ms
        );
        assert!(original.policyai_usage.is_none());
        assert!(original.baseline_usage.is_none());
    }

    #[test]
    fn metrics_debug() {
        let metrics = Metrics {
            policyai_fields_matched: 1,
            policyai_fields_with_wrong_value: 2,
            policyai_fields_missing: 3,
            policyai_extra_fields: 4,
            baseline_fields_matched: 5,
            baseline_fields_with_wrong_value: 6,
            baseline_fields_missing: 7,
            baseline_extra_fields: 8,
            policyai_error: None,
            baseline_error: None,
            policyai_apply_duration_ms: 100,
            baseline_apply_duration_ms: 200,
            policyai_usage: None,
            baseline_usage: None,
        };

        let debug_str = format!("{metrics:?}");
        assert!(debug_str.contains("Metrics"));
        assert!(debug_str.contains("policyai_fields_matched"));
        assert!(debug_str.contains("policyai_apply_duration_ms"));
    }
}
