use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use yammer::{ollama_host, ChatMessage, JsonSchema};

use policyai::data::TestDataPoint;
use policyai::{ApplyError, Field, Manager, Policy};

pub async fn naive_apply(
    policies: &[Policy],
    prompt: &str,
) -> Result<serde_json::Value, ApplyError> {
    let mut messages = vec![ChatMessage {
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
            Output the associated output.  Add the output to the __matched_rules__.
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
    }];
    let mut required = vec![];
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
                    required.push(name.clone());
                    properties[name.clone()] = bool::json_schema();
                }
                Field::Number {
                    name,
                    default: _,
                    on_conflict: _,
                } => {
                    required.push(name.clone());
                    properties[name.clone()] = f64::json_schema();
                }
                Field::String {
                    name,
                    default: _,
                    on_conflict: _,
                } => {
                    required.push(name.clone());
                    properties[name.clone()] = String::json_schema();
                }
                Field::StringEnum {
                    name,
                    values,
                    default: _,
                    on_conflict: _,
                } => {
                    required.push(name.clone());
                    let mut schema = String::json_schema();
                    if let serde_json::Value::Object(object) = &mut schema {
                        object.insert("enum".to_string(), values.clone().into());
                    }
                    properties[name.clone()] = schema;
                }
                Field::StringArray { name } => {
                    required.push(name.clone());
                    properties[name.clone()] = Vec::<String>::json_schema();
                }
            }
        }
        messages.push(ChatMessage {
            role: "system".to_string(),
            content,
            images: None,
            tool_calls: None,
        });
    }
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: prompt.to_string(),
        images: None,
        tool_calls: None,
    });
    let mut schema = serde_json::json! {{}};
    schema["type"] = "object".into();
    schema["required"] = required.into();
    schema["properties"] = properties;
    let req = yammer::ChatRequest {
        model: "qwq:32b-q8_0".to_string(),
        messages,
        format: Some(schema),
        stream: Some(false),
        keep_alive: None,
        options: serde_json::json! {{
            "num_ctx": 16_000,
        }},
        tools: None,
    };
    let resp = req
        .make_request(&ollama_host(None))
        .send()
        .await?
        .error_for_status()?
        .json::<yammer::ChatResponse>()
        .await?;
    Ok(serde_json::from_str(&resp.message.content)?)
}

#[tokio::main]
async fn main() {
    let mut baseline_success = 0u64;
    let mut baseline_fail = 0u64;
    let mut success = 0u64;
    let mut fail = 0u64;
    let mut conflict = 0u64;
    let mut baseline_miss = 0u64;
    let mut baseline_unequal = 0u64;
    let mut experimental_miss = 0u64;
    let mut experimental_unequal = 0u64;
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
                    eprintln!("error parsing policy {}: {}", line, err);
                    continue;
                }
            };
            let mut manager = Manager::default();
            for policy in point.policies.iter() {
                manager.add(policy.clone());
            }
            let baseline = match naive_apply(&point.policies, &point.text).await {
                Ok(baseline) => baseline,
                Err(_) => {
                    baseline_fail += 1;
                    serde_json::json! {{}}
                }
            };
            let expected = &point.expected;
            let serde_json::Value::Object(expected) = expected else {
                panic!("expected value not struct");
            };
            let mut b = false;
            for (k, e) in expected {
                if let Some(r) = baseline.get(k) {
                    if r != e {
                        baseline_unequal += 1;
                        b = true;
                    }
                } else {
                    baseline_miss += 1;
                    b = true;
                }
            }
            if b {
                baseline_fail += 1;
            } else {
                baseline_success += 1;
            }
            let report = match manager
                .apply(
                    None,
                    yammer::ChatRequest {
                        model: "qwq:32b-q8_0".to_string(),
                        format: None,
                        keep_alive: None,
                        messages: vec![],
                        tools: None,
                        stream: None,
                        options: serde_json::json! {{
                            "num_ctx": 16_000
                        }},
                    },
                    &point.text,
                )
                .await
            {
                Ok(returned) => returned,
                Err(_) => {
                    conflict += 1;
                    eprintln!("CONFLICT!");
                    continue;
                }
            };
            let serde_json::Value::Object(returned) = report.value() else {
                panic!("returned value not struct {:?}", report.value());
            };
            let mut f = false;
            for (k, e) in expected {
                if let Some(r) = returned.get(k) {
                    if r != e {
                        experimental_unequal += 1;
                        f = true;
                    }
                } else {
                    experimental_miss += 1;
                    f = true;
                }
            }
            if f {
                fail += 1;
                eprintln!(
                    "{}",
                    serde_json::to_string_pretty(
                        &serde_json::json! {{ "report": report, "point": point, "expected": expected }}
                    )
                    .unwrap()
                );
            } else {
                success += 1;
            }
            eprintln!(
                "baseline={baseline_success}/{} experimental={success}/{}",
                baseline_success + baseline_fail,
                success + fail + conflict
            );
        }
    }
    println!(" success {success} failure {fail} conflict {conflict} baseline-success {baseline_success} baseline-miss {baseline_miss} baseline-unequal {baseline_unequal} experimental-miss {experimental_miss} experimental-unequal {experimental_unequal}");
}
