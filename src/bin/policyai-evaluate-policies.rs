use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use policyai::data::TestDataPoint;
use policyai::Manager;

/*
pub async fn naive_apply(
    policies: &[Policy],
    prompt: &str,
) -> Result<serde_json::Value, ApplyError> {
    let mut messages = vec![ChatMessage {
        role: "system".to_string(),
        content: "
You are a policy application machine.

Your task is to apply fragments of policies to build a JSON object.
"
        .to_string(),
        images: None,
        tool_calls: None,
    }];
    let mut required = vec![];
    let mut properties = serde_json::json! {{}};
    for policy in policies.iter() {
        let mut content = policy.prompt.clone();
        for field in policy.r#type.fields.iter() {
            match field {
                Field::Bool {
                    name,
                    default: _,
                    on_conflict,
                } => {
                    required.push(name.clone());
                    properties[name.clone()] = bool::json_schema();
                }
                Field::Number {
                    name,
                    default,
                    on_conflict,
                } => {
                    required.push(name.clone());
                    properties[name.clone()] = f64::json_schema();
                }
                Field::String {
                    name,
                    default,
                    on_conflict,
                } => {
                    required.push(name.clone());
                    properties[name.clone()] = String::json_schema();
                }
                Field::StringEnum {
                    name,
                    values,
                    default,
                    on_conflict,
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
        model: "phi4".to_string(),
        messages,
        format: Some(schema),
        stream: Some(false),
        keep_alive: None,
        options: serde_json::json! {{
            "temperature": 0.1,
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
*/

#[tokio::main]
async fn main() {
    let mut success = 0u64;
    let mut conflict = 0u64;
    let mut miss = 0u64;
    let mut fail = 0u64;
    let mut unequal = 0u64;
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
            let report = match manager
                .apply(
                    None,
                    yammer::ChatRequest {
                        model: "phi4".to_string(),
                        format: None,
                        keep_alive: None,
                        messages: vec![],
                        tools: None,
                        stream: None,
                        options: serde_json::json! {{ "temperature": 0.1, "num_ctx": 16_000 }},
                    },
                    &point.tweet,
                )
                .await
            {
                /*
                let returned = match naive_apply(&case.policy, &case.prompt).await {
                */
                Ok(returned) => returned,
                Err(_) => {
                    conflict += 1;
                    continue;
                }
            };
            let expected = &point.expected;
            let serde_json::Value::Object(returned) = report.value() else {
                panic!("returned value not struct {:?}", report.value());
            };
            let serde_json::Value::Object(expected) = expected else {
                panic!("expected value not struct");
            };
            let mut f = false;
            for (k, e) in expected {
                if let Some(r) = returned.get(k) {
                    if r != e {
                        unequal += 1;
                        f = true;
                    }
                } else {
                    miss += 1;
                    f = true;
                }
            }
            if f {
                fail += 1;
                eprintln!(
                    "{}",
                    serde_json::to_string_pretty(
                        &serde_json::json! {{ "report": report, "point": point }}
                    )
                    .unwrap()
                );
            } else {
                success += 1;
            }
        }
    }
    println!("success {success} failure {fail} conflict {conflict} miss {miss} unequal {unequal}");
}
