use yammer::JsonSchema;

use crate::Policy;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SemanticInjection {
    pub injections: Vec<String>,
    pub text: String,
}

pub async fn policy_applies(
    host: Option<String>,
    mut req: yammer::GenerateRequest,
    text: &str,
    semantic_injection: &str,
    k: usize,
    n: usize,
) -> Result<bool, yammer::Error> {
    let mut success = 0;
    let mut total = 0;
    while success < k && total < n {
        total += 1;
        let system = r#"
You are tasked with determining whether provided CRITERIA match some UNSTRUCUTRED DATA.

Your job is to return a JSON boolean that indicates if the user's CRITERIA matches the UNSTRUCTURED DATA.

Return {"policy_applies": true} if you are sure the UNSTRUCTURED DATA matches the CRITERIA.
Return {"policy_applies": false} otherwise.
"#
        .to_string();
        let prompt = format!(
            r#"
UNSTRUCTURED DATA:
{text}

CRITERIA:
{semantic_injection}
"#
        );
        req.prompt = prompt;
        req.stream = Some(false);
        req.system = Some(system);
        #[derive(serde::Deserialize, yammer_derive::JsonSchema)]
        struct Answer {
            policy_applies: bool,
            policy_does_not_apply: bool,
        }
        req.format = Some(Answer::json_schema());
        let resp = req
            .make_request(&yammer::ollama_host(host.clone()))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json::<yammer::GenerateResponse>()
            .await
            .unwrap();
        let Ok(answer) = serde_json::from_str::<Answer>(&resp.response) else {
            continue;
        };
        if answer.policy_applies && !answer.policy_does_not_apply {
            success += 1;
        }
    }
    Ok(success >= k)
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct DecidableSemanticInjection {
    pub positives: Vec<String>,
    pub negatives: Vec<String>,
    pub text: String,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct InjectableAction {
    pub inject: String,
    pub action: serde_json::Value,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct TestDataPoint {
    pub text: String,
    pub policies: Vec<Policy>,
    pub expected: serde_json::Value,
}
