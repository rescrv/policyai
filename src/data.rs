use yammer::JsonSchema;

use crate::Policy;

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SemanticInjection {
    pub injections: Vec<String>,
    pub tweet: String,
}

pub async fn policy_applies(
    host: Option<String>,
    mut req: yammer::GenerateRequest,
    tweet: &str,
    semantic_injection: &str,
    k: usize,
    n: usize,
) -> Result<bool, yammer::Error> {
    let mut success = 0;
    let mut total = 0;
    while success < k && total < n {
        total += 1;
        let system = r#"
A user is developing an application for custom policy-driven extraction of information from a stream
of tweets.  To do this, they will specify in plain language a pattern that matches the tweet and an
action to perform when the pattern specified in the rule matches.

Your job is to return a JSON boolean that indicates if the user's policy matches the tweet.

Be very specific.  The user has chosen their words carefully.

Return {"policy_applies": true} if you are sure the tweet matches the polciy.
Return {"policy_applies": false} otherwise.

Policy:
"#
        .to_string();
        let prompt = format!(
            r#"
Tweet:
{tweet}

Policy:
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
    pub tweet: String,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct InjectableAction {
    pub inject: String,
    pub action: serde_json::Value,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct TestDataPoint {
    pub tweet: String,
    pub policies: Vec<Policy>,
    pub expected: serde_json::Value,
}
