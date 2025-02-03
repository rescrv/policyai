use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use arrrg::CommandLine;
use rand::prelude::*;

#[derive(Clone, Default, Debug, Eq, PartialEq, arrrg_derive::CommandLine)]
struct Options {
    #[arrrg(optional, "The ollama host to connect to.")]
    host: Option<String>,
    #[arrrg(
        required,
        "This many tweets will be selected to have policies applied."
    )]
    samples: usize,
    #[arrrg(required, "This many policies will be selected per tweet.")]
    policies: usize,
    #[arrrg(required, "The model to use for generating policies.")]
    model: String,
    #[arrrg(nested)]
    param: yammer::Parameters,
    #[arrrg(
        required,
        "The number of successful verifications required to select a policy."
    )]
    success: usize,
    #[arrrg(
        required,
        "The number of total verifications to perform for each policy."
    )]
    total: usize,
}

#[tokio::main]
async fn main() -> Result<(), std::io::Error> {
    let (options, free) = Options::from_command_line_relaxed(
        "USAGE: policyai-generate-semantic-injections [OPTIONS] TWEETS",
    );
    if free.len() != 1 {
        eprintln!("expected TWEETS");
        std::process::exit(13);
    }
    let tweets_file = BufReader::new(OpenOptions::new().read(true).open(&free[0]).unwrap());
    let mut tweets = vec![];
    for line in tweets_file.lines() {
        let line = line?;
        let json: serde_json::Value = serde_json::from_str(&line)?;
        if let serde_json::Value::String(s) = json {
            tweets.push(s)
        } else {
            eprintln!("{line} is not a string");
            continue;
        }
    }
    let mut rng = rand::rng();
    for _ in 0..options.samples {
        let tweet = tweets.choose(&mut rng).unwrap();
        let mut injections: Vec<String> = vec![];
        while injections.len() < options.policies {
            let system = r#"
A user is developing an application for custom policy-driven extraction of information from a stream
of tweets.  To do this, they will specify in plain language a pattern that matches the tweet and an
action to perform when the pattern specified in the rule matches.

Your job is to write a sample policy for a given tweet.  I will give you the tweet and you will
give me an English sentence that specifies some property of the tweet.

Restrictions:
- Provide just one sentence of response.
- Do not provide any pro-forma formatting or exposition.
- Provide your response in active voice with straightforward instructions, e.g., "Policy:  The tweet
  is about deep learning and neural networks."
- Do not provide instructions for what to extract.  Your responsibility is limited to simply
  specifying the pattern of text to match using natural language.
- Do not tell me why you chose the policy.
"#
            .to_string();
            let prompt = format!(
                r#"
Tweet:
{tweet}
"#
            );
            let req = yammer::GenerateRequest {
                model: options.model.clone(),
                prompt,
                format: None,
                images: None,
                keep_alive: None,
                suffix: None,
                system: Some(system.clone()),
                template: None,
                stream: Some(false),
                raw: None,
                options: Some(options.param.clone().into()),
            };
            let resp = req
                .make_request(&yammer::ollama_host(options.host.clone()))
                .send()
                .await
                .unwrap()
                .error_for_status()
                .unwrap()
                .json::<yammer::GenerateResponse>()
                .await
                .unwrap();
            if resp.response.contains("\n\n") {
                continue;
            }
            let mut response = resp.response;
            while let Some(r) = response.strip_prefix("Policy:") {
                response = r.trim().to_string();
            }
            if policyai::data::policy_applies(
                None,
                yammer::GenerateRequest {
                    model: options.model.to_string(),
                    prompt: "".to_string(),
                    format: None,
                    images: None,
                    keep_alive: None,
                    suffix: None,
                    system: None,
                    template: None,
                    stream: Some(false),
                    raw: None,
                    options: Some(options.param.clone().into()),
                },
                tweet,
                &response,
                options.success,
                options.total,
            )
            .await
            .unwrap()
            {
                injections.push(response);
            }
        }
        println!(
            "{}",
            serde_json::to_string(&policyai::data::SemanticInjection {
                injections,
                tweet: tweet.to_string()
            })
            .unwrap()
        );
    }
    Ok(())
}
