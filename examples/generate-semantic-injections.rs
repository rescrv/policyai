use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use arrrg::CommandLine;
use claudius::{Anthropic, CacheControlEphemeral, Effort, Model, SystemPrompt, TextBlock};
use policyai::data::{EffortArg, ThinkingArg};
use rand::prelude::*;

#[derive(Clone, Default, Debug, Eq, PartialEq, arrrg_derive::CommandLine)]
struct Options {
    #[arrrg(required, "This many texts will be selected to have policies applied.")]
    samples: usize,
    #[arrrg(required, "This many policies will be selected per text.")]
    policies: usize,
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
    #[arrrg(optional, "Anthropic model to use for generation and verification.")]
    model: Option<String>,
    #[arrrg(optional, "Maximum output tokens for generation and verification.")]
    max_tokens: Option<u32>,
    #[arrrg(optional, "Thinking config: adaptive, disabled, or a token budget.")]
    thinking: Option<ThinkingArg>,
    #[arrrg(optional, "Adaptive thinking effort: low, medium, or high.")]
    effort: Option<EffortArg>,
    #[arrrg(optional, "Maximum candidate attempts per accepted policy.")]
    max_attempts_per_policy: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<(), claudius::Error> {
    let (options, free) =
        Options::from_command_line_relaxed("USAGE: generate-semantic-injections [OPTIONS] TEXTS");
    if free.len() != 1 {
        eprintln!("expected TEXTS");
        std::process::exit(13);
    }
    let model = options
        .model
        .as_deref()
        .map(|model| model.parse::<Model>().unwrap())
        .unwrap_or(policyai::DEFAULT_MODEL);
    let max_tokens = options.max_tokens.unwrap_or(2048);
    let max_attempts = options.max_attempts_per_policy.unwrap_or(10) * options.policies;
    let thinking = Some(
        options
            .thinking
            .map(Into::into)
            .unwrap_or_else(claudius::ThinkingConfig::adaptive),
    );
    let effort = Some(options.effort.map(Into::into).unwrap_or(Effort::High));
    let texts_file = BufReader::new(OpenOptions::new().read(true).open(&free[0]).unwrap());
    let mut texts = vec![];
    for line in texts_file.lines() {
        let line = line?;
        let json: serde_json::Value = serde_json::from_str(&line)?;
        if let serde_json::Value::Object(obj) = json {
            if let Some(serde_json::Value::String(text)) = obj.get("text") {
                texts.push(text.clone())
            }
        } else {
            eprintln!("{line} is not a string");
            continue;
        }
    }
    let mut rng = rand::rng();
    let client = Anthropic::new(None).expect("could not connect to claude");
    for _ in 0..options.samples {
        let text = texts.choose(&mut rng).unwrap();
        let mut injections: Vec<String> = vec![];
        let mut rationales: Vec<String> = vec![];
        let mut attempts = 0;
        while injections.len() < options.policies && attempts < max_attempts {
            attempts += 1;
            let system = r#"
You are an expert writer.  We are developing an instruction-processing engine that takes as input
instructions and text to output JSON.  Every instruction has two parts, first it has the _semantic
injection_.  This is natural language text that says something about the content being processed.
Second, an instruction has an associated output.  This, too, is a natural language text but it says
something about the JSON we are constructing.

Our task for now is to take text and generate a semantic injection for it.

Restrictions:
- Provide just one sentence of response.
- Do not provide any pro-forma formatting or exposition.
- Provide your response in active voice with straightforward instructions, e.g., "The text is about
  deep learning and neural networks."
- Do not provide instructions for what to output.  Your responsibility is limited to simply
  specifying the pattern of text to match using natural language.
- Do not tell me why you chose the policy.
"#
            .to_string();
            let prompt = format!(
                r#"
Text:
{text}
"#
            );
            let req = claudius::MessageCreateParams {
                max_tokens,
                messages: vec![prompt.into()],
                model: model.clone(),
                cache_control: None,
                metadata: None,
                output_config: policyai::data::output_config_for_thinking(thinking, effort),
                output_format: None,
                stop_sequences: None,
                system: Some(SystemPrompt::from_blocks(vec![TextBlock {
                    text: system.to_string(),
                    cache_control: Some(CacheControlEphemeral::new()),
                    citations: None,
                }])),
                temperature: None,
                thinking,
                tool_choice: None,
                tools: None,
                top_k: None,
                top_p: None,
                stream: false,
                betas: None,
            };
            let resp = client.send(req).await?;
            let mut injection = String::new();
            let mut thought = String::new();
            for content in resp.content.iter() {
                match content {
                    claudius::ContentBlock::Text(t) => injection.push_str(&t.text),
                    claudius::ContentBlock::Thinking(t) => thought.push_str(&t.thinking),
                    _ => {}
                }
            }
            eprintln!("text: {text}\ninjection: {injection}\nthought: {thought}\n");
            if policyai::data::policy_applies(
                &client,
                text,
                &injection,
                options.success,
                options.total,
                model.clone(),
                max_tokens,
                thinking,
                effort,
            )
            .await?
            {
                injections.push(injection);
                rationales.push(thought);
                eprintln!(
                    "accepted {} of {} policies after {attempts} attempts",
                    injections.len(),
                    options.policies
                );
            } else {
                eprintln!(
                    "rejected candidate; accepted {} of {} policies after {attempts} attempts",
                    injections.len(),
                    options.policies
                );
            }
        }
        if injections.len() < options.policies {
            return Err(claudius::Error::unknown(format!(
                "only generated {} of {} semantic injections after {attempts} attempts",
                injections.len(),
                options.policies
            )));
        }
        println!(
            "{}",
            serde_json::to_string(&policyai::data::SemanticInjection {
                injections,
                rationales,
                text: text.to_string()
            })
            .unwrap()
        );
    }
    Ok(())
}
