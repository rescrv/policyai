# PolicyAI

**Composable, conflict-aware policies for reliable agents**

## The Problem: LLMs Can't Handle Conflicting Instructions

When building agents, you quickly discover that large language models have subtle biases that make structured outputs unreliable:

- **Frequency bias**: If `"priority": "low"` appears 10× more often than `"priority": "high"` in your prompts, the LLM will default to "low" even when "high" is correct.
- **Key name bias**: The names of your JSON fields influence the model's outputs in unexpected ways (e.g. priority).
- **Context leakage**: Mention "the building is on fire" in your prompt and watch the LLM assign high priority to unrelated low-priority messages.
- **Conflicting instructions**: Tell the LLM to prioritize messages from Alice and deprioritize messages from Bob—then send a message from both. The LLM won't report a conflict; it will silently pick one instruction to follow.

These aren't edge cases. They're fundamental limitations of how LLMs process instructions. Your agent appears to work in testing, then fails unpredictably in production.

## What PolicyAI Does

PolicyAI provides a layer on top of structured outputs that makes agent behavior **composable** and **conflict-aware**:

- **Each policy is independent**: Write and test policies in isolation, then compose them.
- **Conflicts are detected**: When instructions conflict, you get an error instead of silent bias.
- **Conflict resolution is explicit**: Choose how to handle conflicts (agreement required, largest value wins, or default).
- **Monotonic overrides**: Use the "largest value" strategy to make important values "sticky"—once set high, they stay high.

PolicyAI trades latency and cost for **reliability and debuggability**. If you're building production agents where correctness matters, that's a trade worth making.

## Show Me the Problem

Here's what happens with vanilla structured outputs when you have conflicting policies:

```python
# Your agent instructions
"""
- When Alice sends a message, set priority to HIGH
- When Bob sends a message, set priority to LOW
"""

# Message from: alice@example.com, bob@example.com
# LLM output: {"priority": "LOW"}  # Wrong! But which instruction should it follow?
# The model picked one silently. No error. No warning.
```

With PolicyAI, this scenario produces a conflict error because two policies disagree on the priority field's value, and you've configured it to require agreement.

## How PolicyAI Works

### 1. Define a PolicyType

A PolicyType is like a schema, but with conflict resolution strategies:

```rust
use policyai::{PolicyType, Field, OnConflict};

let policy_type = PolicyType {
    name: "EmailPolicy".to_string(),
    fields: vec![
        Field::Bool {
            name: "unread".to_string(),
            default: true,
            on_conflict: OnConflict::Default,
        },
        Field::StringEnum {
            name: "priority".to_string(),
            values: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
            default: None,
            on_conflict: OnConflict::LargestValue,  // "high" wins over "low"
        },
        Field::StringArray {
            name: "labels".to_string(),
        },
    ],
};
```

### 2. Create Policies with Semantic Injections

A semantic injection is a natural language instruction that generates structured actions:

```rust
let policy1 = policy_type
    .with_semantic_injection(
        &client,
        "If the email is about football, mark \"unread\" false with low \"priority\""
    )
    .await?;

let policy2 = policy_type
    .with_semantic_injection(
        &client,
        "If the email is from mom@example.org, set high \"priority\" and add Family \"label\""
    )
    .await?;

let policy3 = policy_type
    .with_semantic_injection(
        &client,
        "If the email is about shopping, add Shopping \"label\""
    )
    .await?;
```

### 3. Compose and Apply

```rust
let mut manager = Manager::default();
manager.add(policy1);
manager.add(policy2);
manager.add(policy3);

let report = manager.apply(
    &client,
    template,
    "From: mom@example.org\nSubject: Shopping for football gear",
    None
).await?;

// Result: unread=false, priority=high, labels=["Family", "Shopping"]
// - policy1 sets unread=false, priority=low
// - policy2 sets priority=high (wins via LargestValue)
// - policy3 adds Shopping label
// - labels compose (arrays merge)
```

The policies compose cleanly because:
- `priority` uses `OnConflict::LargestValue` → "high" overrides "low"
- `labels` is an array → values merge automatically
- `unread` uses `OnConflict::Default` → takes the default value when conflicts occur

## Conflict Resolution Strategies

PolicyAI provides three strategies for handling conflicts:

### Agreement
All policies must agree on the value, or you get a conflict error. Best for fields where inconsistency indicates a logic error in your policies.

```rust
Field::String {
    name: "template".to_string(),
    default: None,
    on_conflict: OnConflict::Agreement,
}
```

### LargestValue
The largest value wins. This makes important values "sticky" and enables monotonic overrides:
- For bools: `true > false`
- For numbers: `10 > 5`
- For strings: longer strings win
- For enums: values later in the list win

```rust
Field::StringEnum {
    name: "priority".to_string(),
    values: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
    on_conflict: OnConflict::LargestValue,  // "high" > "medium" > "low"
}
```

**Why this matters**: Once a policy sets priority to "high", no other policy can downgrade it to "low". This prevents surprising interactions between policies.

### Default

Use the field type's default behavior (usually last-writer-wins, but arrays append) when conflicts occur. Useful for fields where you want predictable behavior regardless of policy interactions.

```rust
Field::Bool {
    name: "unread".to_string(),
    default: true,
    on_conflict: OnConflict::Default,
}
```

## PolicyType Syntax

PolicyAI provides a concise syntax for defining policy types:

```text
type policyai::EmailPolicy {
    unread: bool = true,
    priority: ["low", "medium", "high"] @ highest wins,
    category: ["ai", "distributed systems", "other"] @ agreement = "other",
    template: string @ agreement,
    labels: [string],
}
```

You can parse this syntax directly:

```rust
let policy_type = PolicyType::parse(r#"
    type EmailPolicy {
        priority: ["low", "high"] @ highest wins,
        labels: [string]
    }
"#)?;
```

## Use Cases for Agents

PolicyAI excels when your agent needs to:

- **Triage emails or notifications**: Apply multiple categorization rules that may interact
- **Process RSS feeds**: Extract structured metadata from articles using composable rules
- **Label documents**: Assign categories, tags, priorities based on content
- **Extract metadata**: Any scenario where discrete documents need structured descriptors

PolicyAI is **not** the right tool when:
- You have a single, simple classification task
- Latency is more important than correctness
- Your policies never interact or conflict

## Scaling with Policy Retrieval

For production agents with large policy sets, you don't want to apply every policy to every input. Use **vector retrieval** to select relevant policies dynamically:

### Pattern: PolicyAI + Chroma

(Example is illustrative, but likely needs work to work because Claude hallucinated some of this)

```rust
use chromadb::{ChromaClient, Collection};
use policyai::{Policy, Manager};

async fn process_with_retrieval(
    client: &Anthropic,
    chroma: &Collection,
    input: &str,
) -> Result<Report, Box<dyn std::error::Error>> {
    // 1. Retrieve relevant policies from vector database
    let results = chroma.query(
        vec![input.to_string()],
        5,  // top 5 most relevant policies
        None,
        None,
        None,
    ).await?;

    // 2. Deserialize policies from metadata
    let policies: Vec<Policy> = results.metadatas
        .into_iter()
        .flatten()
        .filter_map(|meta| {
            serde_json::from_value(meta.get("policy")?.clone()).ok()
        })
        .collect();

    // 3. Apply only relevant policies
    let mut manager = Manager::default();
    for policy in policies {
        manager.add(policy);
    }

    let report = manager.apply(client, template, input, None).await?;
    Ok(report)
}
```

### Storing Policies for Retrieval

When adding policies to your vector database, store both the semantic injection and the full policy:

```rust
// Create policy
let policy = policy_type
    .with_semantic_injection(
        &client,
        "If email from VIP, set high priority"
    )
    .await?;

// Store in Chroma with embedding of the semantic injection
chroma.add(
    vec![Uuid::new_v4().to_string()],  // id
    vec![policy.prompt.clone()],         // text to embed
    Some(vec![serde_json::json!({
        "policy": policy,
        "type": "email_triage",
    })]),
    None,
).await?;
```

### Why This Works

- **Semantic injections are natural language**: Vector databases embed them naturally
- **Retrieval filters noise**: Only relevant policies are applied, reducing conflicts
- **Scalable**: Support thousands of policies without performance degradation
- **Dynamic**: Add/update policies without redeploying your agent

### Retrieval Best Practices

1. **Embed the semantic injection, not the action**: The natural language prompt (`policy.prompt`) captures intent
2. **Store full policy in metadata**: Retrieve the complete `Policy` object for application
3. **Use top-k = 3-10**: Start small; more policies = more potential conflicts
4. **Monitor conflict rates**: If retrieval pulls conflicting policies, tune your embeddings or increase specificity

This pattern combines the best of both worlds:
- **Vector retrieval** for relevance and scale
- **PolicyAI** for correctness and composability

## Tradeoffs

PolicyAI sacrifices performance for reliability:

| Metric | vs Vanilla Structured Outputs |
|--------|-------------------------------|
| Latency | Higher (additional LLM calls) |
| Token Usage | Higher (policy composition) |
| Cost | Higher (more tokens) |
| Reliability | Much higher (conflict detection) |
| Debuggability | Much higher (isolated policies) |

**Why it's worth it**: In production agents, silent failures are expensive. PolicyAI makes agent behavior predictable and testable. You can verify each policy independently, then compose them with some confidence.

## Getting Started

Add PolicyAI to your `Cargo.toml`:

```toml
[dependencies]
policyai = "0.2"
```

Basic usage:

```rust
use policyai::{PolicyType, Field, OnConflict, Manager};
use claudius::Anthropic;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Anthropic::new(None)?;

    // Define your policy type
    let policy_type = PolicyType::parse(r#"
        type MyPolicy {
            priority: ["low", "high"] @ highest wins
        }
    "#)?;

    // Create a policy from natural language
    let policy = policy_type
        .with_semantic_injection(&client, "Set high priority for urgent messages")
        .await?;

    // Apply it
    let mut manager = Manager::default();
    manager.add(policy);

    let report = manager.apply(
        &client,
        template,
        "This is urgent!",
        None
    ).await?;

    println!("{}", report.value());
    Ok(())
}
```

## Tools

PolicyAI includes tools for testing and debugging:

- `policyai-verify-policies`: Verify policies are well-formed
- `policyai-regression-report`: Generate reports on policy behavior
- `policyai-extract-regressions`: Extract failing cases for analysis
- `policyai-regressions-to-examples`: Convert regressions to test examples

## Implementation Note

PolicyAI deliberately orders arguments in tool calls carefully. Agents are surprisingly susceptible to argument order, so the framework maintains consistent ordering to avoid bias.

## Current Status

- **Model support**: Anthropic Claude only (currently)
- **License**: Apache-2.0
- **Status**: Active development

## Examples

See the [examples/](examples/) directory for:
- Generating semantic injections
- Creating test data
- Evaluating policies

## Contributing

Issues and pull requests welcome at https://github.com/rescrv/policyai

## License

Apache-2.0
