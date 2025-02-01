MODEL := hf.co/unsloth/phi-4-GGUF:f16
SAMPLES := 1000
POLICIES := 5
MATCHING := 1

K := 2
N := 3

.DELETE_ON_ERROR:

all: data/semantic-injections.jsonl data/decidables.jsonl data/actions.jsonl data/test-data.jsonl report

data/semantic-injections.jsonl: data/ai-tweets.jsonl
	cargo run --bin policyai-generate-semantic-injections -- --model $(MODEL) --samples $(SAMPLES) --policies $(POLICIES) --success $(K) --total $(N) $< > $@

data/decidables.jsonl: data/semantic-injections.jsonl
	cargo run --bin policyai-generate-decidables -- --model $(MODEL) --samples $(SAMPLES) --policies $(POLICIES) --success $(K) --total $(N) $< > $@

data/actions.jsonl: data/policy
	cargo run --bin policyai-generate-actions -- < $< > $@

data/test-data.jsonl: data/actions.jsonl data/decidables.jsonl
	cargo run --bin policyai-generate-test-data -- --actions data/actions.jsonl --decidables data/decidables.jsonl --samples $(SAMPLES) --policies $(POLICIES) --matching $(MATCHING) --policy data/policy > $@

report: data/test-data.jsonl
	#cargo run --bin policyai-evaluate-policies data/test-data.jsonl
	echo NOP report
