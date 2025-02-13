MODEL := phi4
SAMPLES := 100
POLICIES := 10
MATCHING := 1

K := 3
N := 3

.DELETE_ON_ERROR:

all: report

data/semantic-injections.jsonl: data/ai-tweets.jsonl
	cargo run --example generate-semantic-injections -- --model $(MODEL) --samples $(SAMPLES) --policies $(POLICIES) --success $(K) --total $(N) $< > $@

data/decidables.jsonl: data/semantic-injections.jsonl
	cargo run --example generate-decidables -- --model $(MODEL) --samples $(SAMPLES) --policies $(POLICIES) --success $(K) --total $(N) $< > $@

data/actions.jsonl: data/policy
	cargo run --example generate-actions -- < $< > $@

data/test-data.1.$(POLICIES).jsonl: data/actions.jsonl data/decidables.jsonl
	cargo run --example generate-test-data -- --actions data/actions.jsonl --decidables data/decidables.jsonl --samples $(SAMPLES) --policies $(POLICIES) --matching 1 --policy data/policy > $@

data/test-data.2.$(POLICIES).jsonl: data/actions.jsonl data/decidables.jsonl
	cargo run --example generate-test-data -- --actions data/actions.jsonl --decidables data/decidables.jsonl --samples $(SAMPLES) --policies $(POLICIES) --matching 2 --policy data/policy > $@

data/test-data.3.$(POLICIES).jsonl: data/actions.jsonl data/decidables.jsonl
	cargo run --example generate-test-data -- --actions data/actions.jsonl --decidables data/decidables.jsonl --samples $(SAMPLES) --policies $(POLICIES) --matching 3 --policy data/policy > $@

data/test-data.4.$(POLICIES).jsonl: data/actions.jsonl data/decidables.jsonl
	cargo run --example generate-test-data -- --actions data/actions.jsonl --decidables data/decidables.jsonl --samples $(SAMPLES) --policies $(POLICIES) --matching 4 --policy data/policy > $@

data/test-data.5.$(POLICIES).jsonl: data/actions.jsonl data/decidables.jsonl
	cargo run --example generate-test-data -- --actions data/actions.jsonl --decidables data/decidables.jsonl --samples $(SAMPLES) --policies $(POLICIES) --matching 5 --policy data/policy > $@

report: data/test-data.1.$(POLICIES).jsonl data/test-data.2.$(POLICIES).jsonl data/test-data.3.$(POLICIES).jsonl data/test-data.4.$(POLICIES).jsonl data/test-data.5.$(POLICIES).jsonl
	touch $@
	truncate -s 0 $@
	printf "1 " >> $@ && cargo run --bin policyai-evaluate-policies data/test-data.1.$(POLICIES).jsonl >> $@
	printf "2 " >> $@ && cargo run --bin policyai-evaluate-policies data/test-data.2.$(POLICIES).jsonl >> $@
	printf "3 " >> $@ && cargo run --bin policyai-evaluate-policies data/test-data.3.$(POLICIES).jsonl >> $@
	printf "4 " >> $@ && cargo run --bin policyai-evaluate-policies data/test-data.4.$(POLICIES).jsonl >> $@
	printf "5 " >> $@ && cargo run --bin policyai-evaluate-policies data/test-data.5.$(POLICIES).jsonl >> $@
