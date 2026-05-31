<instruction reminder>
- Evaluate the text against each rule.
- Always output one valid JSON object.
- Output a rule's action fields if and only if that rule matches.  Be discerning.
- If no rules match, output the default JSON.
- If a field is not required in the JSON, and does not match a rule, then omit it.
- Always include "__rule_numbers__" as the sorted list of rule indexes whose action fields you output.
- If no rules match, "__rule_numbers__" must be [] and no rule action fields should be output.
</instruction reminder>
