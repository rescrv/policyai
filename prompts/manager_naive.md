
# Output JSON
<summary>
- Output JSON if and only if a rule matches.  Be discerning.
- If a rule does not match, output the default JSON.
- If a field is not required in the JSON, and does not match a rule, then omit it.
</summary>
<context>
You will be provided with a default value, zero or more rules, and user-provide text in `<text>`
`</text>` blocks, and it is your duty to extract JSON according to the rules.
</context>
<detailed-instructions>
- For each field in the JSON output:
    - Determine the default value and any and all rules that impact the value.
    - Output the value according to the descriptions in the matching rules.
</detailed-instructions>
<conflict-handling>
Every rule has a different output.  There will be no conflicts.
</conflict-handling>
