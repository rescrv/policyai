# Output JSON
<summary>
- Always output one valid JSON object.
- Output a rule's action fields if and only if that rule matches.  Be discerning.
- If no rules match, output the default JSON plus "__rule_numbers__": [] and "__justification__".
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
- Always include "__rule_numbers__" as the sorted list of matching rule indexes.
- Always include "__justification__" explaining why the listed rules match and why the omitted rules do not.
</detailed-instructions>
<conflict-handling>
Every rule has a different output.  There will be no conflicts.
</conflict-handling>
<example>
<input>
<rule index=\"1\"><match>The text expresses enthusiasm about learning to code as a pathway to opportunities in the modern digital world.</match><action>When this rule matches, output JSON {\"3256dda3-bbd1-4d8d-ba29-e6ebe8bb8f42\": []}.</action></rule>
<rule index=\"2\"><match>The text expresses enthusiasm about learning to code as a pathway to opportunities in the modern digital world.</match><action>When this rule matches, output JSON {\"b22a89fc-8ad6-48e7-8e25-47f3c2929782\": []}.</action></rule>
<rule index=\"3\"><match>The text compares learning to code to learning a new language and emphasizes the opportunities it creates in the digital world.</match><action>When this rule matches, output JSON {\"0b84dfeb-f93f-4819-8f2a-26659f85023a\": []}.</action></rule>
<rule index=\"4\"><match>The text discusses the historical and conceptual connections between computer science, mathematics, and logic fields.</match><action>When this rule matches, output JSON {\"8058c105-7e2b-4b72-9665-eb9f78eba1a4\": [\"a\",\"b\",\"c\",\"d\"]}.</action></rule>
<rule index=\"5\"><match>The text discusses learning programming as analogous to acquiring a new language and emphasizes the opportunities it creates in the modern digital world.</match><action>When this rule matches, output JSON {\"e43d3cdf-1f52-4b65-b16f-58fe32d6ca9d\": []}.</action></rule>
<text>Learning to code is like learning a new language; it opens doors to endless possibilities in today's digital age! #LearnToCode #CS</text>
</input>
<output>
{
  "__rule_numbers__": [1, 2, 3, 5],
  "__justification__": "Rules 1, 2, 3, and 5 match the text's enthusiasm about learning to code as a language-like path to digital opportunity. Rule 4 does not match because the text does not discuss history, mathematics, or logic.",
  "3256dda3-bbd1-4d8d-ba29-e6ebe8bb8f42": [],
  "b22a89fc-8ad6-48e7-8e25-47f3c2929782": [],
  "0b84dfeb-f93f-4819-8f2a-26659f85023a": [],
  "e43d3cdf-1f52-4b65-b16f-58fe32d6ca9d": []
}
</output>
</example>
<example>
<input>
<rule index="1"><match>The text contains an enthusiastic social media post about artificial intelligence's current impact and asks readers about exciting AI advancements they've observed.</match><action>When this rule matches, output JSON {"1c5741ca-4431-409d-a044-66d4b0f0f350": "line1\nline2\nindex:175"}.</action></rule><rule index="2"><match>The text discusses exciting AI advancements across multiple sectors and asks readers which application area they find most compelling.</match><action>When this rule matches, output JSON {"d52ff624-c413-445a-95ae-2028629a460b": ""}.</action></rule><rule index="3"><match>The text promotes learning artificial intelligence by recommending Python as a starting programming language due to its versatility and extensive use in AI and machine learning fields.</match><action>When this rule matches, output JSON {"b4033f54-d284-4ebb-8db2-46be150d6845": "string_336"}.</action></rule><rule index="4"><match>The text promotes learning artificial intelligence by recommending Python as a starting programming language due to its versatility and extensive use in AI and machine learning fields.</match><action>When this rule matches, output JSON {"570db76f-0574-4f9d-933d-589c9fb78270": "This is a longer string with index 676"}.</action></rule><rule index="5"><match>The text discusses the potential of quantum computing to enhance artificial intelligence capabilities through faster information processing.</match><action>When this rule matches, output JSON {"325b47c3-6743-4b67-9214-71aa82789529": "special!@#$%^&*()chars"}.</action></rule><text>Artificial intelligence is not just the future; it's reshaping our present! What advancements have you seen that excite you? #AI #TechTalk</text>
</input>
<output>
{
  "__rule_numbers__": [1],
  "__justification__": "Rule 1 matches because the text enthusiastically says AI is reshaping the present and asks readers what advancements excite them. Rules 2, 3, 4, and 5 do not match.",
  "1c5741ca-4431-409d-a044-66d4b0f0f350": "line1\nline2\nindex:175"
}
</output>
</example>
<example>
<input>
<rule index="1"><match>The text discusses advancements in machine learning technology and how computers are becoming more capable of replicating human decision-making processes.</match><action>When this rule matches, output JSON {"da4b64fd-c521-490e-aeb9-1b414d9399f0": true}.</action></rule><rule index="2"><match>The text advocates for creating an inclusive and beneficial future for AI through collaboration for everyone's benefit.</match><action>When this rule matches, output JSON {"5da1e557-2c50-4d7a-841e-7c26173f4d31": true}.</action></rule><rule index="3"><match>The text expresses enthusiasm about the potential of quantum computing for artificial intelligence and asks readers about their excitement for future technological developments.</match><action>When this rule matches, output JSON {"9dbe9e51-bd77-4d43-9763-d220825e37fd": true}.</action></rule><rule index="4"><match>The text explains the foundational role of computer science in artificial intelligence development, highlighting key technical components like algorithms and data structures.</match><action>When this rule matches, output JSON {"6309ece7-2866-41b1-baa3-ebca0063d890": true}.</action></rule><rule index="5"><match>The text discusses the collaborative future between humans and artificial intelligence in the workplace while encouraging adaptation to change and exploration of new opportunities.</match><action>When this rule matches, output JSON {"23dd33f3-65e4-4a7b-928a-506636ca0171": true}.</action></rule><text>Artificial Intelligence is not just a trend, it's a revolution transforming how we interact with technology every day. #AI #TechRevolution</text>
</input>
<output>
{
  "__rule_numbers__": [],
  "__justification__": "No rules match. The text is a broad statement about AI being transformative; it does not discuss machine learning decision-making, inclusive AI collaboration, quantum computing, computer science foundations, or human-AI workplace collaboration."
}
</output>
</example>
