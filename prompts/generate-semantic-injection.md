Your task is to assume that the following message's conditional ask is true.

Assume it is true and generate a sample response. Your sample response must be a JSON object with the minimal number of fields necessary to satisfy the ask.

Think carefully about how you answer to ensure that for every output field "quux" there is a \"quux\" to be found.
If the user does not request a field to be filled in, omit it from the object.
Do not infer output fields from the condition that makes the policy apply.  Text such as "when the email is
about AI" describes when the policy applies; it does not mean to output a category field unless the user asks
you to set the category field.
Include every field the user explicitly asks to set, mark, or add even if the value equals that field's default.
Never include empty arrays, null fields, or fields that are merely possible according to the schema.

Conditional text describes when the policy applies; it is not output data. If the input says "When ...:" or "If ...," do not create JSON fields from the condition. Create JSON fields only from the requested action.

Every requested action assignment must appear in the output object.

Use JSON value types that match the schema. If the ask quotes a boolean or number, convert it to the schema's JSON type instead of returning a string.

Example Input: Extract the hashtags to field "foo" and set "bar" to true.
Example Output: {"foo": "#HashTag", "bar": true}

Example Input: Set "unread" to "true".
Example Output: {"unread": true}

Example Input: When the email is about AI: Set "priority" to "low" and "unread" to true.
Example Output: {"priority": "low", "unread": true}

Notice how unrelated fields are not set because they do not appear in the input.

Always output JSON and only the relevant JSON.
