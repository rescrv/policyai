Your task is to assume that the following message's conditional ask is true.

Assume it is true and generate a sample response. Your sample response must be a JSON object with the minimal number of fields necessary to satisfy the ask.

Only output fields the user explicitly requests. Do not include a field just because it exists in the schema, has a default value, or is semantically related to the ask.

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
