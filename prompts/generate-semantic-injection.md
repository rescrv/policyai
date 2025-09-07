Your task is to assume that the following message's conditional ask is true.

Assume it is true and generate a sample response.  Your sample response should focus on generating a JSON
object with the minimal number of fields necessary to satisfy the response.

Think carefully about how you answer to ensure that for every output field "quux" there is a \"quux\" to be found.
If the user does not request a field to be filled in, omit it from the object.

Example Input: Extract the hashtags to field \"foo\" and set \"bar\" to true.
Example Output: {"foo": "\#HashTag\", "bar": true}

Notice how in this example, \"baz\" is not set because it does not appear in the example input.

Always output JSON and only the relevant JSON.
