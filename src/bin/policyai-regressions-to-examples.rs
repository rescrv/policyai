use std::fmt::Write as _;
use std::io::{self, BufRead};

use serde_json::Value;

fn main() {
    let stdin = io::stdin();
    let reader = stdin.lock();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.expect("Failed to read line");
        if line.trim().is_empty() {
            continue;
        }

        let regression: Value = serde_json::from_str(&line)
            .unwrap_or_else(|e| panic!("Failed to parse JSON on line {}: {}", line_num + 1, e));

        print!("{}", example_for_regression(&regression));
    }
}

fn example_for_regression(regression: &Value) -> String {
    let rules_content = regression["report"]["messages"][0]["content"]
        .as_str()
        .unwrap_or("");
    let text = regression["input"]["text"].as_str().unwrap_or("");
    let expected = &regression["input"]["expected"];
    let output = if expected.is_null() {
        "{}".to_string()
    } else {
        serde_json::to_string_pretty(expected).unwrap()
    };

    let mut example = String::new();
    writeln!(&mut example, "<example>").unwrap();
    writeln!(&mut example, "<input>").unwrap();
    write!(&mut example, "{rules_content}").unwrap();
    writeln!(&mut example, "<text>{text}</text>").unwrap();
    writeln!(&mut example, "</input>").unwrap();
    writeln!(&mut example, "<output>").unwrap();
    writeln!(&mut example, "{output}").unwrap();
    writeln!(&mut example, "</output>").unwrap();
    writeln!(&mut example, "</example>").unwrap();
    example
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example_output_uses_expected_instead_of_report_ir() {
        let regression = serde_json::json!({
            "input": {
                "text": "example text",
                "expected": {
                    "field": "expected value"
                }
            },
            "report": {
                "messages": [
                    {
                        "content": "rules\n"
                    }
                ],
                "ir": {
                    "field": "bad value"
                }
            }
        });

        let example = example_for_regression(&regression);

        assert!(example.contains("\"field\": \"expected value\""));
        assert!(!example.contains("\"field\": \"bad value\""));
    }

    #[test]
    fn example_output_defaults_to_empty_object_without_expected() {
        let regression = serde_json::json!({
            "input": {
                "text": "example text"
            },
            "report": {
                "messages": [
                    {
                        "content": "rules\n"
                    }
                ]
            }
        });

        let example = example_for_regression(&regression);

        assert!(example.contains("<output>\n{}\n</output>"));
    }
}
