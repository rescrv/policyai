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

        let rules_content = regression["report"]["messages"][0]["content"]
            .as_str()
            .unwrap_or("");
        let text = regression["input"]["text"].as_str().unwrap_or("");
        let ir = &regression["report"]["ir"];

        let output = if ir.is_null() {
            "{}".to_string()
        } else {
            serde_json::to_string_pretty(ir).unwrap()
        };

        println!("<example>");
        println!("<input>");
        print!("{}", rules_content);
        println!("<text>{}</text>", text);
        println!("</input>");
        println!("<output>");
        println!("{}", output);
        println!("</output>");
        println!("</example>");
    }
}
