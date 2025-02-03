use std::io::{BufRead, BufReader};

// SYSTEM:
// You generate tweets for the user.  They will give the topic and you will generate them one per line.
//
// SYSTEM:
// Write 10 tweets about artificial intelligence and computer science.  Write them one per line,numbered.
//
// Example output:
//
// Here are the tweets
//
// 1.  Tweet one.
// 2.  Tweet two.
// 3.  Tweet three.
// 4.  Tweet four.
// 5.  Tweet five.
// 6.  Tweet six.
// 7.  Tweet seven.
// 8.  Tweet eight.
// 9.  Tweet nine.
// 10.  Tweet ten.

fn main() {
    for arg in std::env::args().skip(1) {
        let fin = std::fs::OpenOptions::new().read(true).open(arg).unwrap();
        let fin = BufReader::new(fin);
        for line in fin.lines() {
            let line = line.unwrap();
            let json: serde_json::Value = serde_json::from_str(&line).unwrap();
            let msg = &json["response"]["body"]["choices"];
            if let serde_json::Value::Array(arr) = msg {
                for value in arr {
                    let msg = &value["message"];
                    let chat: yammer::ChatMessage = serde_json::from_value(msg.clone()).unwrap();
                    for line in chat.content.split_terminator("\n") {
                        if line
                            .chars()
                            .next()
                            .map(|c| c.is_ascii_digit())
                            .unwrap_or_default()
                        {
                            let Some((_, tweet)) = line.split_once(' ') else {
                                continue;
                            };
                            let tweet = tweet.trim();
                            println!("{}", serde_json::to_value(tweet).unwrap());
                        }
                    }
                }
            }
        }
    }
}
