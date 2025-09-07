use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use policyai::Policy;

fn main() {
    let mut verified = 0u64;
    for file in std::env::args().skip(1) {
        let file = OpenOptions::new()
            .read(true)
            .open(file)
            .expect("could not read input");
        let file = BufReader::new(file);
        for line in file.lines() {
            let line = line.expect("could not read data");
            let _policy: Policy = match serde_json::from_str(&line) {
                Ok(policy) => policy,
                Err(err) => {
                    eprintln!("error parsing policy {line}: {err}");
                    continue;
                }
            };
            verified += 1;
        }
    }
    eprintln!("verified {verified} policies");
}
