use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Read};

use guacamole::combinators::*;
use guacamole::Guacamole;

use policyai::{Field, Policy, PolicyType};

fn main() {
    let mut buf = vec![];
    std::io::stdin()
        .read_to_end(&mut buf)
        .expect("could not read policy type on stdin");
    let buf = String::from_utf8(buf).expect("policy type should be UTF8");
    let policy_type = PolicyType::parse(&buf).expect("policy type should be valid");
    println!("{}", serde_json::to_value(policy_type).unwrap());
}
