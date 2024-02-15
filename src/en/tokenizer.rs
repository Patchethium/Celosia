use anyhow::{Context, Result};
use regex::Regex;

/// Split an English sentence into words and puncuations
pub fn naive_split(input: &str) -> Result<Vec<&str>> {
  let re = Regex::new(r"(\w+|[.,!?(){}])")?;
  let mut result: Vec<&str> = Vec::new();

  for capture in re.captures_iter(input) {
    let token = capture.get(0).context("no items")?.as_str();
    result.push(token);
  }

  Ok(result)
}
