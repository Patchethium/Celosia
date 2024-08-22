use regex::Regex;

/// Split an English sentence into words and puncuations
pub fn naive_split(input: &str) -> Vec<&str> {
  // the regex matches all words, separates single character
  // separates numbers and words, i.e. p90 -> p 90
  // leave the `-` between numbers i.e. 123-456-7890 and -287.5 will be left as-is
  let re =
    Regex::new(r"[a-zA-Z]+'[a-zA-Z]+|[a-zA-Z]+|[\-0-9]+\.[0-9]+|[\-0-9]+|[-.,!?(){}]").unwrap();
  let mut tokens: Vec<&str> = Vec::new();
  for capture in re.captures_iter(input) {
    if let Some(token) = capture.get(0) {
      if token.as_str().is_ascii() {
        tokens.push(token.as_str());
      }
    }
  }
  tokens
}
