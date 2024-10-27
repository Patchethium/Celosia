use celosia::en::Phonemizer;

#[cfg(test)]
mod tests {
  use super::*;

  fn phonemize(text: &str) -> Vec<Vec<&str>> {
    let mut phonemizer = Phonemizer::default();
    phonemizer.phonemize(text)
  }

  #[test]
  fn test_phonemizer() {
    let text = "I read a book by Jane Austen.";
    let result: Vec<Vec<&str>> = phonemize(text);
    println!("{:?}", result);
  }
  #[test]
  fn test_normal_number() {
    let text = "I read 100 books by Jane Austen.";
    let result = phonemize(text);
    println!("{:?}", result);
  }
  #[test]
  fn test_negative_number() {
    let text = "The absolute zero is -273.15 degree Celsius.";
    let result = phonemize(text);
    println!("{:?}", result);
  }
}
