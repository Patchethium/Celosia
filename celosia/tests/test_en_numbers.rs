use celosia::en::Phonemizer;

#[cfg(test)]
mod tests {
  use super::*;

  fn phonemize(text: &str) -> Vec<Vec<&str>> {
    let mut phonemizer = Phonemizer::default();
    phonemizer.phonemize(text)
  }

  #[test]
  fn test_normal_number() {
    let text = "I read 100 books by Jane Austen.";
    let result = phonemize(text);
    println!("{:?}", result);
  }

  #[test]
  fn test_large_number() {
    // this will be interpreted one by one
    let text = "The speed of light is 299792458 m/s";
    let result = phonemize(text);
    println!("{:?}", result);
  }

  #[test]
  fn test_float_number() {
    let text = "The absolute zero is -273.15 degree Celsius.";
    let result = phonemize(text);
    println!("{:?}", result);
  }

  #[test]
  fn test_phone_number() {
    let text = "My phone number is 123-456-7890.";
    let result = phonemize(text);
    println!("{:?}", result);
  }

  #[test]
  fn test_phone_number_with_parentheses() {
    let text = "My phone number is (123) 456-7890.";
    let result = phonemize(text);
    println!("{:?}", result);
  }
}
