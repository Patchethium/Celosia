use celosia::en::phonemizer::Phonemizer;
use lazy_static::lazy_static;

lazy_static! {
  static ref PHONEMIZER: Phonemizer = Phonemizer::default();
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_normal_number() {
    let text = "I read 100 books by Jane Austen.";
    let result = PHONEMIZER.phonemize(text);
    println!("{:?}", result);
  }

  #[test]
  fn test_large_number() {
    // this will be interpreted one by one
    let text = "The speed of light is 299792458 m/s";
    let result = PHONEMIZER.phonemize(text);
    println!("{:?}", result);
  }

  #[test]
  fn test_float_number() {
    let text = "The absolute zero is -273.15 degree Celsius.";
    let result = PHONEMIZER.phonemize(text);
    println!("{:?}", result);
  }

  #[test]
  fn test_phone_number() {
    let text = "My phone number is 123-456-7890.";
    let result = PHONEMIZER.phonemize(text);
    println!("{:?}", result);
  }

  #[test]
  fn test_phone_number_with_parentheses() {
    let text = "My phone number is (123) 456-7890.";
    let result = PHONEMIZER.phonemize(text);
    println!("{:?}", result);
  }
}
