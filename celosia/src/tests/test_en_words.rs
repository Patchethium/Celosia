use celosia::en::Phonemizer;
use lazy_static::lazy_static;

lazy_static! {
  static ref PHONEMIZER: Phonemizer = Phonemizer::default();
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_phonemizer() {
    let text = "I read a book by Jane Austen.";
    let result: Vec<Vec<&str>> = PHONEMIZER.phonemize(text);
    println!("{:?}", result);
  }
  #[test]
  fn test_normal_number() {
    let text = "I read 100 books by Jane Austen.";
    let result = PHONEMIZER.phonemize(text);
    println!("{:?}", result);
  }
  #[test]
  fn test_negative_number() {
    let text = "The absolute zero is -273.15 degree Celsius.";
    let result = PHONEMIZER.phonemize(text);
    println!("{:?}", result);
  }
}
