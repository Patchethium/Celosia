use celosia::en::Phonemizer;

#[cfg(test)]
mod tests {
  use super::*;

  fn _test_phonemize(text: &str, phonemes: Vec<Vec<&str>>) {
    let mut phonemizer = Phonemizer::default();
    let pred_phonemes = phonemizer.phonemize(text);
    assert_eq!(pred_phonemes, phonemes);
  }

  #[test]
  fn test_normal_number() {
    let text = "I read 100 books.";
    let phonemes = vec![
      vec!["ay1"],
      vec!["r", "iy1", "d"],
      vec!["w", "ah1", "n"],
      vec!["hh", "ah1", "n", "d", "r", "ih0", "d"],
      vec!["b", "uh1", "k", "s"],
      vec![],
    ];
    _test_phonemize(text, phonemes);
  }

  #[test]
  fn test_large_number() {
    // numbers too long will be interpreted one by one
    let text = "The speed of light is 299792458 m/s";
    let phonemes = vec![
      vec!["dh", "ax"],
      vec!["s", "p", "iy1", "d"],
      vec!["ah1", "v"],
      vec!["l", "ay1", "t"],
      vec!["ih1", "z"],
      vec!["t", "uw1"],
      vec!["n", "ay1", "n"],
      vec!["n", "ay1", "n"],
      vec!["s", "eh1", "v", "ax", "n"],
      vec!["n", "ay1", "n"],
      vec!["t", "uw1"],
      vec!["f", "ao1", "r"],
      vec!["f", "ay1", "v"],
      vec!["ey1", "t"],
      vec!["eh1", "m"],
      vec!["eh1", "s"],
    ];
    _test_phonemize(text, phonemes);
  }

  #[test]
  fn test_float_number() {
    let text = "The absolute zero is -273.15 degree Celsius.";
    let phonemes = vec![vec!["dh", "iy1"], vec!["ae1", "b", "s", "ax", "l", "uw2", "t"], vec!["z", "iy1", "r", "ow0"], vec!["ih1", "z"], vec!["m", "ay1", "n", "ax", "s"], vec!["t", "uw1"], vec!["hh", "ah1", "n", "d", "r", "ih0", "d"], vec!["s", "eh1", "v", "ax", "n", "t", "iy0"], vec!["th", "r", "iy1"], vec!["p", "oy1", "n", "t"], vec!["w", "ah1", "n"], vec!["f", "ay1", "v"], vec!["d", "ih0", "g", "r", "iy1"], vec!["s", "eh1", "l", "s", "iy0", "ax", "s"], vec![]];
    _test_phonemize(text, phonemes);
  }

  #[test]
  fn test_phone_number() {
    let text = "My phone number is 123-456-7890.";
    let phonemes = vec![
      vec!["m", "ay1"],
      vec!["f", "ow1", "n"],
      vec!["n", "ah1", "m", "b", "axr"],
      vec!["ih1", "z"],
      vec!["w", "ah1", "n"],
      vec!["t", "uw1"],
      vec!["th", "r", "iy1"],
      vec!["f", "ao1", "r"],
      vec!["f", "ay1", "v"],
      vec!["s", "ih1", "k", "s"],
      vec!["s", "eh1", "v", "ax", "n"],
      vec!["ey1", "t"],
      vec!["n", "ay1", "n"],
      vec!["z", "iy1", "r", "ow0"],
      vec![],
    ];
    _test_phonemize(text, phonemes);
  }
}
