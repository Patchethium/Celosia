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
  fn test_phonemizer() {
    let text = "I read a book by Jane Austen.";
    let phoneme = vec![
      vec!["ay1"],
      vec!["r", "iy1", "d"],
      vec!["ey1"],
      vec!["b", "uh1", "k"],
      vec!["b", "ay1"],
      vec!["jh", "ey1", "n"],
      vec!["ao1", "s", "t", "ih0", "n"],
      vec![],
    ];
    _test_phonemize(text, phoneme);
  }

  #[test]
  fn test_past_tense() {
    let text = "Yesterday I read a book by Jane Austen.";
    let phoneme = vec![
      vec!["y", "eh1", "s", "t", "axr", "d", "ey2"],
      vec!["ay1"],
      vec!["r", "eh1", "d"], // past tense is detected
      vec!["ey1"],
      vec!["b", "uh1", "k"],
      vec!["b", "ay1"],
      vec!["jh", "ey1", "n"],
      vec!["ao1", "s", "t", "ih0", "n"],
      vec![],
    ];
    _test_phonemize(text, phoneme);
  }

  #[test]
  fn test_normal_number() {
    let text = "I read 100 books by Jane Austen.";
    let phoneme = vec![
      vec!["ay1"],
      vec!["r", "iy1", "d"],
      vec!["w", "ah1", "n"],
      vec!["hh", "ah1", "n", "d", "r", "ih0", "d"],
      vec!["b", "uh1", "k", "s"],
      vec!["b", "ay1"],
      vec!["jh", "ey1", "n"],
      vec!["ao1", "s", "t", "ih0", "n"],
      vec![],
    ];
    _test_phonemize(text, phoneme);
  }
  #[test]
  fn test_negative_number() {
    let text = "The absolute zero is -273.15 degree Celsius.";
    let phoneme = vec![
      vec!["dh", "iy1"],
      vec!["ae1", "b", "s", "ax", "l", "uw2", "t"],
      vec!["z", "iy1", "r", "ow0"],
      vec!["ih1", "z"],
      vec!["m", "ay1", "n", "ax", "s"],
      vec!["t", "uw1"],
      vec!["hh", "ah1", "n", "d", "r", "ih0", "d"],
      vec!["s", "eh1", "v", "ax", "n", "t", "iy0"],
      vec!["th", "r", "iy1"],
      vec!["p", "oy1", "n", "t"],
      vec!["w", "ah1", "n"],
      vec!["f", "ay1", "v"],
      vec!["d", "ih0", "g", "r", "iy1"],
      vec!["s", "eh1", "l", "s", "iy0", "ax", "s"],
      vec![],
    ];
    _test_phonemize(text, phoneme);
  }

  #[test]
  fn test_prevowel() {
    let text = "I ate the apple.";
    let phoneme = vec![
      vec!["ay1"],
      vec!["ey1", "t"],
      vec!["dh", "iy1"], // prevowel changes dh ax -> dh iy
      vec!["ae1", "p", "ax", "l"],
      vec![],
    ];
    _test_phonemize(text, phoneme);
  }

  #[test]
  fn test_an() {
    let text = "I am an apple.";
    let phoneme = vec![
      vec!["ay1"],
      vec!["ae1", "m"],
      vec!["ax", "n"], // an -> ax n, it worth testing because `amepd` has some weird entries
      vec!["ae1", "p", "ax", "l"],
      vec![],
    ];
    _test_phonemize(text, phoneme);
  }

  #[test]
  fn test_slash() {
    let text = "I will visit Yoshida-san.";
    let phoneme = vec![
      vec!["ay1"],
      vec!["w", "ih1", "l"],
      vec!["v", "ih1", "z", "ih0", "t"],
      vec!["y", "ow0", "sh", "iy1", "d", "ax"],
      vec![], // the `-` will be ignored
      vec!["s", "ae1", "n"],
      vec![],
    ];
    _test_phonemize(text, phoneme);
  }
}
