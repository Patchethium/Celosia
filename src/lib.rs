mod en;

#[cfg(test)]
mod tests {
  use crate::en::tagger::PerceptronTagger;
  use std::collections::{BTreeMap, BTreeSet};
  use std::fs::File;
  use std::io::Read;
  use pickle::DeOptions;
  use serde_pickle as pickle;


  #[test]
  fn test_tagger() {
    const PATH: &str = "./data/averaged_perceptron_tagger.pickle";

    let tagger = PerceptronTagger::new(PATH).unwrap();

    let v = "I read that book yesterday";

    let tokens: Vec<String> = v.split_whitespace().map(|s| s.to_string()).collect();

    let res = tagger.tag(tokens, false, true);

    let tags: Vec<&str> = res.iter().map(|t| &t.1[..]).collect();

    assert_eq!(tags, vec!["PRP", "VBP", "IN", "NN", "NN"]);
  }

  #[test]
  fn test_pickle() {
    let reader: Box<dyn Read> = Box::new(File::open("./data/averaged_perceptron_tagger.pickle").unwrap());
    let decoded = pickle::value_from_reader(reader, DeOptions::new().decode_strings()).unwrap();
    let _ = serde_pickle::from_value::<(
      BTreeMap<String, BTreeMap<String, f64>>,
      BTreeMap<String, String>,
      BTreeSet<String>,
    )>(decoded)
    .unwrap();
  }
}
