pub mod en;

#[cfg(test)]
mod tests {
  use crate::en::rnn::{GRU, Linear, GRUCell};
  use crate::en::tagger::PerceptronTagger;
  use crate::en::tokenizer::naive_split;
  use ndarray::{arr2, Array, Array1, Array2};
  use ndarray_rand::rand_distr::Uniform;
  use ndarray_rand::RandomExt;
  use pickle::DeOptions;
  use serde_pickle as pickle;
  use std::collections::{BTreeMap, BTreeSet};
  use std::fs::File;
  use std::io::Read;

  const TEXT: &str = "Printing , in the only sense with which we are at present concerned ,
  differs from most if not from all the arts (and crafts)
  represented in the Exhibition in being comparatively modern .";

  #[test]
  fn test_tagger() {
    const PATH: &str = "./data/averaged_perceptron_tagger.pickle";
    let tagger = PerceptronTagger::new(PATH).unwrap();

    let v = naive_split(TEXT).unwrap();
    let tokens: Vec<String> = v.iter().map(|s| s.to_string()).collect();

    let res = tagger.tag(tokens, false, true);
    let tags: Vec<&str> = res.iter().map(|t| &t.1[..]).collect();

    assert_eq!(
      tags,
      vec![
        "NN", ",", "IN", "DT", "JJ", "NN", "IN", "WDT", "PRP", "VBP", "IN", "JJ", "JJ", ",", "NNS",
        "IN", "JJS", "IN", "RB", "IN", "PDT", "DT", "NNS", "(", "CC", "NNS", ")", "VBN", "IN",
        "DT", "NN", "IN", "VBG", "RB", "JJ", "."
      ]
    );
  }

  #[test]
  fn test_pickle() {
    let reader: Box<dyn Read> =
      Box::new(File::open("./data/averaged_perceptron_tagger.pickle").unwrap());
    let decoded = pickle::value_from_reader(reader, DeOptions::new().decode_strings()).unwrap();
    let _ = serde_pickle::from_value::<(
      BTreeMap<String, BTreeMap<String, f64>>,
      BTreeMap<String, String>,
      BTreeSet<String>,
    )>(decoded)
    .unwrap();
  }

  #[test]
  fn test_tokenizer() {
    let splitted = naive_split(TEXT.into()).unwrap();
    println!("{:#?}", splitted);
  }

  #[test]
  fn test_gru() {
    let h_dim = 32;
    let seq_dim = 126;
    let w_ih: Array2<f32> = Array::random((3 * h_dim, h_dim), Uniform::new(0., 10.));
    let w_hh: Array2<f32> = Array::random((3 * h_dim, h_dim), Uniform::new(0., 10.));

    let b_ih: Array1<f32> = Array::random((3 * h_dim,), Uniform::new(0., 10.));
    let b_hh: Array1<f32> = Array::random((3 * h_dim,), Uniform::new(0., 10.));

    let linear_ih = Linear::new(w_ih, b_ih);
    let linear_hh = Linear::new(w_hh, b_hh);

    let gru_cell = GRUCell::new(linear_ih, linear_hh);
    
    let gru = GRU::new(gru_cell, true);

    let x: Array2<f32> = Array::random((seq_dim, h_dim), Uniform::new(0., 10.));

    println!("{:?}", gru.forward(&x).shape());
  }
}
