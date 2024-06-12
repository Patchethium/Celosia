pub mod en;
pub mod g2p;

pub fn get_g2p(lang: g2p::constant::LANG) -> g2p::model::G2P {
  let path = match lang {
    g2p::constant::LANG::EN => "./assets/en.bin",
    g2p::constant::LANG::FR => "./assets/fr.bin",
    g2p::constant::LANG::DE => "./assets/de.bin",
  };
  g2p::model::G2P::load(path, g2p::constant::G2PConfig::new(lang)).unwrap()
}

#[cfg(test)]
mod tests {
  // use crate::en::model::{Attention, Embedding, Encoder, GRUCell, Linear, G2P, GRU};
  use crate::en::tagger::PerceptronTagger;
  use crate::en::tokenizer::naive_split;
  use pickle::DeOptions;
  use serde_pickle as pickle;
  use std::collections::{BTreeMap, BTreeSet};
  use std::fs::File;
  use std::io::Read;

  use crate::g2p::model::{G2P, LSTMCell, Linear};

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
  fn test_linear() {
    let weight = vec![
      vec![1., 2., 3.],
      vec![4., 5., 6.],
      vec![7., 8., 9.],
      vec![10., 11., 12.],
    ]; // 4x3
    let bias = vec![1., 2., 3., 4.]; // 4
    let x = vec![1., 2., 3.]; // 3
    let mut xout = vec![0.; 4]; // 4
    let linear = Linear::new(weight, bias);
    linear.forward(&x, &mut xout);
    println!("{:?}", xout);
  }

  #[test]
  fn test_lstm_cell() {
    let tensor = vec![
        vec![1., 2., 3., 4.],
        vec![5., 6., 7., 8.],
        vec![9., 10., 11., 12.],
        vec![13., 14., 15., 16.],
        vec![1., 2., 3., 4.],
        vec![5., 6., 7., 8.],
        vec![9., 10., 11., 12.],
        vec![13., 14., 15., 16.],
        vec![1., 2., 3., 4.],
        vec![5., 6., 7., 8.],
        vec![9., 10., 11., 12.],
        vec![13., 14., 15., 16.],
        vec![1., 2., 3., 4.],
        vec![5., 6., 7., 8.],
        vec![9., 10., 11., 12.],
        vec![13., 14., 15., 16.],
    ];
    let bias = vec![1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.]; // 16
    let x = vec![1., 2., 3., 4.]; // 4

    let h0 = vec![1., 2., 3., 4.]; // 4
    let c0 = vec![1., 2., 3., 4.]; // 4

    let cell = LSTMCell::new(
      Linear::new(tensor.clone(), bias.clone()),
      Linear::new(tensor.clone(), bias.clone()),
    );

    let mut hout = vec![0.; 4];
    let mut cout = vec![0.; 4];

    cell.forward(&x, &h0, &c0, &mut hout, &mut cout);

    let correct_h = vec![0.9640, 0.9951, 0.9993, 0.9999]; // from PyTorch
    let correct_c = vec![2., 3., 4., 5.]; // from PyTorch

    for i in 0..4 {
      assert!((hout[i] - correct_h[i]).abs() < 1e-4); // 1e-4 is epsilon
      assert!((cout[i] - correct_c[i]).abs() < 1e-4);
    }
  }

  #[test]
  fn test_load_g2p() {
    let path = "./assets/en.pickle";
    let _g2p = G2P::load(path, Default::default()).unwrap();
  }

  #[test]
  fn test_g2p_infer() {
    let path = "./assets/en.pickle";
    let g2p = G2P::load(path, Default::default()).unwrap();
    let word = "gutenberg";
    let phoneme = g2p.inference(word).unwrap();
    let answer = vec!["g", "uw1", "t", "ax", "n", "b", "axr", "g"];
    assert_eq!(phoneme, answer);
  }

  #[test]
  fn test_g2p_export() {
    let path = "./assets/en.bin";
    let g2p = G2P::load("./assets/en.pickle", Default::default()).unwrap();
    g2p.export(path).unwrap();
  }
}
