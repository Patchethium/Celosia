pub mod en;

#[cfg(test)]
mod tests {
  use crate::en::constants::AMEPD_PHONE_SET;
  use crate::en::model::{Attention, Decoder, Embedding, Encoder, GRUCell, Linear, G2P, GRU};
  use crate::en::tagger::PerceptronTagger;
  use crate::en::tokenizer::naive_split;
  use ndarray::{Array, Array1, Array2};
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
    let dist = Uniform::new(0., 10.);

    let w_ih: Array2<f32> = Array::random((3 * h_dim, h_dim), dist);
    let w_hh: Array2<f32> = Array::random((3 * h_dim, h_dim), dist);

    let b_ih: Array1<f32> = Array::random((3 * h_dim,), dist);
    let b_hh: Array1<f32> = Array::random((3 * h_dim,), dist);

    let linear_ih = Linear::new(w_ih, b_ih);
    let linear_hh = Linear::new(w_hh, b_hh);

    let gru_cell = GRUCell::new(linear_ih, linear_hh);

    let gru = GRU::new(gru_cell, true);

    let h = Array::zeros((h_dim,));

    let x: Array2<f32> = Array::random((seq_dim, h_dim), dist);

    println!("{:?}", gru.forward(&x, &h).0.shape());
  }

  #[test]
  fn test_attn() {
    let h_dim = 32;
    let seq_dim = 126;
    let dist = Uniform::new(0., 10.);
    let w_k: Array2<f32> = Array::random((2 * h_dim, h_dim), dist);
    let w_v: Array2<f32> = Array::random((h_dim, 2 * h_dim), dist);

    let b_k: Array1<f32> = Array::random((2 * h_dim,), dist);
    let b_v: Array1<f32> = Array::random((h_dim,), dist);

    let attn = Attention::new(Linear::new(w_k, b_k), Linear::new(w_v, b_v));

    let enc_o = Array::random((seq_dim, 2 * h_dim), dist);
    let dec_o = Array::random((h_dim,), dist);

    println!("{:?}", attn.forward(&enc_o, &dec_o).shape())
  }

  fn get_gru_cell(i_dim: usize, h_dim: usize) -> GRUCell {
    let w_ih: Array2<f32> = Array::random((3 * h_dim, i_dim), Uniform::new(0., 10.));
    let w_hh: Array2<f32> = Array::random((3 * h_dim, h_dim), Uniform::new(0., 10.));

    let b_ih: Array1<f32> = Array::random((3 * h_dim,), Uniform::new(0., 10.));
    let b_hh: Array1<f32> = Array::random((3 * h_dim,), Uniform::new(0., 10.));

    let linear_ih = Linear::new(w_ih, b_ih);
    let linear_hh = Linear::new(w_hh, b_hh);

    let gru_cell = GRUCell::new(linear_ih, linear_hh);

    gru_cell
  }

  fn get_gru(i_dim: usize, h_dim: usize) -> GRU {
    let gru_cell = get_gru_cell(i_dim, h_dim);

    GRU::new(gru_cell, true)
  }

  fn get_linear(i_dim: usize, o_dim: usize) -> Linear {
    let weight: Array2<f32> = Array::random((o_dim, i_dim), Uniform::new(0., 10.));
    let bias: Array1<f32> = Array::random((o_dim,), Uniform::new(0., 10.));
    Linear::new(weight, bias)
  }

  #[test]
  fn test_encoder() {
    let h_dim = 128;
    let seq_dim = 12;
    let dist = Uniform::new(0., 10.);
    let dist_usize = Uniform::<usize>::new(0, 10);

    let gru = get_gru(h_dim, h_dim);
    let gru_rev = get_gru(h_dim, h_dim);

    let emb_weight = Array::random((10, h_dim), dist);
    let emb = Embedding::new(emb_weight);

    let mut data = Array::random((seq_dim,), dist_usize);

    let encoder = Encoder::new(emb, gru, gru_rev, get_linear(2 * h_dim, h_dim));
    let (opt, h) = encoder.forward(&mut data).unwrap();
    println!("{:?}, {:?}", opt.shape(), h.shape());
  }
  #[test]
  fn test_g2p() {
    let path = "./data/en_rnn.bin";
    let g2p = G2P::from_file(path).unwrap();
    let indices: Array1<usize> = Array1::from_vec(vec![1, 7, 4, 12, 22, 28, 2]);
    let output = g2p.forward(&indices).unwrap();
    let phoneme: Vec<&str> = output
      .iter()
      .map(|i| AMEPD_PHONE_SET[i.clone() as usize])
      .collect();
    println!("{:?}", phoneme);

    // println!("{:?}", g2p.attn.k.weight);
  }
}
