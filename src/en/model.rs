use crate::en::constants::*;
use ndarray::{array, concatenate, s, Array1, Array2, ArrayView1, Axis, Dim, Dimension};
use ndarray_stats::QuantileExt;
use std::fs::File;
use std::io::Read;

/// rnn model for oov prediction
pub struct GRUCell {
  pub(crate) linear_ih: Linear,
  pub(crate) linear_hh: Linear,
}

pub struct Linear {
  pub(crate) weight: Array2<f32>,
  pub(crate) bias: Array1<f32>,
}

impl Linear {
  pub fn new(weight: Array2<f32>, bias: Array1<f32>) -> Linear {
    Linear { weight, bias }
  }
  fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
    x.dot(&self.weight.t()) + &self.bias
  }
}

fn sigmoid(x: &Array1<f32>) -> Array1<f32> {
  1.0 / (1.0 + (-x).map(|v| v.exp()))
}

fn tanh(x: &Array1<f32>) -> Array1<f32> {
  x.map(|v| v.tanh())
}

fn softmax(arr: &Array1<f32>) -> Array1<f32> {
  let exp_arr = arr.mapv(f32::exp);
  let sum_exp: f32 = exp_arr.sum();
  let softmax_probs = &exp_arr / sum_exp;

  softmax_probs
}

impl GRUCell {
  pub fn new(linear_ih: Linear, linear_hh: Linear) -> GRUCell {
    GRUCell {
      linear_ih,
      linear_hh,
    }
  }

  fn forward(&self, x: &Array1<f32>, h: &Array1<f32>) -> Array1<f32> {
    let rzn_ih: Array1<f32> = self.linear_ih.forward(x);
    let rzn_hh: Array1<f32> = self.linear_hh.forward(h);

    let rzn_ih_len: usize = rzn_ih.shape()[0] * 2 / 3;
    let rzn_hh_len: usize = rzn_hh.shape()[0] * 2 / 3;

    let rz_ih: ArrayView1<f32> = rzn_ih.slice(s![0..rzn_ih_len]);
    let n_ih: ArrayView1<f32> = rzn_ih.slice(s![rzn_ih_len..]);

    let rz_hh: ArrayView1<f32> = rzn_hh.slice(s![0..rzn_ih_len]);
    let n_hh: ArrayView1<f32> = rzn_hh.slice(s![rzn_hh_len..]);

    let rz: Array1<f32> = sigmoid(&(&rz_ih + &rz_hh));

    let r_len: usize = rz_ih.shape()[0] / 2;

    let r: ArrayView1<f32> = rz.slice(s![0..r_len]);
    let z: ArrayView1<f32> = rz.slice(s![r_len..]);

    let n: Array1<f32> = tanh(&(&n_ih + &r * &n_hh));

    let updated_h: Array1<f32> = (1.0 - &z) * &n + &z * h;

    updated_h
  }
}

pub struct GRU {
  pub(crate) cell: GRUCell,
  pub(crate) reversed: bool,
}

impl GRU {
  pub fn new(cell: GRUCell, reversed: bool) -> GRU {
    GRU { cell, reversed }
  }
  pub fn forward(&self, x: &Array2<f32>, h: &Array1<f32>) -> (Array2<f32>, Array1<f32>) {
    let seq_len = x.shape()[0];
    let h_dim = x.shape()[1];
    let mut h = h.clone();
    let mut output = Array2::zeros((seq_len, h_dim));

    if self.reversed {
      for t in (0..seq_len).rev() {
        h = self.cell.forward(&x.index_axis(Axis(0), t).to_owned(), &h);
        output.index_axis_mut(Axis(0), t).assign(&h);
      }
    } else {
      for t in 0..seq_len {
        h = self.cell.forward(&x.index_axis(Axis(0), t).to_owned(), &h);
        output.index_axis_mut(Axis(0), t).assign(&h);
      }
    }

    (output, h)
  }
}
#[derive(Clone)]
pub struct Embedding {
  pub(crate) weight: Array2<f32>,
  pub(crate) embedding_dim: usize,
}

impl Embedding {
  pub fn new(weight: Array2<f32>) -> Embedding {
    let embedding_dim = weight.shape()[1];
    Embedding {
      weight,
      embedding_dim,
    }
  }

  pub fn forward(&self, x: &Array1<usize>) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let mut embedding: Array2<f32> = Array2::zeros((seq_len, self.embedding_dim));
    // embedding layer is basically assigning weight for input index
    for (i, &index) in x.iter().enumerate() {
      embedding
        .slice_mut(s![i, ..])
        .assign(&self.weight.slice(s![index, ..]));
    }
    embedding
  }
}

pub struct Encoder {
  pub(crate) embedding: Embedding,
  pub(crate) gru: GRU,
  pub(crate) gru_rev: GRU,
  pub(crate) fc: Linear,
}

impl Encoder {
  pub fn new(embedding: Embedding, gru: GRU, gru_rev: GRU, fc: Linear) -> Encoder {
    Encoder {
      embedding,
      gru,
      gru_rev,
      fc,
    }
  }

  pub fn forward(&self, x: &Array1<usize>) -> anyhow::Result<(Array2<f32>, Array1<f32>)> {
    let embedded = self.embedding.forward(x);
    let h_0 = Array1::zeros((embedded.shape()[1],)); // [N]
    let (output, h) = self.gru.forward(&embedded, &h_0);
    let (output_rev, h_rev) = self.gru_rev.forward(&embedded, &h_0);
    let out_final = concatenate(Axis(1), &[output.view(), output_rev.view()])?;
    let h_final = tanh(
      &self
        .fc
        .forward(&concatenate(Axis(0), &[h.view(), h_rev.view()])?),
    );
    Ok((out_final, h_final))
  }
}

pub struct Attention {
  pub(crate) k: Linear,
  pub(crate) v: Linear,
  pub(crate) attn_scale: f32,
}

impl Attention {
  pub fn new(k: Linear, v: Linear) -> Attention {
    let attn_scale = f32::sqrt(k.bias.shape()[0] as f32);
    Attention { k, v, attn_scale }
  }

  pub fn forward(&self, enc_o: &Array2<f32>, dec_o: &Array1<f32>) -> Array1<f32> {
    // dec_o: [N]
    // enc_o: [S,N]
    let dec_o = self.k.forward(&dec_o); // [2N]
    let expanded_dec_o = dec_o.insert_axis(Axis(1));
    let enc_o = enc_o.view();
    let attn_score: Array1<f32> =
      softmax(&(enc_o.dot(&expanded_dec_o) / self.attn_scale).remove_axis(Axis(1))); // [S, 1]
    let attn_score = attn_score.insert_axis(Axis(1));
    let output = self.v.forward(&(&attn_score * &enc_o).sum_axis(Axis(0)));

    output // [N]
  }
}

pub struct Decoder {
  pub(crate) embedding: Embedding,
  pub(crate) gru_cell: GRUCell,
  pub(crate) fc: Linear,
}

impl Decoder {
  pub fn new(embedding: Embedding, gru_cell: GRUCell, fc: Linear) -> Decoder {
    Decoder {
      embedding,
      gru_cell,
      fc,
    }
  }

  pub fn forward(
    &self,
    attn_o: &Array1<f32>,
    dec_idx: usize,
    h: &Array1<f32>,
  ) -> anyhow::Result<(usize, Array1<f32>)> {
    let dec_emb: Array1<f32> = self
      .embedding
      .forward(&array![dec_idx])
      .remove_axis(Axis(0)); // [N]
    let rnn_input:Array1<f32> = concatenate(Axis(0), &[attn_o.view(), dec_emb.view()])?;
    let h: Array1<f32> = self.gru_cell.forward(&rnn_input, &h); // dec_o:[1,N], h:[N]
    let dec_fc_o: usize = self.fc.forward(&h).argmax()?;
    Ok((dec_fc_o, h))
  }
}

pub struct G2P {
  pub(crate) encoder: Encoder,
  pub(crate) attn: Attention,
  pub(crate) decoder: Decoder,
  sos_idx: usize,
  eos_idx: usize,
}

struct Counter {
  count: usize,
}

impl Counter {
  pub fn new(count: usize) -> Counter {
    Counter { count }
  }
  pub fn next(&mut self) -> usize {
    let count = self.count.clone();
    self.count += 1;
    count
  }
}

impl G2P {
  pub fn new(
    encoder: Encoder,
    attn: Attention,
    decoder: Decoder,
    sos_idx: usize,
    eos_idx: usize,
  ) -> G2P {
    G2P {
      encoder,
      attn,
      decoder,
      sos_idx,
      eos_idx,
    }
  }

  pub fn from_file(file: &str) -> anyhow::Result<G2P> {
    let mut f = File::open(file)?;
    let mut weights: Vec<Array2<f32>> = Vec::new();
    for shape in G2P_MODEL_SHAPES {
      let weight = G2P::read_with_shape_from_buffer(&mut f, &Dim(shape))?;
      weights.push(weight);
    }
    G2P::from_weights(weights)
  }

  fn read_with_shape_from_buffer(
    f: &mut File,
    shape: &Dim<[usize; 2]>,
  ) -> anyhow::Result<Array2<f32>> {
    let mut buf = [0u8; 4];
    let length: usize = shape[0] * shape[1];
    let mut arr: Vec<f32> = Vec::new();
    for _ in 0..length {
      f.read_exact(&mut buf)?;
      let float_value = f32::from_le_bytes(buf);
      arr.push(float_value);
    }
    Ok(Array2::<f32>::from_shape_vec(shape.clone(), arr)?)
  }

  pub fn forward(&self, alphabet_indices: &Array1<usize>) -> anyhow::Result<Vec<usize>> {
    let tgt_len = alphabet_indices.shape()[0] as usize;
    let (enc_o, mut h) = self.encoder.forward(alphabet_indices)?;

    let mut phoneme = Vec::<usize>::new();

    let mut dec_idx: usize = self.sos_idx;
    phoneme.push(dec_idx.clone());
    let t: usize = 0;

    let mut attn_o = self.attn.forward(&enc_o, &h);

    while t < tgt_len && dec_idx != self.eos_idx {
      (dec_idx, h) = self.decoder.forward(&attn_o, dec_idx, &h)?;
      phoneme.push(dec_idx);
      attn_o = self.attn.forward(&enc_o, &h);
    }
    Ok(phoneme)
  }

  fn from_weights(weights: Vec<Array2<f32>>) -> anyhow::Result<G2P> {
    anyhow::ensure!(weights.len() == G2P_MODEL_SHAPES.len());
    let mut c = Counter::new(0);
    let enc_emb_weight: Array2<f32> = weights[c.next()].to_owned();
    let enc_rnn_weight_ih: Array2<f32> = weights[c.next()].to_owned();
    let enc_rnn_weight_hh: Array2<f32> = weights[c.next()].to_owned();
    let enc_rnn_bias_ih: Array1<f32> = weights[c.next()].to_owned().remove_axis(Axis(0));
    let enc_rnn_bias_hh: Array1<f32> = weights[c.next()].to_owned().remove_axis(Axis(0));

    let enc_emb = Embedding::new(enc_emb_weight);
    let enc_gru = GRU::new(
      GRUCell::new(
        Linear::new(enc_rnn_weight_ih, enc_rnn_bias_ih),
        Linear::new(enc_rnn_weight_hh, enc_rnn_bias_hh),
      ),
      false,
    );

    let enc_rnn_weight_ih_rev: Array2<f32> = weights[c.next()].to_owned();
    let enc_rnn_weight_hh_rev: Array2<f32> = weights[c.next()].to_owned();
    let enc_rnn_bias_ih_rev: Array1<f32> = weights[c.next()].to_owned().remove_axis(Axis(0));
    let enc_rnn_bias_hh_rev: Array1<f32> = weights[c.next()].to_owned().remove_axis(Axis(0));

    let enc_gru_rev = GRU::new(
      GRUCell::new(
        Linear::new(enc_rnn_weight_ih_rev, enc_rnn_bias_ih_rev),
        Linear::new(enc_rnn_weight_hh_rev, enc_rnn_bias_hh_rev),
      ),
      true,
    );

    let enc_fc_weight: Array2<f32> = weights[c.next()].to_owned();
    let enc_fc_bias: Array1<f32> = weights[c.next()].to_owned().remove_axis(Axis(0));

    let enc_fc = Linear::new(enc_fc_weight, enc_fc_bias);

    let encoder = Encoder::new(enc_emb, enc_gru, enc_gru_rev, enc_fc);

    let attn_k_weight: Array2<f32> = weights[c.next()].to_owned();
    let attn_k_bias: Array1<f32> = weights[c.next()].to_owned().remove_axis(Axis(0));

    let attn_v_weight: Array2<f32> = weights[c.next()].to_owned();
    let attn_v_bias: Array1<f32> = weights[c.next()].to_owned().remove_axis(Axis(0));

    let attn = Attention::new(
      Linear::new(attn_k_weight, attn_k_bias),
      Linear::new(attn_v_weight, attn_v_bias),
    );

    let dec_emb_weight: Array2<f32> = weights[c.next()].to_owned();
    let dec_rnn_weight_ih: Array2<f32> = weights[c.next()].to_owned();
    let dec_rnn_weight_hh: Array2<f32> = weights[c.next()].to_owned();
    let dec_rnn_bias_ih: Array1<f32> = weights[c.next()].to_owned().remove_axis(Axis(0));
    let dec_rnn_bias_hh: Array1<f32> = weights[c.next()].to_owned().remove_axis(Axis(0));

    let dec_emb = Embedding::new(dec_emb_weight);
    let dec_gru = GRUCell::new(
      Linear::new(dec_rnn_weight_ih, dec_rnn_bias_ih),
      Linear::new(dec_rnn_weight_hh, dec_rnn_bias_hh),
    );

    let dec_fc_weight: Array2<f32> = weights[c.next()].to_owned();
    let dec_fc_bias: Array1<f32> = weights[c.next()].to_owned().remove_axis(Axis(0));

    let dec_fc = Linear::new(dec_fc_weight, dec_fc_bias);

    let decoder = Decoder::new(dec_emb, dec_gru, dec_fc);

    let g2p = G2P::new(
      encoder,
      attn,
      decoder,
      AMEPD_PHONE_SET.iter().position(|s| s == &"<sos>").unwrap(),
      AMEPD_PHONE_SET.iter().position(|s| s == &"<eos>").unwrap(),
    );

    Ok(g2p)
  }
}