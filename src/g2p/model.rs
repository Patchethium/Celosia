use std::io::Write;
use std::{collections::BTreeMap, fs::File};

use rayon::prelude::*;

use crate::g2p::constant::SPECIAL_LEN;

use super::constant::{G2PConfig, D_INTER, D_MODEL, EOS_IDX, MAX_LEN, N_LAYER, PAD_IDX, SOS_IDX};
use half::f16;
use pickle::DeOptions;
use pickle::{HashableValue, Value};
use serde_pickle as pickle;

// if we use const array it will overflow the stack
// vec is slower for its length is unknown at compile time
// and can't be optimized by rustc but it's still usable
// the reason why I don't use ndarray is becuase I'm an idiot
// who wants everything from scratch
// Claim: It's 100 times slower than ndarray
type Tensor1d<T> = Vec<T>;
type Tensor2d<T> = Vec<Vec<T>>;

fn get_shape1d<T>(x: &Vec<T>) -> usize {
  x.len()
}

fn get_shape2d<T>(x: &Vec<Vec<T>>) -> (usize, usize) {
  (x.len(), x[0].len())
}

// utility functions
fn sigmoid(x: &mut f32) {
  *x = 1. / (1. + (-*x).exp());
}

fn softmax(x: &mut Vec<f32>) {
  let max = x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.).clone();
  let sum: f32 = x.par_iter().map(|v| (v - max).exp()).sum();
  x.par_iter_mut().for_each(|v| *v = (*v - max).exp() / sum);
}

fn tanh(x: &mut f32) {
  *x = x.tanh();
}

fn argmax(x: &[f32]) -> usize {
  x.iter()
    .enumerate()
    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
    .map(|(i, _)| i)
    .unwrap_or(EOS_IDX)
}

fn b2v(bytes: &[u8], dim: &usize) -> Tensor1d<f32> {
  let vec: Tensor1d<f32> = bytes
    .par_chunks_exact(2)
    .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
    .collect();
  assert_eq!(bytes.len(), *dim * 2);
  assert_eq!(vec.len(), *dim);
  vec
}

// it calls bytes2tensor1d then fold it into 2d tensor
// it returns a tensor with [dim1, dim2] shape
fn b2m(bytes: &[u8], dim1: &usize, dim2: &usize) -> Tensor2d<f32> {
  let mat: Tensor2d<f32> = bytes
    .par_chunks(dim2 * 2)
    .map(|chunk| {
      chunk
        .par_chunks(2)
        .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
    })
    .collect();

  assert_eq!(bytes.len(), dim1 * dim2 * 2);
  assert_eq!(mat.len(), *dim1);
  assert_eq!(mat[0].len(), *dim2);
  mat
}

pub struct Linear {
  weight: Tensor2d<f32>,
  bias: Tensor1d<f32>,
}

// benchmark: on a 8 core cpu, inference time of a 1024x1024 Linear layer
// is 143 us
impl Linear {
  pub fn new(weight: Tensor2d<f32>, bias: Tensor1d<f32>) -> Self {
    // the weight shape is [d_in, d_out]
    Self { weight, bias }
  }
  // [d_in, d_out].T = [d_out, d_in]
  // [d_out, d_in] @ [d_in] = [d_out] // weight mul x
  // [d_out] + [d_out] = [d_out] // bias add x
  pub fn forward(&self, x: &Tensor1d<f32>, xout: &mut Tensor1d<f32>) {
    xout.par_iter_mut().enumerate().for_each(|(i, v)| {
      *v = self.weight[i]
        .iter()
        .zip(x.iter())
        .fold(self.bias[i], |acc, (&_w, &_x)| acc + _w * _x);
    });
    // xout
    //   .par_iter_mut()
    //   .zip(self.bias.par_iter())
    //   .for_each(|(v, b)| *v += b);
  }
}

pub struct LSTMCell {
  linear_ih: Linear,
  linear_hh: Linear,
}

impl LSTMCell {
  pub fn new(linear_ih: Linear, linear_hh: Linear) -> Self {
    Self {
      linear_ih,
      linear_hh,
    }
  }
  // benchmark: 236us for a 256x256 LSTMCell
  pub fn forward(
    &self,
    x: &Tensor1d<f32>,
    h: &Tensor1d<f32>,
    c: &Tensor1d<f32>,
    hout: &mut Tensor1d<f32>,
    cout: &mut Tensor1d<f32>,
  ) {
    let d_model = h.len();
    let d_inter = d_model * 4;

    let mut ih_buf = vec![0.; d_inter];
    let mut hh_buf = vec![0.; d_inter];
    let mut z_buf = vec![0.; d_inter];
    
    self.linear_ih.forward(x, &mut ih_buf);
    self.linear_hh.forward(h, &mut hh_buf);
    // z = ih + hh
      z_buf
      .par_iter_mut()
      .zip_eq(ih_buf.par_iter())
      .zip_eq(hh_buf.par_iter())
      .for_each(|((z, ih), hh)| *z = ih + hh);

    // i, f, g, o
    // only g needs tanh, others need sigmoid
    z_buf.par_iter_mut().enumerate().for_each(|(i, v)| {
      if 2 * d_model < i && i < 3 * d_model {
        tanh(v);
      } else {
        sigmoid(v);
      }
    });

    cout
      .par_iter_mut()
      .enumerate()
      .zip_eq(hout.par_iter_mut())
      .for_each(|((idx, c_t), h_t)| {
        let _i = z_buf[idx];
        let _f = z_buf[idx + d_model];
        let _g = z_buf[idx + d_model * 2];
        let _o = z_buf[idx + d_model * 3];

        *c_t = _f * c[idx] + _i * _g;
        *h_t = _o * c_t.tanh();
      });
  }
}

pub struct Embedding {
  weight: Tensor2d<f32>,
}

impl Embedding {
  // weight: [d_emb, d_model]
  pub fn new(weight: Tensor2d<f32>) -> Self {
    Self { weight }
  }

  // x: [seq_len], xout: [seq_len, d_model]
  pub fn forward(&self, x: &Tensor1d<usize>, xout: &mut Tensor2d<f32>) {
    xout.par_iter_mut().enumerate().for_each(|(i, v)| {
      v.copy_from_slice(&self.weight[x[i]]);
    });
  }
}

pub struct G2P {
  config: G2PConfig,
  enc_emb: Embedding,
  enc_lstm: [LSTMCell; N_LAYER],
  dec_emb: Embedding,
  dec_lstm: [LSTMCell; N_LAYER],
  post: Linear,
}

impl G2P {
  pub fn new(
    config: G2PConfig,
    enc_emb: Embedding,
    enc_lstm: [LSTMCell; N_LAYER],
    dec_emb: Embedding,
    dec_lstm: [LSTMCell; N_LAYER],
    post: Linear,
  ) -> Self {
    Self {
      config,
      enc_emb,
      enc_lstm,
      dec_emb,
      dec_lstm,
      post,
    }
  }

  pub fn forward(&self, src: &Tensor1d<usize>) -> Tensor1d<usize> {
    let d_model = D_MODEL;
    let mut h: Tensor2d<f32> = vec![vec![0.; d_model]; N_LAYER];
    let mut c: Tensor2d<f32> = vec![vec![0.; d_model]; N_LAYER];
    let mut hout: Tensor2d<f32> = vec![vec![0.; d_model]; N_LAYER];
    let mut cout: Tensor2d<f32> = vec![vec![0.; d_model]; N_LAYER];

    let mut tgt = vec![SOS_IDX; 1];

    let mut enc_emb = vec![vec![0.; d_model]; src.len()];
    self.enc_emb.forward(src, &mut enc_emb);
    for (i, lstm) in self.enc_lstm.iter().enumerate() {
      for x in enc_emb.iter_mut() {
        lstm.forward(x, &h[i], &c[i], &mut hout[i], &mut cout[i]);
        h[i].copy_from_slice(&hout[i]);
        c[i].copy_from_slice(&cout[i]);
        x.copy_from_slice(&hout[i]); // reuse the memory
      }
    }
    let mut prev = SOS_IDX;
    let mut emb_buf = vec![vec![0.; d_model]; 1];
    let mut logit_buf = vec![0.; self.config.d_phoneme];

    while prev != EOS_IDX && prev != PAD_IDX && tgt.len() < MAX_LEN {
      self.dec_emb.forward(&vec![prev], &mut emb_buf);
      for (i, lstm) in self.dec_lstm.iter().enumerate() {
        // emb_buf: [1, d_model], use [0] as squeeze(0)
        lstm.forward(&emb_buf[0], &h[i], &c[i], &mut hout[i], &mut cout[i]);
        h[i].copy_from_slice(&hout[i]);
        c[i].copy_from_slice(&cout[i]);
        emb_buf[0].copy_from_slice(&hout[i]);
      }
      self.post.forward(&h[N_LAYER-1], &mut logit_buf);
      softmax(&mut logit_buf);
      prev = argmax(&logit_buf);
      tgt.push(prev);
    }
    tgt
  }

  /*
    enc_emb.weight
    enc_rnn.weight_ih_l0
    enc_rnn.weight_hh_l0
    enc_rnn.bias_ih_l0
    enc_rnn.bias_hh_l0
    enc_rnn.weight_ih_l1
    enc_rnn.weight_hh_l1
    enc_rnn.bias_ih_l1
    enc_rnn.bias_hh_l1
    dec_emb.weight
    dec_rnn.weight_ih_l0
    dec_rnn.weight_hh_l0
    dec_rnn.bias_ih_l0
    dec_rnn.bias_hh_l0
    dec_rnn.weight_ih_l1
    dec_rnn.weight_hh_l1
    dec_rnn.bias_ih_l1
    dec_rnn.bias_hh_l1
    post.weight
    post.bias
  */
  /// loading this takes 10ms on a 8 core cpu
  pub fn load(path: &str, config: G2PConfig) -> Result<Self, anyhow::Error> {
    let file = Box::new(File::open(path)?);
    let decoded = pickle::value_from_reader(file, DeOptions::new().decode_strings())?;
    let mut wmap = BTreeMap::new();
    if let Value::Dict(_wmap) = decoded {
      for (k, v) in _wmap {
        if let Value::Bytes(b) = v {
          if let HashableValue::String(s) = k {
            wmap.insert(s, b);
          } else {
            return Err(anyhow::anyhow!("invalid pickle file"));
          }
        } else {
          return Err(anyhow::anyhow!("invalid pickle file"));
        }
      }
    } else {
      return Err(anyhow::anyhow!("invalid pickle file"));
    };

    // enc_emb
    let enc_emb = Embedding::new(b2m(&wmap["enc_emb.weight"], &config.d_alphabet, &D_MODEL));
    let dec_emb = Embedding::new(b2m(&wmap["dec_emb.weight"], &config.d_phoneme, &D_MODEL));

    // enc_rnn
    let enc_rnn_weight_ih_l0 = b2m(&wmap["enc_rnn.weight_ih_l0"], &D_INTER, &D_MODEL);
    let enc_rnn_weight_hh_l0 = b2m(&wmap["enc_rnn.weight_hh_l0"], &D_INTER, &D_MODEL);
    let enc_rnn_bias_ih_l0 = b2v(&wmap["enc_rnn.bias_ih_l0"], &D_INTER);
    let enc_rnn_bias_hh_l0 = b2v(&wmap["enc_rnn.bias_hh_l0"], &D_INTER);
    let enc_rnn_weight_ih_l1 = b2m(&wmap["enc_rnn.weight_ih_l1"], &D_INTER, &D_MODEL);
    let enc_rnn_weight_hh_l1 = b2m(&wmap["enc_rnn.weight_hh_l1"], &D_INTER, &D_MODEL);
    let enc_rnn_bias_ih_l1 = b2v(&wmap["enc_rnn.bias_ih_l1"], &D_INTER);
    let enc_rnn_bias_hh_l1 = b2v(&wmap["enc_rnn.bias_hh_l1"], &D_INTER);

    let enc_lstm = [
      LSTMCell::new(
        Linear::new(enc_rnn_weight_ih_l0, enc_rnn_bias_ih_l0),
        Linear::new(enc_rnn_weight_hh_l0, enc_rnn_bias_hh_l0),
      ),
      LSTMCell::new(
        Linear::new(enc_rnn_weight_ih_l1, enc_rnn_bias_ih_l1),
        Linear::new(enc_rnn_weight_hh_l1, enc_rnn_bias_hh_l1),
      ),
    ];

    // dec_rnn
    let dec_rnn_weight_ih_l0 = b2m(&wmap["dec_rnn.weight_ih_l0"], &D_INTER, &D_MODEL);
    let dec_rnn_weight_hh_l0 = b2m(&wmap["dec_rnn.weight_hh_l0"], &D_INTER, &D_MODEL);
    let dec_rnn_bias_ih_l0 = b2v(&wmap["dec_rnn.bias_ih_l0"], &D_INTER);
    let dec_rnn_bias_hh_l0 = b2v(&wmap["dec_rnn.bias_hh_l0"], &D_INTER);
    let dec_rnn_weight_ih_l1 = b2m(&wmap["dec_rnn.weight_ih_l1"], &D_INTER, &D_MODEL);
    let dec_rnn_weight_hh_l1 = b2m(&wmap["dec_rnn.weight_hh_l1"], &D_INTER, &D_MODEL);
    let dec_rnn_bias_ih_l1 = b2v(&wmap["dec_rnn.bias_ih_l1"], &D_INTER);
    let dec_rnn_bias_hh_l1 = b2v(&wmap["dec_rnn.bias_hh_l1"], &D_INTER);

    let dec_lstm = [
      LSTMCell::new(
        Linear::new(dec_rnn_weight_ih_l0, dec_rnn_bias_ih_l0),
        Linear::new(dec_rnn_weight_hh_l0, dec_rnn_bias_hh_l0),
      ),
      LSTMCell::new(
        Linear::new(dec_rnn_weight_ih_l1, dec_rnn_bias_ih_l1),
        Linear::new(dec_rnn_weight_hh_l1, dec_rnn_bias_hh_l1),
      ),
    ];

    // post
    let post = Linear::new(
      b2m(&wmap["post.weight"], &config.d_phoneme, &D_MODEL),
      b2v(&wmap["post.bias"], &config.d_phoneme),
    );

    // all set
    Ok(Self::new(
      config, enc_emb, enc_lstm, dec_emb, dec_lstm, post,
    ))
  }

  // inference takes 12 ms for word "gutenberg" on a 8 core cpu
  pub fn inference(&self, word: &str) -> anyhow::Result<Vec<&str>> {
    let mut incidies = word
      .par_chars()
      .map(|c| {
        self
          .config
          .alphabet
          .get_by_left(&c)
          .ok_or_else(|| anyhow::anyhow!("invalid char"))
          .map(|index| *index)
      })
      .collect::<Result<Vec<usize>, anyhow::Error>>()?;
    // attach sos and eos token
    incidies.par_iter_mut().for_each(|i| *i += SPECIAL_LEN);
    incidies.insert(0, SOS_IDX);
    incidies.push(EOS_IDX);
    let mut output = self.forward(&incidies);
    let len = output.len();
    // get rid of the sos and eos token
    let output = &mut output[1..len - 1];
    // offset
    output.par_iter_mut().for_each(|i| *i -= SPECIAL_LEN);
    let phoneme: Vec<&str> = output
      .iter()
      .map(|i| *self.config.phoneme.get_by_right(&i).unwrap_or(&""))
      .collect();
    Ok(phoneme)
  }

  // test use, export the weights for examination
  // should not be accessible by the user
  pub(crate) fn export(&self, path: &str) -> anyhow::Result<()> {
    // export the wanted weights so that I can see if it's correct
    // I'll hard code the key for now
    let mut file = File::create(path)?;
    let weight = &self.enc_lstm[0].linear_hh.weight;
    for v in weight.iter() {
      for i in v.iter() {
        file.write(&i.to_le_bytes())?;
      }
    }
    Ok(())
  }
}
