use core::f32;

use crate::g2p::constant::{EOS_IDX, MAX_LEN, SOS_IDX};
use itertools::izip;
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use serde::{Deserialize, Serialize};

fn argmax(a: ArrayView1<f32>) -> usize {
  let mut max_idx = 0;
  let mut max_val = a[0];
  for (i, &val) in a.iter().enumerate() {
    if val > max_val {
      max_val = val;
      max_idx = i;
    }
  }
  max_idx
}
#[derive(Serialize, Deserialize)]
pub struct Linear {
  pub weight: Array2<f32>,
  pub bias: Array1<f32>,
}

impl Linear {
  pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
    x.dot(&self.weight.t()) + &self.bias
  }
}

#[derive(Serialize, Deserialize)]
pub struct Embedding {
  pub weight: Array2<f32>, // [vocab_size, d_model]
}

impl Embedding {
  pub fn forward(&self, idx: ArrayView1<usize>) -> Array2<f32> {
    self.weight.select(Axis(0), idx.as_slice().unwrap())
  }
}
#[derive(Serialize, Deserialize)]
pub struct LayerNorm {
  pub weight: Array1<f32>, //[d_model]
  pub bias: Array1<f32>,   //[d_model]
}

impl LayerNorm {
  pub fn forward_inplace(&self, x: &mut Array2<f32>) {
    // x: (seq, d_model)
    let mean = x.mean_axis(Axis(1)).unwrap().insert_axis(Axis(1)); // (seq, 1)
    let var = x.var_axis(Axis(1), 0.0).insert_axis(Axis(1)); // (seq, 1)
    let eps = 1.0e-5;

    // Broadcast operations
    *x -= &mean;
    *x /= &(var + eps).mapv(f32::sqrt);

    // Element-wise multiplication and addition
    *x *= &self.weight.view().insert_axis(Axis(0));
    *x += &self.bias.view().insert_axis(Axis(0));
  }
}

#[derive(Serialize, Deserialize)]
pub struct MHSA {
  pub d_model: usize,
  pub n_head: usize,
  pub d_head: usize,
  pub scale: f32,
  pub wq: Linear,
  pub wk: Linear,
  pub wv: Linear,
  pub wo: Linear,
}

impl MHSA {
  pub fn new(wq: Linear, wk: Linear, wv: Linear, wo: Linear, n_head: usize) -> Self {
    let d_model = wq.weight.shape()[1];
    let d_head = d_model / n_head;
    let scale = (d_head as f32).sqrt();
    MHSA {
      d_model,
      n_head,
      d_head,
      scale,
      wq,
      wk,
      wv,
      wo,
    }
  }

  // q, k, v: (seq_len, d_model)
  // beta: (n_head, seq_len_q, seq_len_k)
  pub fn forward(
    &self,
    q: ArrayView2<f32>,
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    beta: ArrayView3<f32>, // [n_head, seq_len_q, seq_len_k]
  ) -> Array2<f32> {
    let seq_len_q = q.shape()[0];
    let seq_len_k = k.shape()[0];

    let q = self.wq.forward(q);
    let k = self.wk.forward(k);
    let v = self.wv.forward(v);

    let q_reshaped = q.to_shape([seq_len_q, self.n_head, self.d_head]).unwrap();
    let k_reshaped = k.to_shape([seq_len_k, self.n_head, self.d_head]).unwrap();
    let v_reshaped = v.to_shape([seq_len_k, self.n_head, self.d_head]).unwrap();

    let mut attention = Array3::zeros((seq_len_q, self.n_head, seq_len_k));

    // attention, ndarray doesn't support batched dot product for Array3
    // so we manually iterate over the heads
    izip!(
      attention.axis_iter_mut(Axis(1)),
      q_reshaped.axis_iter(Axis(1)),
      k_reshaped.axis_iter(Axis(1)),
    )
    .for_each(|(mut attn, q, k)| {
      attn.assign(&q.dot(&k.t()));
    });

    attention /= self.scale; // [seq_len_q, n_head, seq_len_k]
    attention = attention + beta.permuted_axes([1, 0, 2]);

    // softmax
    attention.axis_iter_mut(Axis(1)).for_each(|mut attn| {
      // [seq_len_q, seq_len_k], we do softmax along the last axis
      attn.axis_iter_mut(Axis(0)).for_each(|mut row| {
        row.mapv_inplace(f32::exp);
        let sum = row.sum();
        row /= sum;
      });
    });

    // attn @ v
    let mut output = Array3::<f32>::zeros((seq_len_q, self.n_head, self.d_head));
    izip!(
      output.axis_iter_mut(Axis(1)),
      attention.axis_iter(Axis(1)),
      v_reshaped.axis_iter(Axis(1)),
    )
    .for_each(|(mut out, attn, v)| {
      out.assign(&attn.dot(&v));
    });

    let output = output.to_shape([seq_len_q, self.d_model]).unwrap();
    self.wo.forward(output.view())
  }
}

#[derive(Serialize, Deserialize)]
pub struct FFN {
  pub linear1: Linear,
  pub linear2: Linear,
}

impl FFN {
  pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
    let mut x = self.linear1.forward(x);
    x.mapv_inplace(|x| f32::max(0.0, x));
    self.linear2.forward(x.view())
  }
}

#[derive(Serialize, Deserialize)]
pub struct EncoderLayer {
  pub mhsa: MHSA,
  pub ffn: FFN,
  pub ln1: LayerNorm,
  pub ln2: LayerNorm,
}

impl EncoderLayer {
  pub fn forward(&self, x: ArrayView2<f32>, beta: ArrayView3<f32>) -> Array2<f32> {
    let residual = x.view();
    let mut x = self.mhsa.forward(x, x, x, beta) + &residual;
    self.ln1.forward_inplace(&mut x);
    let residual = x.view();
    let mut x = self.ffn.forward(x.view()) + &residual;
    self.ln2.forward_inplace(&mut x);
    x
  }
}
#[derive(Serialize, Deserialize)]
pub struct DecoderLayer {
  pub self_mhsa: MHSA,
  pub cross_mhsa: MHSA,
  pub ffn: FFN,
  pub ln1: LayerNorm,
  pub ln2: LayerNorm,
  pub ln3: LayerNorm,
}

impl DecoderLayer {
  pub fn forward(
    &self,
    tgt: ArrayView2<f32>,
    enc_out: ArrayView2<f32>,
    beta1: ArrayView3<f32>,
    beta2: ArrayView3<f32>,
  ) -> Array2<f32> {
    let residual = tgt.view();
    let mut tgt = self.self_mhsa.forward(tgt, tgt, tgt, beta1) + &residual;
    self.ln1.forward_inplace(&mut tgt);
    let residual = tgt.view();
    let mut tgt = self.cross_mhsa.forward(tgt.view(), enc_out, enc_out, beta2) + &residual;
    self.ln2.forward_inplace(&mut tgt);
    let residual = tgt.view();
    let mut tgt = self.ffn.forward(tgt.view()) + &residual;
    self.ln3.forward_inplace(&mut tgt);
    tgt += &residual;
    tgt
  }
}
#[derive(Serialize, Deserialize)]
pub struct Beta {
  weight: Array2<f32>, // (2*max_len+1, n_head)
  max_len: usize,
  n_head: usize,
}

impl Beta {
  pub fn new(weight: Array2<f32>) -> Self {
    let max_len = weight.shape()[0] / 2;
    let n_head = weight.shape()[1];
    Self {
      weight,
      max_len,
      n_head,
    }
  }
  pub fn forward(&self, q_len: usize, k_len: usize) -> Array3<f32> {
    let q_range = Array1::<isize>::from_iter(0..q_len as isize);
    let k_range = Array1::<isize>::from_iter(0..k_len as isize);
    let mut diff =
      q_range.insert_axis(Axis(1)) - k_range.insert_axis(Axis(0)) + self.max_len as isize;
    diff.mapv_inplace(|x| x.clamp(0, 2 * self.max_len as isize));
    let shape = [self.n_head, q_len, k_len];
    let mut buff = Array3::<f32>::zeros(shape);
    for i in 0..q_len {
      for j in 0..k_len {
        buff
          .slice_mut(s![.., i, j])
          .assign(&self.weight.index_axis(Axis(0), diff[[i, j]] as usize));
      }
    }
    buff
  }
}

#[derive(Serialize, Deserialize)]
pub struct Encoder {
  pub layers: Vec<EncoderLayer>,
  pub beta: Beta,
}

impl Encoder {
  pub fn forward(&self, mut x: Array2<f32>) -> Array2<f32> {
    let beta = self.beta.forward(x.shape()[0], x.shape()[0]);
    for layer in &self.layers {
      x.assign(&layer.forward(x.view(), beta.view()));
    }
    x
  }
}
#[derive(Serialize, Deserialize)]
pub struct Decoder {
  pub layers: Vec<DecoderLayer>,
  pub beta1: Beta, // for self-attention
  pub beta2: Beta, // for cross-attention
}

impl Decoder {
  pub fn forward(&self, mut tgt: Array2<f32>, enc_out: ArrayView2<f32>) -> Array2<f32> {
    let beta1 = self.beta1.forward(tgt.shape()[0], tgt.shape()[0]);
    let beta2 = self.beta2.forward(tgt.shape()[0], enc_out.shape()[0]);
    for layer in &self.layers {
      tgt.assign(&layer.forward(tgt.view(), enc_out, beta1.view(), beta2.view()));
    }
    tgt
  }
}

#[derive(Serialize, Deserialize)]
pub struct Transformer {
  pub src_emb: Embedding,
  pub tgt_emb: Embedding,
  pub encoder: Encoder,
  pub decoder: Decoder,
  pub fc: Linear,
}

impl Transformer {
  // the all-in-one function for inference
  // including autoregressive decoding
  pub fn inference(&self, src: Array1<usize>) -> Array1<usize> {
    let mut tgt = Array1::<usize>::zeros(MAX_LEN);
    tgt[0] = SOS_IDX;
    let src = self.src_emb.forward(src.view());
    let enc_out = self.encoder.forward(src);
    let mut i = 1;
    while i < MAX_LEN {
      let dec_in = self.tgt_emb.forward(tgt.slice(s![..i]).view());
      let dec_out = self.decoder.forward(dec_in, enc_out.view());
      let dec_out = self.fc.forward(dec_out.view());
      let next_token = argmax(dec_out.index_axis(Axis(0), i - 1));
      tgt[i] = next_token;
      if next_token == EOS_IDX {
        break;
      }
      i += 1;
    }
    tgt.slice(s![1..i]).to_owned()
  }
}
