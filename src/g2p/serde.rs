// load the transformer from the file
use std::io::Cursor;

use crate::g2p::model::{
  Beta, Decoder, DecoderLayer, Embedding, Encoder, EncoderLayer, LayerNorm, Linear, Transformer,
  FFN, MHSA,
};

use super::constant::{N_DEC_LAYER, N_ENC_LAYER, N_HEAD};
use ndarray_npy::NpzReader;

// the npz will automatically append a `.npy` appendix to all the keys
fn get_linear(reader: &mut NpzReader<Cursor<&[u8]>>, prefix: &str) -> Linear {
  let weight = reader.by_name(&format!("{}.weight.npy", prefix)).unwrap();
  let bias = reader.by_name(&format!("{}.bias.npy", prefix)).unwrap();
  Linear { weight, bias }
}

fn get_embedding(reader: &mut NpzReader<Cursor<&[u8]>>, prefix: &str) -> Embedding {
  let weight = reader.by_name(&format!("{}.weight.npy", prefix)).unwrap();
  Embedding { weight }
}

fn get_ln(reader: &mut NpzReader<Cursor<&[u8]>>, prefix: &str) -> LayerNorm {
  let weight = reader.by_name(&format!("{}.weight.npy", prefix)).unwrap();
  let bias = reader.by_name(&format!("{}.bias.npy", prefix)).unwrap();
  LayerNorm { weight, bias }
}

// example: encoder.beta.weight
fn get_beta(reader: &mut NpzReader<Cursor<&[u8]>>, key: &str) -> Beta {
  let weight = reader.by_name(key).unwrap();
  Beta::new(weight)
}

fn get_mhsa(reader: &mut NpzReader<Cursor<&[u8]>>, prefix: &str) -> MHSA {
  let q_weight = reader.by_name(&format!("{}.q.weight.npy", prefix)).unwrap();
  let q_bias = reader.by_name(&format!("{}.q.bias.npy", prefix)).unwrap();
  let k_weight = reader.by_name(&format!("{}.k.weight.npy", prefix)).unwrap();
  let k_bias = reader.by_name(&format!("{}.k.bias.npy", prefix)).unwrap();
  let v_weight = reader.by_name(&format!("{}.v.weight.npy", prefix)).unwrap();
  let v_bias = reader.by_name(&format!("{}.v.bias.npy", prefix)).unwrap();
  let o_weight = reader.by_name(&format!("{}.o.weight.npy", prefix)).unwrap();
  let o_bias = reader.by_name(&format!("{}.o.bias.npy", prefix)).unwrap();
  MHSA::new(
    Linear {
      weight: q_weight,
      bias: q_bias,
    },
    Linear {
      weight: k_weight,
      bias: k_bias,
    },
    Linear {
      weight: v_weight,
      bias: v_bias,
    },
    Linear {
      weight: o_weight,
      bias: o_bias,
    },
    N_HEAD,
  )
}

// example: "encoder.layers.0.mhsa.q.weight", where layer_num = 0
// this will take huge time to load the model...
fn get_encoder_layer(reader: &mut NpzReader<Cursor<&[u8]>>, layer_num: usize) -> EncoderLayer {
  let mhsa = get_mhsa(reader, &format!("encoder.layers.{}.mhsa", layer_num));
  let ffn_l1_weight = reader
    .by_name(&format!("encoder.layers.{}.ffn.l1.weight.npy", layer_num))
    .unwrap();
  let ffn_l1_bias = reader
    .by_name(&format!("encoder.layers.{}.ffn.l1.bias.npy", layer_num))
    .unwrap();
  let ffn_l2_weight = reader
    .by_name(&format!("encoder.layers.{}.ffn.l2.weight.npy", layer_num))
    .unwrap();
  let ffn_l2_bias = reader
    .by_name(&format!("encoder.layers.{}.ffn.l2.bias.npy", layer_num))
    .unwrap();
  let ffn = FFN {
    linear1: Linear {
      weight: ffn_l1_weight,
      bias: ffn_l1_bias,
    },
    linear2: Linear {
      weight: ffn_l2_weight,
      bias: ffn_l2_bias,
    },
  };
  let ln1 = get_ln(reader, &format!("encoder.layers.{}.ln1", layer_num));
  let ln2 = get_ln(reader, &format!("encoder.layers.{}.ln2", layer_num));
  EncoderLayer {
    mhsa,
    ffn,
    ln1,
    ln2,
  }
}

fn get_decoder_layer(reader: &mut NpzReader<Cursor<&[u8]>>, layer_num: usize) -> DecoderLayer {
  let mhsa1 = get_mhsa(reader, &format!("decoder.layers.{}.mhsa1", layer_num));
  let mhsa2 = get_mhsa(reader, &format!("decoder.layers.{}.mhsa2", layer_num));
  let ffn_l1_weight = reader
    .by_name(&format!("decoder.layers.{}.ffn.l1.weight.npy", layer_num))
    .unwrap();
  let ffn_l1_bias = reader
    .by_name(&format!("decoder.layers.{}.ffn.l1.bias.npy", layer_num))
    .unwrap();
  let ffn_l2_weight = reader
    .by_name(&format!("decoder.layers.{}.ffn.l2.weight.npy", layer_num))
    .unwrap();
  let ffn_l2_bias = reader
    .by_name(&format!("decoder.layers.{}.ffn.l2.bias.npy", layer_num))
    .unwrap();
  let ffn = FFN {
    linear1: Linear {
      weight: ffn_l1_weight,
      bias: ffn_l1_bias,
    },
    linear2: Linear {
      weight: ffn_l2_weight,
      bias: ffn_l2_bias,
    },
  };
  let ln1 = get_ln(reader, &format!("decoder.layers.{}.ln1", layer_num));
  let ln2 = get_ln(reader, &format!("decoder.layers.{}.ln2", layer_num));
  let ln3 = get_ln(reader, &format!("decoder.layers.{}.ln3", layer_num));
  DecoderLayer {
    self_mhsa: mhsa1,
    cross_mhsa: mhsa2,
    ffn,
    ln1,
    ln2,
    ln3,
  }
}

fn get_encoder(reader: &mut NpzReader<Cursor<&[u8]>>) -> Encoder {
  let mut layers = Vec::with_capacity(N_ENC_LAYER);
  for i in 0..N_ENC_LAYER {
    layers.push(get_encoder_layer(reader, i));
  }
  let beta = get_beta(reader, "encoder.beta.weight.npy");
  Encoder { layers, beta }
}

fn get_decoder(reader: &mut NpzReader<Cursor<&[u8]>>) -> Decoder {
  let mut layers = Vec::with_capacity(N_DEC_LAYER);
  for i in 0..N_DEC_LAYER {
    layers.push(get_decoder_layer(reader, i));
  }
  let beta1 = get_beta(reader, "decoder.beta1.weight.npy");
  let beta2 = get_beta(reader, "decoder.beta2.weight.npy");
  Decoder {
    layers,
    beta1,
    beta2,
  }
}

// data is raw bytes of the npz file
// included with `include_bytes!` macro
pub fn load_trf(data: &[u8]) -> Transformer {
  let cursor = Cursor::new(data);
  let mut npz = NpzReader::new(cursor).unwrap();
  if npz.is_empty() {
    panic!("empty npz file");
  }
  let src_emb = get_embedding(&mut npz, "src_emb");
  let tgt_emb = get_embedding(&mut npz, "tgt_emb");
  let encoder = get_encoder(&mut npz);
  let decoder = get_decoder(&mut npz);
  let fc = get_linear(&mut npz, "fc");
  Transformer {
    src_emb,
    tgt_emb,
    encoder,
    decoder,
    fc,
  }
}
