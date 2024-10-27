use crate::g2p::constant::{EOS_IDX, SOS_IDX};
use crate::g2p::model::Transformer;
use crate::g2p::serde::load_trf;
use ndarray::Array1;

pub struct G2P {
  trf: Transformer,
}

impl G2P {
  pub fn new(data: &[u8]) -> Self {
    let trf = load_trf(data);
    Self { trf }
  }

  pub fn inference(&self, mut indices: Vec<usize>) -> Vec<usize> {
    indices.insert(0, SOS_IDX);
    indices.push(EOS_IDX);

    let input = Array1::from_vec(indices.clone());
    let output = self.trf.inference(input).to_vec();

    output
  }
}
