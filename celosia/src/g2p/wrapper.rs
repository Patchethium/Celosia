use crate::g2p::constant::{EOS_IDX, SOS_IDX};
use crate::g2p::model::Transformer;
use ndarray::Array1;

use super::constant::SPECIAL_LEN;

pub struct G2P {
  trf: Transformer,
}

impl G2P {
  pub fn new(trf: Transformer) -> Self {
    Self { trf }
  }

  pub fn inference(&self, mut indices: Vec<usize>) -> Vec<usize> {
    indices.insert(0, SOS_IDX);
    indices.push(EOS_IDX);

    let input = Array1::from_vec(indices.clone());
    let output = self.trf.inference(input) - SPECIAL_LEN;

    output.to_vec()
  }
}
