use ndarray::{s, Array1, Array2, ArrayView1, Axis};

/// rnn model for oov prediction
pub struct GRUCell {
  linear_ih: Linear,
  linear_hh: Linear,
}

pub struct Linear {
  weight: Array2<f32>,
  bias: Array1<f32>,
}

impl Linear {
  pub fn new(weight: Array2<f32>, bias: Array1<f32>) -> Linear {
    Linear { weight, bias }
  }
  fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
    x.dot(&self.weight.t()) + &self.bias
  }
}

impl GRUCell {
  pub fn new(linear_ih: Linear, linear_hh: Linear) -> GRUCell {
    GRUCell {
      linear_ih,
      linear_hh,
    }
  }
  fn sigmoid(&self, x: &Array1<f32>) -> Array1<f32> {
    1.0 / (1.0 + (-x).map(|v| v.exp()))
  }
  fn tanh(&self, x: &Array1<f32>) -> Array1<f32> {
    x.map(|v| v.tanh())
  }

  fn forward(&self, x: &Array1<f32>, h: &Array1<f32>) -> Array1<f32> {
    let rzn_ih: Array1<f32> = self.linear_ih.forward(x);
    let rzn_hh: Array1<f32> = self.linear_hh.forward(x);

    let rzn_ih_len: usize = rzn_ih.shape()[0] * 2 / 3;
    let rzn_hh_len: usize = rzn_hh.shape()[0] * 2 / 3;

    let rz_ih: ArrayView1<f32> = rzn_ih.slice(s![0..rzn_ih_len]);
    let n_ih: ArrayView1<f32> = rzn_ih.slice(s![rzn_ih_len..]);

    let rz_hh: ArrayView1<f32> = rzn_hh.slice(s![0..rzn_ih_len]);
    let n_hh: ArrayView1<f32> = rzn_hh.slice(s![rzn_hh_len..]);

    let rz: Array1<f32> = self.sigmoid(&(&rz_ih + &rz_hh));

    let r_len: usize = rz_ih.shape()[0] / 2;

    let r: ArrayView1<f32> = rz.slice(s![0..r_len]);
    let z: ArrayView1<f32> = rz.slice(s![r_len..]);

    let n: Array1<f32> = self.tanh(&(&n_ih + &r * &n_hh));

    let updated_h: Array1<f32> = (1.0 - &z) * &n + &z * h;

    updated_h
  }
}

pub struct GRU {
  cell: GRUCell,
  reversed: bool,
}

impl GRU {
  pub fn new(cell: GRUCell, reversed: bool) -> GRU {
    GRU { cell, reversed }
  }
  pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let h_dim = x.shape()[1];
    let mut h = Array1::zeros(h_dim);
    let mut output = Array2::zeros((seq_len, h_dim));

    for t in 0..seq_len {
      h = self.cell.forward(&x.index_axis(Axis(0), t).to_owned(), &h);
      output.index_axis_mut(Axis(0), t).assign(&h);
    }

    output
  }
}
