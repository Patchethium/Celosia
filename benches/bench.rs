use anthelia::en::{
  model::{GRUCell, Linear, GRU, Embedding, Encoder},
  tagger::PerceptronTagger,
};
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array, Array1, Array2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

fn tagger_benchmark(c: &mut Criterion) {
  const PATH: &str = "./data/averaged_perceptron_tagger.pickle";
  let tagger = PerceptronTagger::new(PATH).unwrap();

  let v = "Printing , in the only sense with which we are at present concerned ,
  differs from most if not from all the arts and crafts
  represented in the Exhibition in being comparatively modern .";
  let tokens: Vec<String> = v.split_whitespace().map(|s| s.to_string()).collect();

  c.bench_function("bench_tagger", |b| {
    b.iter(|| tagger.tag(tokens.clone(), false, true))
  });
}
fn get_gru(h_dim: usize) -> GRU {
  let w_ih: Array2<f32> = Array::random((3 * h_dim, h_dim), Uniform::new(0., 10.));
  let w_hh: Array2<f32> = Array::random((3 * h_dim, h_dim), Uniform::new(0., 10.));

  let b_ih: Array1<f32> = Array::random((3 * h_dim,), Uniform::new(0., 10.));
  let b_hh: Array1<f32> = Array::random((3 * h_dim,), Uniform::new(0., 10.));

  let linear_ih = Linear::new(w_ih, b_ih);
  let linear_hh = Linear::new(w_hh, b_hh);

  let gru_cell = GRUCell::new(linear_ih, linear_hh);

  GRU::new(gru_cell, true)
}

fn get_linear(i_dim: usize, o_dim: usize) ->Linear {
  let weight: Array2<f32> = Array::random((o_dim, i_dim), Uniform::new(0., 10.));
  let bias: Array1<f32> = Array::random((o_dim,), Uniform::new(0., 10.));
  Linear::new(weight,bias)
}

fn gru_benchmark(c: &mut Criterion) {
  let h_dim = 256;
  let seq_dim = 12;
  
  let gru = get_gru(h_dim);

  let x: Array2<f32> = Array::random((seq_dim, h_dim), Uniform::new(0., 10.));
  let h = Array1::zeros(h_dim);

  c.bench_function("bench_gru", |b| b.iter(|| gru.forward(&x, &h)));
}

fn encoder_benchmark(c: &mut Criterion) {
  let h_dim = 256;
  let seq_dim = 12;
  let dist = Uniform::new(0., 10.);
  let dist_usize = Uniform::<usize>::new(0, 10);

  let gru = get_gru(h_dim);
  let gru_rev = get_gru(h_dim);

  let emb_weight = Array::random((h_dim, 10), dist);
  let emb = Embedding::new(emb_weight);

  let mut data = Array::random((seq_dim, ), dist_usize);

  let encoder = Encoder::new(emb, gru, gru_rev, get_linear(2*h_dim, h_dim));

  c.bench_function("bench_encoder", |b| b.iter(|| encoder.forward(&mut data)));
}

criterion_group!(benches, tagger_benchmark, gru_benchmark, encoder_benchmark);
criterion_main!(benches);
