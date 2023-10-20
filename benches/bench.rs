use anthelia::en::{tagger::PerceptronTagger, rnn::{Linear, GRUCell, GRU}};
use criterion::{criterion_group, criterion_main, Criterion};
use ndarray::{Array2, Array, Array1};
use ndarray_rand::{RandomExt, rand_distr::Uniform};

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

fn gru_benchmark(c: &mut Criterion) {
  let h_dim = 32;
  let seq_dim = 126;
  let w_ih: Array2<f32> = Array::random((3 * h_dim, h_dim), Uniform::new(0., 10.));
  let w_hh: Array2<f32> = Array::random((3 * h_dim, h_dim), Uniform::new(0., 10.));

  let b_ih: Array1<f32> = Array::random((3 * h_dim,), Uniform::new(0., 10.));
  let b_hh: Array1<f32> = Array::random((3 * h_dim,), Uniform::new(0., 10.));

  let linear_ih = Linear::new(w_ih, b_ih);
  let linear_hh = Linear::new(w_hh, b_hh);

  let gru_cell = GRUCell::new(linear_ih, linear_hh);
  
  let gru = GRU::new(gru_cell, true);

  let x: Array2<f32> = Array::random((seq_dim, h_dim), Uniform::new(0., 10.));

  c.bench_function("bench_gru", |b| {
    b.iter(|| gru.forward(&x))
  });
}


criterion_group!(benches, tagger_benchmark, gru_benchmark);
criterion_main!(benches);
