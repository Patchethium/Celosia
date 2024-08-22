#![allow(unused)]

use celosia::en::phonemizer::Phonemizer;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn bench_phonemize(c: &mut Criterion) {
  let phonemizer = Phonemizer::default();
  c.bench_function("phonemize", |b| {
    b.iter(|| phonemizer.phonemize(black_box("Warning: Unable to complete 100 samples in 5.0s. You may wish to increase target time to 8.4s, enable flat sampling, or reduce sample count to 50.")))
  });
}

pub fn bench_loading(c: &mut Criterion) {
  c.bench_function("load phonemizer", |b| {
    b.iter(|| {let phonemizer = Phonemizer::default();})
  });
}

criterion_group!(benches, bench_phonemize, bench_loading);
criterion_main!(benches);
