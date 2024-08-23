#![allow(unused)]

use celosia::en::Phonemizer;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

// 5 ms/iter
pub fn bench_phonemize(c: &mut Criterion) {
  let phonemizer = Phonemizer::default();
  c.bench_function("phonemize", |b| {
    black_box(b.iter(|| {
      phonemizer.phonemize(
        "Printing, in the only \
              sense with which we are at present concerned, \
              differs from most if not from all the arts and \
              crafts represented in the Exhibition in being \
              comparatively modern.",
      )
    }))
  });
}

// 2 ms/iter
pub fn bench_phonemize_oov(c: &mut Criterion) {
  let phonemizer = Phonemizer::default();
  c.bench_function("phonemize_oov", |b| {
    black_box(b.iter(|| {
      // we use some pokemon names as OOV
      phonemizer.phonemize(
        "Bulbasaur, Ivysaur, Venusaur,\
         Charmander, Charmeleon, Charizard, Squirtle, Wartortle, Blastoise",
      )
    }))
  });
}

// 150 ms/iter
pub fn bench_phonemize_oov_uncached(c: &mut Criterion) {
  let phonemizer = Phonemizer::new(0); // no cache
  let mut group = c.benchmark_group("phonemize_oov_uncached");
  group.sample_size(50);
  group.measurement_time(std::time::Duration::from_secs(10));
  group.bench_function("phonemize_oov_uncached", |b| {
    black_box(b.iter(|| {
      // we use some pokemon names as OOV
      phonemizer.phonemize(
        "Bulbasaur, Ivysaur, Venusaur,\
         Charmander, Charmeleon, Charizard, Squirtle, Wartortle, Blastoise",
      )
    }))
  });
}

// 700 ms/iter
pub fn bench_loading(c: &mut Criterion) {
  let mut group = c.benchmark_group("loading");
  group.sample_size(10); // it is a heavy function, we use smaller samples to avoid taking too long
  group.measurement_time(std::time::Duration::from_secs(10));
  group.bench_function("load-phonemizer", |b| {
    black_box(b.iter(|| Phonemizer::default()))
  });
}

criterion_group!(
  benches,
  bench_phonemize,
  bench_phonemize_oov,
  bench_phonemize_oov_uncached,
  bench_loading
);
criterion_main!(benches);
