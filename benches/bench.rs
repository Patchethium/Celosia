use anthelia::en::tagger::PerceptronTagger;
use criterion::black_box;
use criterion::{criterion_group, criterion_main, Criterion};

use anthelia::g2p::model::G2P;

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

fn bench_load_g2p(c: &mut Criterion) {
  let path = "./assets/en.pickle";
  c.bench_function("bench_load_g2p", |b| {
    b.iter(|| G2P::load(path, Default::default()).unwrap())
  });
}

fn bench_g2p(c: &mut Criterion) {
  let path = "./assets/en.pickle";
  let g2p = G2P::load(path, Default::default()).unwrap();
  let word = "gutenberg";
  c.bench_function("bench_g2p", |b| {
    b.iter(|| {
      black_box(g2p.inference(word).unwrap());
    })
  });
}

criterion_group!(benches, tagger_benchmark, bench_load_g2p, bench_g2p);
criterion_main!(benches);
