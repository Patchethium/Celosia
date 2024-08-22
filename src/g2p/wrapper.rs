use crate::g2p::constant::{G2PConfig, EOS_IDX, SOS_IDX};
use crate::g2p::model::Transformer;
use crate::g2p::serde::load_trf;
use lru::LruCache;
use ndarray::Array1;
use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::sync::Mutex;

use super::constant::PAD_IDX;

pub struct G2P {
  config: G2PConfig,
  trf: Transformer,
  // workaround to make the compiler happy
  cache: Mutex<RefCell<lru::LruCache<String, Vec<&'static str>>>>,
}

impl G2P {
  pub fn new(config: G2PConfig, data: &[u8], cache_size: NonZeroUsize) -> Self {
    let trf = load_trf(data);
    Self {
      config,
      trf,
      cache: Mutex::new(RefCell::new(LruCache::new(cache_size))),
    }
  }

  pub fn inference(&self, word: &str) -> Vec<&str> {
    if let Some(phoneme) = self.cache.lock().unwrap().borrow_mut().get(word) {
      return phoneme.clone();
    }

    let mut indices = word
      .to_lowercase()
      .chars()
      .map(|c| self.config.alphabet.get_by_left(&c).unwrap_or(&PAD_IDX))
      .collect::<Vec<_>>();
    indices.insert(0, &SOS_IDX);
    indices.push(&EOS_IDX);

    let input = Array1::from(indices).map(|&x| *x);
    let output = self.trf.inference(input);

    let phoneme = output
      .iter()
      .map(|&idx| *self.config.phoneme.get_by_right(&idx).unwrap())
      .collect::<Vec<_>>();

    self
      .cache
      .lock()
      .unwrap()
      .borrow_mut()
      .put(word.to_string(), phoneme.clone());
    phoneme
  }

  pub fn clean_cache(&mut self) {
    self.cache.lock().unwrap().borrow_mut().clear();
  }
}
