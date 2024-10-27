use crate::g2p::constant::{EOS_IDX, SOS_IDX};
use crate::g2p::model::Transformer;
use crate::g2p::serde::load_trf;
use lru::LruCache;
use ndarray::Array1;
use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::sync::Mutex;

pub struct G2P {
  trf: Transformer,
  // The Mutex<RefCell<...>>> stuff is a workaround to make the compiler happy with initializing this struct as lazy_static.
  cache: Option<Mutex<RefCell<lru::LruCache<Vec<usize>, Vec<usize>>>>>,
}

impl G2P {
  pub fn new(data: &[u8], cache_size: usize) -> Self {
    let trf = load_trf(data);
    let cache = match cache_size {
      0 => None,
      x => Some(Mutex::new(RefCell::new(LruCache::new(
        NonZeroUsize::new(x).unwrap(),
      )))),
    };
    Self { trf, cache }
  }

  pub fn inference(&self, mut indices: Vec<usize>) -> Vec<usize> {
    if let Some(cache) = &self.cache {
      if let Some(phoneme) = cache.lock().unwrap().borrow_mut().get(&indices) {
        return phoneme.clone();
      }
    }

    indices.insert(0, SOS_IDX);
    indices.push(EOS_IDX);

    let input = Array1::from_vec(indices.clone());
    let output = self.trf.inference(input).to_vec();

    if let Some(cache) = &self.cache {
      cache
        .lock()
        .unwrap()
        .borrow_mut()
        .put(indices.clone(), output.clone());
    }
    output
  }

  pub fn clean_cache(&mut self) {
    if let Some(cache) = &self.cache {
      cache.lock().unwrap().borrow_mut().clear();
    }
  }

  /// reset the cache size, this will clear the cache as well
  pub fn set_cache_size(&mut self, size: usize) {
    match size {
      0 => self.cache = None,
      x => self.cache = Some(Mutex::new(RefCell::new(LruCache::new(
        NonZeroUsize::new(x).unwrap(),
      )))),
    }
  }
}
