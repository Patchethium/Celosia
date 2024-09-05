use crate::en::constant::EN_G2P_DATA;
use crate::g2p::constant::{EOS_IDX, LANG, SOS_IDX};
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
  // I don't really know what it would actually work in a real-world scenario, i.e. multi-threads.
  // I also use rc as the value type to avoid cloning the whole vector.
  cache: Option<Mutex<RefCell<lru::LruCache<Vec<usize>, Vec<usize>>>>>,
}

impl G2P {
  pub fn new(lang: LANG, cache_size: usize) -> Self {
    let data = match lang {
      LANG::EN => EN_G2P_DATA,
      _ => panic!("Unsupported language {}", lang.to_string()),
    };
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
}
