//! # G2P module for phonogram languages
//! This module contains inference functionalities for a Grapheme-to-Phoneme (G2P) model.
//! The term `G2P` is the process of converting a *single* word into its phonetic representation.
//! This module is mainly used on Out-Of-Vocabulary (OOV) words that dictionary didn't cover.
//! ## Example
//! ```rust
//! use celosia::g2p::G2P;
//! use celosia::g2p::LANG;
//! 
//! let g2p = G2P::new(LANG::EN, 128);
//! 
//! let word = "celosia";
//! let phoneme = g2p.inference(word);
//! ```
//! The G2P model is a encoder-decoder transformer model, each has 256 attention dim,
//! 2 layers and 4 heads. This makes inference very time consuming, by default it contains
//! a LRU cache with a size of 128, you can specify the cache size in the `new` function,
//! or pass 0 to disable the cache (not recommended).
pub(crate) mod constant;
pub(crate) mod model;
pub(crate) mod serde;
pub(crate) mod wrapper;

pub use wrapper::G2P;
pub use constant::LANG;