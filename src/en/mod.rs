//! # English phonemizer module
//! This module contains the English specific functionalities.
//!
//! The most common usage is as follows:
//! ```
//! use celosia::en::Phonemizer as EnPhonemizer;
//! let phonemizer = EnPhonemizer::default();
//! // to specify the cache size
//! // let phonemizer = Phonemizer::new(NonZeroUsize(128).unwrap())
//! let text = "Printing, in the only sense with which we are at present concerned.";
//! let phonemes = phonemizer.phonemize(text);
//! println!("{:?}", phonemes);
//! ```
//! ### Numbers
//! By default the module decides how to spell out numbers accroding
//! to some very simple rules, you can specify the way it is spelled out by
//! calling the corresponding functions:
//! ```
//! use celosia::en::number::{spell_as_is, spell_as_digit};
//!
//! let num: i64 = -12345;
//! spell_as_is(num); // [minus, one, two, three, four, five]
//! spell_as_digit(num); // [minus, twelve, thousand, three, hundred, forty, five]
//! ```

pub(crate) mod constant;
pub(crate) mod phonemizer;

pub mod number;
pub mod tagger;
pub mod tokenizer;

pub use constant::{EN_ALPHABET, EN_PHONEME};
pub use phonemizer::Phonemizer; // re-export for external use
