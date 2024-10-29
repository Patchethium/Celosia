/// The all-in-one phonemizer for English
use hashbrown::{HashMap, HashSet};
use std::io::{BufReader, Read};
use std::num::NonZeroUsize;

use anyhow::Result;
use bimap::BiMap;
use bitcode;
use lru::LruCache;
use zstd;

use super::constant::{
  EN_ALPHABET, EN_PHONEME, EN_VOWELS, PHONEMIZER_DATA, PUNCTUATION, UNK_TOKEN,
};
use super::number::get_num;
use super::tagger::PerceptronTagger;
use super::tagger::{TaggerClasses, TaggerTagdict, TaggerWeight};
use super::tokenizer::naive_split;
use crate::en::tagger::match_pos;
use crate::g2p::constant::SPECIAL_LEN;
use crate::g2p::model::Transformer;
use crate::g2p::wrapper::G2P;
use crate::utils::to_bimap;

pub type NormalDict = HashMap<String, Vec<usize>>;
pub type HomoDict = HashMap<String, HashMap<String, Vec<usize>>>;

pub type PhonemizerData = (
  (NormalDict, HomoDict),
  (TaggerWeight, TaggerTagdict, TaggerClasses),
  Transformer,
);

pub fn parse_data(data: &[u8]) -> Result<PhonemizerData> {
  let reader = BufReader::new(data);
  let mut decoder = zstd::Decoder::new(reader)?;
  let mut buffer = Vec::new();
  decoder.read_to_end(&mut buffer)?;
  let data = bitcode::deserialize::<PhonemizerData>(&buffer)?;
  Ok(data)
}

/// The all-in-one phonemizer for English.
///
/// It transforms a whole sentence into phonemes, by:
/// - Splitting the sentence into words and punctuation
/// - Normalizing the words
/// - Get the POS tag with an average perceptron tagger
/// - Checking for homophones and disambiguate them with POS tags
/// - Checking for normal words. Those POS tags don't match any homophones
/// will also fall back to this category
/// - If not found, use the G2P model to get the phonemes
/// - The G2P model has a LRU cache by default, its size can be specified
///
/// **Example:**
/// ```rust
/// use celosia::en::Phonemizer as EnPhonemizer;
/// let mut phonemizer = EnPhonemizer::default();
/// // to specify the cache size
/// // let phonemizer = Phonemizer::new(NonZeroUsize(128).unwrap())
/// let text = "Printing, in the only sense with which we are at present concerned.";
/// let phonemes = phonemizer.phonemize(text);
/// println!("{:?}", phonemes);
/// ```
/// Note: the initialization of the phonemizer is quite heavy (takes one whole second), so it is recommended
/// to keep the instance alive and reuse it.
pub struct Phonemizer {
  char_map: BiMap<char, usize>,
  ph_map: BiMap<&'static str, usize>,
  normal: NormalDict,
  homo: HomoDict,
  tagger: PerceptronTagger,
  g2p: G2P,
  punc_map: HashSet<char>,
  cache: Option<LruCache<String, Vec<usize>>>,
}

impl Default for Phonemizer {
  fn default() -> Self {
    Self::new(PHONEMIZER_DATA, 128)
  }
}

impl Phonemizer {
  pub fn new(data: &[u8], cache_size: usize) -> Self {
    let char_map = to_bimap(&EN_ALPHABET, SPECIAL_LEN);
    let ph_map = to_bimap(&EN_PHONEME, 0);
    let ((normal, homo), tagger_data, trf) = parse_data(data).unwrap();
    let tagger = PerceptronTagger::new(tagger_data);
    let g2p = G2P::new(trf);
    let punc_map = PUNCTUATION.chars().collect();
    let cache = match cache_size {
      0 => None,
      x => Some(LruCache::new(NonZeroUsize::new(x).unwrap())),
    };
    Self {
      char_map,
      ph_map,
      normal,
      homo,
      tagger,
      g2p,
      punc_map,
      cache,
    }
  }

  pub fn char2idx(&self, word: impl AsRef<str>) -> Vec<usize> {
    word
      .as_ref()
      .chars()
      .into_iter()
      .map(|c| self.char_map.get_by_left(&c).unwrap_or(&0).to_owned())
      .collect()
  }

  pub fn idx2ph(&self, indices: &Vec<usize>) -> Vec<&'static str> {
    indices
      .iter()
      .map(|i| *self.ph_map.get_by_right(i).unwrap_or(&UNK_TOKEN))
      .collect()
  }

  pub fn set_cache_size(&mut self, size: usize) {
    match size {
      0 => self.cache = None,
      x => self.cache = Some(LruCache::new(NonZeroUsize::new(x).unwrap())),
    }
  }

  /// Call the phonemizer and process the sentence.
  /// The returned phonemes are stored in a manner of Vec<Vec<&str>>.
  /// Be noted that the punctuations in `()[]{},!?'` are preserved,
  /// their phonemes are empty `Vec`s.
  ///
  /// **Example:**
  /// ```text
  /// "Hello, world!" -> ["Hello", ",", "world", "!"] -> [["hh","ax","l","ow1"], [], ["w","er1","l","d"], []]
  /// ```
  // TODO: `clone()` are everywhere and may drag the performance down
  pub fn phonemize_indices(&mut self, text: &str) -> Vec<Vec<usize>> {
    let mut words = naive_split(text);
    let mut result: Vec<Vec<usize>> = Vec::new();

    // check for numbers
    let mut i = 0;
    while i < words.len() {
      if let Ok(transformed) = get_num(words[i]) {
        let len = transformed.len();
        words.remove(i);
        words.splice(i..i, transformed.into_iter());
        i += len;
      }
      i += 1;
    }

    let data = self.tagger.tag(&words, false, true);

    for (word, tag, _) in data.iter() {
      if word.len() == 1 && self.punc_map.contains(&word.chars().next().unwrap()) {
        result.push(vec![]);
        continue;
      }
      let word = word.clone().to_lowercase();
      let tag = tag.clone();
      let mut flag = true;
      if let Some(ph_map) = self.homo.get(&word) {
        for (possible_pos, ph) in ph_map.iter() {
          if match_pos(&tag, possible_pos) {
            flag = false;
            result.push(ph.clone());
            break;
          }
        }
      }
      if flag {
        if let Some(ph) = self.normal.get(&word) {
          result.push(ph.clone())
        } else {
          // not found in dictionary, use g2p to predict the possible spelling
          // first check if it's inside cache
          let cached = self.cache.as_mut().and_then(|cache| cache.get(&word));
          if let Some(ph) = cached {
            result.push(ph.to_owned())
          } else {
            let char_indices = self.char2idx(&word);
            let ph_indices = self.g2p.inference(char_indices);
            result.push(ph_indices.to_owned()); // clone also happens here.
            if let Some(cache) = &mut self.cache {
              cache.put(word, ph_indices);
            }
          }
        }
      }
    }
    self.post_process(&words, &mut result);
    result
  }

  pub fn phonemize(&mut self, text: &str) -> Vec<Vec<&'static str>> {
    let vec_indices = self.phonemize_indices(text);
    vec_indices
      .iter()
      .map(|indices| self.idx2ph(indices))
      .collect()
  }

  fn post_process(&self, words: &Vec<&str>, phonemes: &mut Vec<Vec<usize>>) {
    // the -> dh ax, but will become dh iy when followed by a vowel
    for (i, w) in words.iter().enumerate() {
      if w.to_lowercase() == "the" && i + 1 < words.len() {
        if let Some(next) = phonemes.get(i + 1) {
          if EN_VOWELS.contains(self.ph_map.get_by_right(&next[0]).unwrap()) {
            phonemes[i] = vec![23, 40];
          }
        }
      }
    }
  }
}
