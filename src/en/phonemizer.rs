/// The all-in-one phonemizer for English
use std::collections::{HashMap, HashSet};

use bimap::BiMap;
use serde_json::from_str;

use super::constant::{
  EN_ALPHABET, EN_HOMO_DICT, EN_NORMAL_DICT, EN_PHONEME, EN_TAGGER_DATA, PUNCTUATION, UNK_TOKEN,
};
use super::number::get_num;
use super::tagger;
use super::tokenizer::naive_split;
use crate::en::tagger::match_pos;
use crate::g2p::constant::LANG;
use crate::g2p::constant::SPECIAL_LEN;
use crate::g2p::wrapper::G2P;
use crate::utils::to_bimap;

type NormalDict = HashMap<String, Vec<usize>>;
type HomoDict = HashMap<String, HashMap<String, Vec<usize>>>;

fn parse_dict() -> (NormalDict, HomoDict) {
  let normal = from_str(&EN_NORMAL_DICT).unwrap();
  let homo = from_str(&EN_HOMO_DICT).unwrap();
  (normal, homo)
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
/// let phonemizer = EnPhonemizer::default();
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
  tagger: tagger::PerceptronTagger,
  g2p: G2P,
  punc_map: HashSet<char>,
}

impl Default for Phonemizer {
  /// The default cache size of G2P model is 128
  fn default() -> Self {
    Self::new(128)
  }
}

impl Phonemizer {
  pub fn new(cache_size: usize) -> Self {
    let char_map = to_bimap(&EN_ALPHABET, SPECIAL_LEN);
    let ph_map = to_bimap(&EN_PHONEME, SPECIAL_LEN);
    let (normal, homo) = parse_dict();
    let tagger = tagger::PerceptronTagger::new(EN_TAGGER_DATA);
    let g2p = G2P::new(LANG::EN, cache_size);
    let punc_map = PUNCTUATION.chars().collect();

    Self {
      char_map,
      ph_map,
      normal,
      homo,
      tagger,
      g2p,
      punc_map,
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
  pub fn phonemize_indices(&self, text: &str) -> Vec<Vec<usize>> {
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
          let char_indices = self.char2idx(&word);
          let ph_indices = self.g2p.inference(char_indices);
          result.push(ph_indices.to_vec()); // clone also happens here.
        }
      }
    }
    result
  }

  pub fn phonemize(&self, text: &str) -> Vec<Vec<&'static str>> {
    let vec_indices = self.phonemize_indices(text);
    vec_indices
      .iter()
      .map(|indices| self.idx2ph(indices))
      .collect()
  }
}
