use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;

use serde_json::from_str;

use super::data::{EN_G2P_DATA, EN_HOMO_DICT, EN_NORMAL_DICT, EN_TAGGER_DATA, PUNCTUATION};
use super::number::get_num;
use super::tagger;
use super::tokenizer::naive_split;
use crate::en::tagger::match_pos;
use crate::g2p::constant::{G2PConfig, LANG};
use crate::g2p::wrapper::G2P;

type NormalDict = HashMap<String, Vec<&'static str>>;
type HomoDict = HashMap<String, HashMap<String, Vec<&'static str>>>;

pub fn parse_dict() -> (NormalDict, HomoDict) {
  let normal = from_str(&EN_NORMAL_DICT).unwrap();
  let homo = from_str(&EN_HOMO_DICT).unwrap();
  (normal, homo)
}

pub struct Phonemizer {
  normal: NormalDict,
  homo: HomoDict,
  tagger: tagger::PerceptronTagger,
  g2p: G2P,
  punc_map: HashSet<char>,
}

impl Default for Phonemizer {
  fn default() -> Self {
    Self::new(NonZeroUsize::new(128).unwrap())
  }
}

impl Phonemizer {
  pub fn new(cache_size: NonZeroUsize) -> Self {
    let (normal, homo) = parse_dict();
    let tagger = tagger::PerceptronTagger::new(EN_TAGGER_DATA);
    let config = G2PConfig::new(LANG::EN);
    let g2p = G2P::new(config, EN_G2P_DATA, cache_size);
    let punc_map = PUNCTUATION.chars().collect();

    Self {
      normal,
      homo,
      tagger,
      g2p,
      punc_map,
    }
  }

  // FIXME: `clone()` are everywhere and may drag the performance down
  pub fn phonemize(&self, text: &str) -> Vec<Vec<&str>> {
    let mut words = naive_split(text);
    let mut result = Vec::new();

    // check for all UPPER CASE words, we spell them one-by-one

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
          let ph = self.g2p.inference(&word);
          result.push(ph.clone());
        }
      }
    }
    result
  }
}
