use std::collections::BTreeMap;

use super::tagger;
use super::tokenizer;

struct Phonemizer {
  dict: BTreeMap<String, Vec<String>>,
  tagger: tagger::PerceptronTagger,
}
