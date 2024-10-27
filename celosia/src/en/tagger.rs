use ordered_float::OrderedFloat;
use std::cmp::{Ordering, PartialOrd};
use std::collections::{BTreeMap, BTreeSet};

pub type TaggerWeight = BTreeMap<String, BTreeMap<String, f64>>;
pub type TaggerTagdict = BTreeMap<String, String>;
pub type TaggerClasses = BTreeSet<String>;

pub type TaggerData = (TaggerWeight, TaggerTagdict, TaggerClasses);

#[derive(Debug)]
struct AveragedPerceptron {
  weights: BTreeMap<String, BTreeMap<String, f64>>,
  classes: BTreeSet<String>,
  _totals: BTreeMap<(String, String), f64>,
  _tstamps: BTreeMap<(String, String), i32>,
}

impl AveragedPerceptron {
  pub fn new(weights: Option<BTreeMap<String, BTreeMap<String, f64>>>) -> AveragedPerceptron {
    let weights = weights.unwrap_or_default();
    let classes = BTreeSet::new();
    let _totals: BTreeMap<(String, String), f64> = BTreeMap::new();
    let _tstamps: BTreeMap<(String, String), i32> = BTreeMap::new();

    AveragedPerceptron {
      weights,
      classes,
      _totals,
      _tstamps,
    }
  }

  pub fn predict(
    &self,
    features: &BTreeMap<String, i32>,
    return_conf: bool,
  ) -> (String, Option<f64>) {
    let mut scores: BTreeMap<String, f64> = BTreeMap::new();
    for (feat, value) in features.iter() {
      if !self.weights.contains_key(feat) || *value == 0 {
        continue;
      }
      if let Some(weights) = self.weights.get(feat) {
        for (label, weight) in weights.iter() {
          *scores.entry(label.into()).or_insert(0.0) +=
            (value.to_owned() as f64 * weight as &f64) as f64
        }
      }
    }
    // Do a secondary alphabetic sort, for stability
    let best_label = self
      .classes
      .iter()
      .max_by_key(|&label| {
        (
          // hate Rust, no compare float >:(
          OrderedFloat::<f64>(scores.get(label).unwrap_or(&0.0).to_owned()),
          label,
        )
      })
      .unwrap()
      .to_string();
    if return_conf {
      let softmax = self.softmax(&scores);
      let conf = softmax
        .iter()
        .cloned()
        .max_by(|&a, &b| a.partial_cmp(&b).unwrap_or(Ordering::Equal));
      (best_label, conf)
    } else {
      (best_label, None)
    }
  }

  // update and avg are training only, use python for it
  // pub fn update
  // pub fn average_weights

  fn softmax(&self, scores: &BTreeMap<String, f64>) -> Vec<f64> {
    let s: Vec<f64> = scores.values().cloned().collect();
    let exps: Vec<f64> = s.iter().map(|&score| score.exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&exp| exp / sum).collect()
  }
}

/// An averaged perceptron tagger, using the weights from nltk data,
/// manually transcribed from its Python version, Apache 2.0.
///
/// **Usage:**
/// ```
/// use celosia::en::tagger::PerceptronTagger as tagger;
/// use celosia::en::tokenizer::naive_split;
/// use celosia::en::parse_data;
/// use celosia::en::PHONEMIZER_DATA;
/// 
/// let (_, tagger_data, _) = parse_data(PHONEMIZER_DATA).unwrap();
/// let tagger = tagger::new(tagger_data);
/// let sentence = "The quick brown fox jumps over the lazy dog.";
/// let tokens = naive_split(sentence);
/// let tagged = tagger.tag(&tokens, false, true);
/// println!("{:?}", tagged);
/// ```
#[derive(Debug)]
pub struct PerceptronTagger {
  model: AveragedPerceptron,
  tagdict: BTreeMap<String, String>,
  classes: BTreeSet<String>,
  _sentences: Vec<Vec<(String, String)>>,
}

static START: [&str; 2] = ["-START-", "-START2-"];
static END: [&str; 2] = ["-END-", "-END2-"];

impl PerceptronTagger {
  pub fn new(data: TaggerData) -> PerceptronTagger {
    let mut model = AveragedPerceptron::new(None);
    let (weights, tagdict, classes) = data;
    model.weights = weights;
    model.classes = classes.clone();
    let mut sentences: Vec<Vec<(String, String)>> = Vec::new();

    let mut tagger = PerceptronTagger {
      model,
      tagdict,
      classes,
      _sentences: Vec::new(),
    };
    
    tagger.make_tagdict(&mut sentences);

    tagger
  }

  pub fn tag(
    &self,
    tokens: &Vec<&str>,
    return_conf: bool,
    use_tagdict: bool,
  ) -> Vec<(String, String, Option<f64>)> {
    let mut prev = START[0].to_string();
    let mut prev2 = START[1].to_string();
    let mut output: Vec<(String, String, Option<f64>)> = Vec::new();

    let mut context: Vec<String> = Vec::new();
    context.extend(START.iter().map(|s| s.to_string()));
    context.extend(tokens.iter().map(|w| self.normalize(w)));
    context.extend(END.iter().map(|s| s.to_string()));

    for (i, word) in tokens.iter().enumerate() {
      let (tag, conf_old) = if use_tagdict {
        (self.tagdict.get(*word), Some(1.0 as f64))
      } else {
        (None, None)
      };

      let (tag, conf) = match tag {
        None => {
          let features: BTreeMap<String, i32> =
            self._get_features(i, word, &context, &prev, &prev2);
          self.model.predict(&features, return_conf)
        }
        Some(s) => (s.to_owned(), conf_old),
      };

      output.push((word.to_string(), tag.to_string(), conf));

      prev2 = prev.to_string();
      prev = tag.to_string();
    }

    output
  }

  fn make_tagdict(&mut self, sentences: &mut Vec<Vec<(String, String)>>) {
    let mut counts: BTreeMap<String, BTreeMap<String, i32>> = BTreeMap::new();

    for sentence in sentences.iter() {
      for (word, tag) in sentence {
        counts
          .entry(word.to_string())
          .or_insert(BTreeMap::new())
          .entry(tag.to_string())
          .and_modify(|count| *count += 1)
          .or_insert(1);
        self.classes.insert(tag.to_string());
      }
    }

    let freq_thresh = 20;
    let ambiguity_thresh = OrderedFloat::<f64>(0.97);

    for (word, tag_freqs) in counts.iter() {
      if let Some((tag, mode)) = tag_freqs
        .iter()
        .max_by(|(_, count1), (_, count2)| count1.cmp(count2))
      {
        let n: i32 = tag_freqs.values().sum();
        if n > freq_thresh && OrderedFloat::<f64>(*mode as f64 / n as f64) >= ambiguity_thresh {
          self.tagdict.insert(word.to_string(), tag.to_string());
        }
      }
    }
  }

  fn normalize(&self, word: &str) -> String {
    if word.contains('-') && word.chars().next() != Some('-') {
      "!HYPHEN".to_string()
    } else if word.chars().all(char::is_numeric) && word.len() == 4 {
      "!YEAR".to_string()
    } else if !word.is_empty() && word.chars().next().unwrap_or_default().is_numeric() {
      "!DIGITS".to_string()
    } else {
      word.to_lowercase()
    }
  }

  fn _get_features(
    &self,
    i: usize,
    word: &str,
    context: &Vec<String>,
    prev: &str,
    prev2: &str,
  ) -> BTreeMap<String, i32> {
    let mut i = i as i32;
    i += START.len() as i32;

    let mut features: BTreeMap<String, i32> = BTreeMap::new();

    // It's useful to have a constant feature, which acts sort of like a prior
    features.insert("bias".to_string(), 1);

    let i_suffix = &word[word.len().saturating_sub(3)..];
    let i_pref1 = &word[0..1];
    let i_minus_1_tag = prev;
    let i_minus_2_tag = prev2;
    let i_tag_i_minus_2_tag = format!("{} {}", prev, prev2);
    let i_word = &context[i as usize];
    let i_minus_1_tag_i_word = format!("{} {}", prev, context[i as usize]);
    let i_minus_1_word = &context[i as usize - 1];
    let i_minus_1_suffix = &context[i as usize - 1][safe_sub(context[i as usize - 1].len(), 3)..];
    let i_minus_2_word = &context[i as usize - 2];
    let i_plus_1_word = &context[i as usize + 1];
    let i_plus_1_suffix = &context[i as usize + 1][safe_sub(context[i as usize + 1].len(), 3)..];
    let i_plus_2_word = &context[i as usize + 2];

    add(&mut features, "i suffix", i_suffix);
    add(&mut features, "i pref1", i_pref1);
    add(&mut features, "i-1 tag", i_minus_1_tag);
    add(&mut features, "i-2 tag", i_minus_2_tag);
    add(&mut features, "i tag+i-2 tag", &i_tag_i_minus_2_tag);
    add(&mut features, "i word", i_word);
    add(&mut features, "i-1 tag+i word", &i_minus_1_tag_i_word);
    add(&mut features, "i-1 word", i_minus_1_word);
    add(&mut features, "i-1 suffix", i_minus_1_suffix);
    add(&mut features, "i-2 word", i_minus_2_word);
    add(&mut features, "i+1 word", i_plus_1_word);
    add(&mut features, "i+1 suffix", i_plus_1_suffix);
    add(&mut features, "i+2 word", i_plus_2_word);

    fn add(features: &mut BTreeMap<String, i32>, name: &str, args: &str) {
      *features.entry(format!("{} {}", name, args)).or_insert(0) += 1;
    }

    fn safe_sub(a: usize, b: usize) -> usize {
      // performs a - b, 0 if b > a, to sim python's negative index
      if b > a {
        0
      } else {
        a - b
      }
    }

    features
  }
}

pub(crate) fn match_pos(nltk_pos: &str, amepd_pos: &str) -> bool {
  match amepd_pos {
    "adv" if nltk_pos == "RB" || nltk_pos == "RBR" || nltk_pos == "RBS" => true,
    "prep" if nltk_pos == "IN" => true,
    "verb" if &nltk_pos[..2.min(nltk_pos.len())] == "VB" => true,
    "noun" if &nltk_pos[..2.min(nltk_pos.len())] == "NN" => true,
    "num" if nltk_pos == "CD" => true,
    amepd
      if &amepd[..3.min(amepd.len())] == "adj" && &nltk_pos[..2.min(nltk_pos.len())] == "JJ" =>
    {
      true
    }
    "pron" if &nltk_pos[..3.min(amepd_pos.len())] == "PRP" => true,
    "conj" if nltk_pos == "CC" => true,
    "det" if &nltk_pos[nltk_pos.len().saturating_sub(3)..] == "DT" => true,
    "verb@past" if nltk_pos == "VBD" || nltk_pos == "VBN" => true,
    "intj" if nltk_pos == "UH" => true,
    _ => false,
  }
}
