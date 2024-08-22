use std::hash::Hash;

use bimap::BiMap;

pub(crate) const N_ENC_LAYER: usize = 2;
pub(crate) const N_DEC_LAYER: usize = 2;
pub(crate) const N_HEAD: usize = 4;

pub(crate) const SPECIAL_TOKENS: [&str; 3] = ["<pad>", "<sos>", "<eos>"];
pub(crate) const SPECIAL_LEN: usize = SPECIAL_TOKENS.len();

pub(crate) const PAD_IDX: usize = 0;
pub const SOS_IDX: usize = 1;
pub const EOS_IDX: usize = 2;

pub(crate) const MAX_LEN: usize = 32;

pub(crate) const EN_ALPHABET: [char; 27] = [
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
  't', 'u', 'v', 'w', 'x', 'y', 'z', '\'',
];

pub(crate) const EN_PHONEME: [&str; 71] = [
  "aa0", "aa1", "aa2", "ae0", "ae1", "ae2", "ah0", "ah1", "ah2", "ao0", "ao1", "ao2", "aw0", "aw1",
  "aw2", "ax", "axr", "ay0", "ay1", "ay2", "b", "ch", "d", "dh", "eh0", "eh1", "eh2", "er0", "er1",
  "er2", "ey0", "ey1", "ey2", "f", "g", "hh", "ih0", "ih1", "ih2", "iy0", "iy1", "iy2", "jh", "k",
  "l", "m", "n", "ng", "ow0", "ow1", "ow2", "oy0", "oy1", "oy2", "p", "r", "s", "sh", "t", "th",
  "uh0", "uh1", "uh2", "uw0", "uw1", "uw2", "v", "w", "y", "z", "zh",
];

pub(crate) const D_EN_ALPHABET: usize = SPECIAL_TOKENS.len() + EN_ALPHABET.len();
pub(crate) const D_EN_PHONEME: usize = SPECIAL_TOKENS.len() + EN_PHONEME.len();

pub(crate) const FR_ALPHABET: [char; 45] = [
  '\'', '-', '.', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
  'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'à', 'â', 'ã', 'ç', 'è', 'é', 'ê', 'ë', 'î',
  'ï', 'ñ', 'ô', 'ö', 'ù', 'û', 'ü',
];

pub(crate) const FR_PHONEME: [&str; 31] = [
  "@", "^", "a", "b", "cinq", "d", "deux", "e", "f", "g", "huit", "i", "j", "k", "l", "m", "n",
  "neuf", "o", "p", "r", "s", "t", "to", "u", "un", "v", "w", "x", "y", "z",
];

pub(crate) const D_FR_ALPHABET: usize = SPECIAL_TOKENS.len() + FR_ALPHABET.len();
pub(crate) const D_FR_PHONEME: usize = SPECIAL_TOKENS.len() + FR_PHONEME.len();

pub(crate) const DE_ALPHABET: [char; 30] = [
  '#', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
  's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'ö', 'ü',
];

pub(crate) const DE_PHONEME: [&str; 68] = [
  "#0", "#1", "$1", "&0", "&1", ")0", ")1", "+", "/0", "/1", "=", "@0", "^1", "_", "a0", "a1", "b",
  "b0", "b1", "c0", "d", "drei0", "e0", "e1", "eins1", "f", "g", "h", "i0", "i1", "j", "k", "l",
  "m", "n", "null0", "null1", "o0", "o1", "p", "q0", "q1", "r", "s", "sechs1", "t", "u0", "u1",
  "v", "v1", "vier0", "w", "w0", "w1", "x", "x0", "x1", "y0", "y1", "z", "zwei0", "zwei1", "{0",
  "{1", "|0", "|1", "~0", "~1",
];

pub(crate) const D_DE_ALPHABET: usize = SPECIAL_TOKENS.len() + DE_ALPHABET.len();
pub(crate) const D_DE_PHONEME: usize = SPECIAL_TOKENS.len() + DE_PHONEME.len();

pub enum LANG {
  EN,
  FR,
  DE,
}

impl Default for LANG {
  fn default() -> Self {
    LANG::EN
  }
}

fn to_bimap<T: Copy + Eq + Hash>(slice: &[T]) -> BiMap<T, usize> {
  slice
    .iter()
    .enumerate()
    .map(|(i, &v)| (v, i + SPECIAL_LEN))
    .collect()
}

// the config should contain:
// - the language
// - the alphabet of the language
// - the phoneme of the language
// - the dimension of the alphabet
// - the dimension of the phoneme
pub struct G2PConfig {
  pub lang: LANG,
  pub alphabet: BiMap<char, usize>,
  pub phoneme: BiMap<&'static str, usize>,
  pub d_alphabet: usize,
  pub d_phoneme: usize,
}

impl G2PConfig {
  pub fn new(lang: LANG) -> Self {
    match lang {
      LANG::EN => Self {
        lang,
        alphabet: to_bimap(&EN_ALPHABET),
        phoneme: to_bimap(&EN_PHONEME),
        d_alphabet: D_EN_ALPHABET,
        d_phoneme: D_EN_PHONEME,
      },
      LANG::FR => Self {
        lang,
        alphabet: to_bimap(&FR_ALPHABET),
        phoneme: to_bimap(&FR_PHONEME),
        d_alphabet: D_FR_ALPHABET,
        d_phoneme: D_FR_PHONEME,
      },
      LANG::DE => Self {
        lang,
        alphabet: to_bimap(&DE_ALPHABET),
        phoneme: to_bimap(&DE_PHONEME),
        d_alphabet: D_DE_ALPHABET,
        d_phoneme: D_DE_PHONEME,
      },
    }
  }
}

impl Default for G2PConfig {
  fn default() -> Self {
    G2PConfig::new(LANG::EN)
  }
}
