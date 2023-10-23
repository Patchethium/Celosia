pub const ATTN_DIM: usize = 128;
pub const ALPHABET_DIM: usize = 30;
pub const PHONEME_DIM: usize = 74;

pub const PADDING: [&'static str; 3] = ["<pad>", "<sos>", "<eos>"];

pub const AMEPD_PHONE_SET: [&'static str; 74] = [
  "<pad>", "<sos>", "<eos>", "aa0", "aa1", "aa2", "ae0", "ae1", "ae2", "ah0", "ah1", "ah2", "ao0",
  "ao1", "ao2", "aw0", "aw1", "aw2", "ax", "axr", "ay0", "ay1", "ay2", "b", "ch", "d", "dh", "eh0",
  "eh1", "eh2", "er0", "er1", "er2", "ey0", "ey1", "ey2", "f", "g", "hh", "ih0", "ih1", "ih2",
  "iy0", "iy1", "iy2", "jh", "k", "l", "m", "n", "ng", "ow0", "ow1", "ow2", "oy0", "oy1", "oy2",
  "p", "r", "s", "sh", "t", "th", "uh0", "uh1", "uh2", "uw0", "uw1", "uw2", "v", "w", "y", "z",
  "zh",
];

pub const ALPHABETS: [&str; 29] = [
  "<pad>", "<sos>", "<eos>", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
  "t", "u", "v", "w", "x", "y", "z",
];

pub const CONSONANTS: [&'static str; 24] = [
  "b", "ch", "d", "dh", "f", "g", "hh", "jh", "k", "l", "m", "n", "ng", "p", "r", "s", "sh", "t",
  "th", "v", "w", "y", "z", "zh",
];

pub const VOWEL: [&'static str; 17] = [
  "aa", "ae", "ah", "ao", "aw", "ax", "axr", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh",
  "uw",
];

pub const STRESS: [u8; 3] = [0, 1, 2];

pub const G2P_MODEL_SHAPES: [[usize; 2]; 22] = [
  [30, 128],
  [384, 128],
  [384, 128],
  [1, 384],
  [1, 384],
  [384, 128],
  [384, 128],
  [1, 384],
  [1, 384],
  [128, 256],
  [1, 128],
  [256, 128],
  [1, 256],
  [128, 256],
  [1, 128],
  [74, 128],
  [384, 256],
  [384, 128],
  [1, 384],
  [1, 384],
  [74, 128],
  [1, 74],
];
