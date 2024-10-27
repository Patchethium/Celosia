pub static PHONEMIZER_DATA: &[u8] = include_bytes!("data/en.pack.zst");
pub(crate) static PUNCTUATION: &str = r##"[.,!?(){}]"##;

pub static EN_ALPHABET: [char; 27] = [
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
  't', 'u', 'v', 'w', 'x', 'y', 'z', '\'',
];

pub static EN_PHONEME: [&str; 71] = [
  "aa0", "aa1", "aa2", "ae0", "ae1", "ae2", "ah0", "ah1", "ah2", "ao0", "ao1", "ao2", "aw0", "aw1",
  "aw2", "ax", "axr", "ay0", "ay1", "ay2", "b", "ch", "d", "dh", "eh0", "eh1", "eh2", "er0", "er1",
  "er2", "ey0", "ey1", "ey2", "f", "g", "hh", "ih0", "ih1", "ih2", "iy0", "iy1", "iy2", "jh", "k",
  "l", "m", "n", "ng", "ow0", "ow1", "ow2", "oy0", "oy1", "oy2", "p", "r", "s", "sh", "t", "th",
  "uh0", "uh1", "uh2", "uw0", "uw1", "uw2", "v", "w", "y", "z", "zh",
];

pub const UNK_TOKEN: &'static str = "<unk>";
