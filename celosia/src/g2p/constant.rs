pub(crate) const N_ENC_LAYER: usize = 2;
pub(crate) const N_DEC_LAYER: usize = 2;
pub(crate) const N_HEAD: usize = 4;

pub(crate) const SPECIAL_TOKENS: [&str; 3] = ["<pad>", "<sos>", "<eos>"];
pub(crate) const SPECIAL_LEN: usize = SPECIAL_TOKENS.len();

#[allow(dead_code)]
pub(crate) const PAD_IDX: usize = 0;
pub const SOS_IDX: usize = 1;
pub const EOS_IDX: usize = 2;

pub(crate) const MAX_LEN: usize = 32;

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub enum LANG {
  EN,
  FR,
  DE,
}

impl ToString for LANG {
  fn to_string(&self) -> String {
    match self {
      LANG::EN => "en".to_string(),
      LANG::FR => "fr".to_string(),
      LANG::DE => "de".to_string(),
    }
  }
}

impl Default for LANG {
  fn default() -> Self {
    LANG::EN
  }
}
