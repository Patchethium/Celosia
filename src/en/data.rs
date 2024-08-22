pub(crate) const EN_G2P_DATA: &[u8] = include_bytes!("data/en.npz");
pub(crate) const EN_TAGGER_DATA: &[u8] = include_bytes!("data/averaged_perceptron_tagger.pickle");
pub(crate) const EN_NORMAL_DICT: &'static str = include_str!("data/normal.en.json");
pub(crate) const EN_HOMO_DICT: &'static str = include_str!("data/homo.en.json");

pub(crate) const PUNCTUATION: &str = r##"[.,!?(){}]"##;
