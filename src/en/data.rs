pub(crate) static EN_G2P_DATA: &[u8] = include_bytes!("data/en.npz");
pub(crate) static EN_TAGGER_DATA: &[u8] = include_bytes!("data/averaged_perceptron_tagger.pickle");
pub(crate) static EN_NORMAL_DICT: &'static str = include_str!("data/normal.en.json");
pub(crate) static EN_HOMO_DICT: &'static str = include_str!("data/homo.en.json");

pub(crate) static PUNCTUATION: &str = r##"[.,!?(){}]"##;
