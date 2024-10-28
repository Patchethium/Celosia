use super::g2p::load_trf;
use anyhow::{Error, Result};
use bimap::BiMap;
use celosia::en::EN_PHONEME;
use celosia::en::{
  HomoDict, NormalDict, PhonemizerData, TaggerClasses, TaggerTagdict, TaggerWeight,
};
use serde_pickle as pickle;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use zstd::Encoder;

fn parse_dict(path: &Path) -> Result<(HomoDict, NormalDict)> {
  let dict_file = File::open(path)?;
  let reader = BufReader::new(dict_file);
  let ph_map =
    EN_PHONEME
      .iter()
      .enumerate()
      .fold(BiMap::<&str, usize>::new(), |mut acc, (i, s)| {
        acc.insert(&s, i);
        acc
      });
  let mut normal = NormalDict::new();
  let mut homo = HomoDict::new();
  for line in reader.lines() {
    if let Ok(line) = line {
      let mut line = line.trim().to_ascii_lowercase().to_string();
      // EOF or empty line
      if line.len() == 0 {
        line.clear();
        continue;
      }
      // strip comments
      if line.starts_with(";;;") {
        line.clear();
        continue;
      }
      // strip inline comments
      let parts = line.split("#").collect::<Vec<_>>();
      if parts.len() > 1 {
        line = parts[0].trim().to_string();
      }
      // parse line
      // the line is like word(tag)  phoneme1 phoneme2 ...
      let parts = line.split("  ").collect::<Vec<_>>();
      if parts.len() != 2 {
        line.clear();
        continue;
      }
      let word = parts[0].to_string();
      let phoneme = parts[1]
        .split_whitespace()
        .filter(|x| ph_map.contains_left(x))
        .map(|x| *ph_map.get_by_left(x).unwrap())
        .collect::<Vec<_>>();
      if word.contains("(") {
        let tags = word.split("(").collect::<Vec<_>>();
        let word = tags[0].to_string();
        let tag = tags[1].split(")").collect::<Vec<_>>()[0].to_string();
        if homo.get(&word).is_none() {
          homo.insert(word.clone(), HashMap::new());
        }
        homo.get_mut(&word).unwrap().insert(tag, phoneme);
      } else {
        normal.insert(word, phoneme);
      }
      line.clear();
    }
  }
  Ok((homo, normal))
}

fn parse_tagger(pickle_path: &Path) -> Result<(TaggerWeight, TaggerTagdict, TaggerClasses)> {
  let f = File::open(pickle_path)?;
  let reader = BufReader::new(f);
  let decoded = pickle::value_from_reader(reader, pickle::DeOptions::new().decode_strings())?;
  let (weights, tagdict, classes) =
    pickle::from_value::<(TaggerWeight, TaggerTagdict, TaggerClasses)>(decoded)?;
  Ok((weights, tagdict, classes))
}

fn read_g2p(g2p_path: &Path) -> Result<Vec<u8>> {
  let f = File::open(&g2p_path)?;
  let mut reader = BufReader::new(f);
  let mut data = Vec::new();
  reader.read_to_end(&mut data)?;
  data.shrink_to_fit();
  Ok(data)
}

pub fn pack_en(path: &str, output: &str) -> Result<()> {
  let root = Path::new(path);
  let output = Path::new(output);
  let dict_path = root.join("cmudict");
  let tagger_path = root.join("average_perceptron_tagger.pickle");
  let g2p_path = root.join("g2p.npz");
  if !(dict_path.is_file() && tagger_path.is_file() && g2p_path.is_file()) {
    return Err(Error::msg("Missing assets"));
  }
  let (homo, normal) = parse_dict(dict_path.as_path())?;
  let (weight, tagdict, classes) = parse_tagger(&tagger_path)?;
  let g2p_data = read_g2p(&g2p_path)?;
  let trf = load_trf(&g2p_data);
  let encoded =
    bitcode::serialize::<PhonemizerData>(&((normal, homo), (weight, tagdict, classes), trf))?;
  let o = File::create(output)?;
  let mut encoder = Encoder::new(o, 0)?;
  encoder.write_all(&encoded)?;
  encoder.finish()?;
  Ok(())
}
