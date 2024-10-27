use anyhow::{Context, Result};
use lazy_static::lazy_static;
use regex::Regex;
use std::collections::HashMap;

// transform a number into English words.
// lazy initialize a hash map of common words.
lazy_static! {
  static ref NUMMAP: HashMap<u32, &'static str> = {
    let mut m = HashMap::new();
    m.insert(0, "zero");
    m.insert(1, "one");
    m.insert(2, "two");
    m.insert(3, "three");
    m.insert(4, "four");
    m.insert(5, "five");
    m.insert(6, "six");
    m.insert(7, "seven");
    m.insert(8, "eight");
    m.insert(9, "nine");
    m.insert(10, "ten");
    m.insert(11, "eleven");
    m.insert(12, "twelve");
    m.insert(13, "thirteen");
    m.insert(14, "fourteen");
    m.insert(15, "fifteen");
    m.insert(16, "sixteen");
    m.insert(17, "seventeen");
    m.insert(18, "eighteen");
    m.insert(19, "nineteen");
    m.insert(20, "twenty");
    m.insert(30, "thirty");
    m.insert(40, "forty");
    m.insert(50, "fifty");
    m.insert(60, "sixty");
    m.insert(70, "seventy");
    m.insert(80, "eighty");
    m.insert(90, "ninety");
    m.insert(100, "hundred");
    m.insert(1000, "thousand");
    m
  };
}

/// Spells out numbers one by one.
/// ```text
/// 12345 -> [one, two, three, four, five]
/// ```
pub fn spell_as_is(num: i64) -> Vec<&'static str> {
  let mut result = Vec::new();
  if num < 0 {
    result.push("minus");
  }
  let num = num.abs() as u64;
  num
    .to_string()
    .chars()
    .map(|c| *NUMMAP.get(&c.to_digit(10).unwrap()).unwrap())
    .collect()
}

/// Spells out numbers as digits.
/// ```text
/// 12345 -> [twelve, thousand, three, hundred, forty, five]
/// ```
pub fn spell_as_digit(num: i64) -> Vec<&'static str> {
  let mut result = Vec::new();
  if num < 0 {
    result.push("minus");
  }
  let mut num = num.abs() as u32;
  while num > 0 {
    match num {
      1000..=9999 => {
        let thousand = num / 1000;
        result.push(*NUMMAP.get(&thousand).unwrap());
        result.push(*NUMMAP.get(&1000).unwrap());
        num %= 1000;
      }
      100..=999 => {
        let hundred = num / 100;
        result.push(*NUMMAP.get(&hundred).unwrap());
        result.push(*NUMMAP.get(&100).unwrap());
        num %= 100;
      }
      20..=99 => {
        let ten = num / 10;
        result.push(*NUMMAP.get(&(ten * 10)).unwrap());
        num %= 10;
      }
      1..=19 => {
        result.push(*NUMMAP.get(&num).unwrap());
        num = 0;
      }
      // for nums larger than 9999, spell one by one
      _ => {
        result.extend(spell_as_is(num as i64));
        num = 0;
      }
    }
  }
  result
}

/// Normalize a number word.
///
/// This function recognizes
/// - Phone numbers
/// - Float numbers (i.e. -287.5)
/// - Normal demical numbers (i.e. 123, -123)
///   - If the number is larger than 9999, it will be spelled one by one,
///     as I think it's closer to the way people speak
/// Any other number will be split into digits and spelled one by one.
pub fn get_num(word: &str) -> Result<Vec<&str>> {
  let mut split = Vec::new();
  // check if it's a phone number
  let phone_number_re = Regex::new(r"^[0-9]+\-[0-9]+\-[0-9]+$")?;
  let float_number_re = Regex::new(r"^[\-0-9]+\.[0-9]+$")?;
  let single_number_re = Regex::new(r"^\-{0,1}[0-9]+$")?;
  let number_re = Regex::new(r"[0-9]+")?;
  match word {
    // phone number, spell each part one by one
    _ if phone_number_re.is_match(word) => {
      for captures in number_re.captures_iter(word) {
        if let Some(num) = captures.get(0) {
          split.extend(spell_as_is(num.as_str().parse::<i64>()?));
        }
      }
    }
    // float number, split by the dot, spell the former as digit, the latter one by one
    _ if float_number_re.is_match(word) => {
      let (left, right) = word.split_at(
        word
          .find('.')
          .context(format!("No dot in the word {:?}", word))?,
      );
      let left = left.parse::<i64>()?;
      split.extend(spell_as_digit(left));
      // the dot
      split.push("point");
      // the right part
      split.extend(spell_as_is(right[1..].parse::<i64>()?));
    }
    // normal number, spell as digit
    // be aware that if the number is too big, it will fall back to be spoken one by one
    _ if single_number_re.is_match(word) => {
      let num = word.parse::<i64>()?;
      split.extend(spell_as_digit(num));
    }
    // unknown, collect every number and spell them one by one
    _ => {
      for captures in number_re.captures_iter(word) {
        if let Some(num) = captures.get(0) {
          split.extend(spell_as_is(num.as_str().parse::<i64>()?));
        }
      }
    }
  }
  if split.is_empty() {
    return Err(anyhow::anyhow!("No number found in the word {:?}", word));
  }
  Ok(split)
}
