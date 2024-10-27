use bimap::BiMap;
use std::hash::Hash;

pub(crate) fn to_bimap<T: Copy + Eq + Hash>(slice: &[T], offset: usize) -> BiMap<T, usize> {
  slice
    .iter()
    .enumerate()
    .map(|(i, &v)| (v, i + offset))
    .collect()
}
