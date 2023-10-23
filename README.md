<div align="center">

# Anthelia

**Fast and accurate phonemizer, built in Rust**

</div>

`Anthelia` (ae0 n th iy1 l iy0 ax) is a Rust phonemizer<sup>[1](#terminology)</sup> that focuses on turning a sentence of natural language into its phoneme transcript automatically. Supports English (amepd), Japanese (romaji), Mandarin (pinyin) and French (prosodylab), with language-specific data (stress, accent and tones).

**🚧 WIP, DO NOT USE 🚧**

## Overview

#### English

1. Look up words in `amepd` for spelling and stress.
2. For words that have multiple spellings, use the POS tag provided by `amepd` and a `Averaged Perceptron Tagger` to disambiguate them.
3. For OOV (out-of-vocabulary) words, predict the spelling with a g2p<sup>[2](#terminology)</sup> model.

#### Japanese

1. Retrieve the full context label from the text with `openjtalk-rs`.
2. Parse the context label, retrieve accent phrases and their mora boundary information.
3. We ignore OOV words for Mandarin Character's UTF code doesn't have anything to do with spelling.

#### Mandarin Chinese

1. Segment text with `jieba-rs` into words.
2. Look up words in `CC-CEDICT` for pinyin and tones.
3. For words that have multiple spellings, use a CRF model to disambiguate them.
4. We ignore OOV words for Japanese Kanji's UTF code doesn't have anything to do with spelling.

#### French

French generally doesn't have the "one word, multiple spelling" problem that is common in the languages above, nor does it have segmentation problems.
1. Look up words in prosodylab's dictionary.
2. For OOV words, predict the spelling with a g2p<sup>[2](#terminology)</sup> model.

#### Spanish

We use a rule-based automaton to predict Spanish.

## Motivation

We have alternatives like `espeak-ng`, but they lack some essential features, such as pitch accent prediction in Japanese and homograph disambiguation in Chinese. When actually building a production-ready TTS application like `Anthe`, we have to build our wheel, in which we design different approaches for each language. This results in much less flexibility but also better accuracy at an acceptable size.

## License

Dual-licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.

## Terminology

1. `Phonemize` here refers to the procedure of transforming one or more word(s) into a phoneme sequence

```
"hello, world" -> "hh ax l ow1 _ w er1 l d" # Yes
"world" -> "w er1 l d" # Yes
```
2. `G2P`/`g2p` here refers to the procedure of transforming one single word into a phoneme sequence

```
"hello, world" -> "hh ax l ow1 _ w er1 l d" # No
"world" -> "w er1 l d" # Yes
```