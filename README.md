<div align="center">

# Anthelia

**Fast and accurate phonemizer, built in Rust**

</div>

`Anthelia` (ae0 n th iy1 l iy0 ax) is a Rust phonemizer that focuses on turning a sentence of natural language into its phoneme transcript automatically. Supports English (amepd), Japanese (romaji), Mandarin (pinyin) and French (prosodylab), with language-specific data (stress, accent and tones).

**🚧 WIP, DO NOT USE 🚧**

## Overview

#### English

1. Look up words in `amepd` for spelling and stress.
2. For words that have multiple spellings, use the POS tag provided by `amepd` and a `Averaged Perceptron Tagger` to disambiguate them.
3. For out-of-domain vocabulary, predict the spelling with a tiny RNN model.

#### Japanese

1. Retrieve the full context label from the text with `openjtalk-rs`
2. Parse the context label, retrieve accent phrases and mora boundary information

#### Mandarin Chinese

1. Segment text with `jieba-rs` into words.
2. Look up words in `CC-CEDICT` for pinyin and tones.
3. For words that have multiple spellings, use a CRF model trained on Aishell-3 to disambiguate them.
4. We ignore OOV (out-of-vocabulary) in Mandarin.

#### French

French generally doesn't have the "one word, multiple spelling" problem that is common in the languages above, nor does it have segmentation problems.
1. Look up words in prosodylab's dictionary.
2. For OOV, predict the spelling with a tiny RNN model.

## Why?

We have alternatives like `espeak-ng`, but they lack some essential features, such as pitch accent prediction in Japanese and homograph disambiguation in Chinese. When actually building a production-ready TTS application like `Anthe`, we have to build our wheel, in which we design different approaches for each language. This results in much less flexibility but also better accuracy at an acceptable size.

## License

Dual-licensed under the Apache License, Version 2.0 http://www.apache.org/licenses/LICENSE-2.0 or the MIT license http://opensource.org/licenses/MIT, at your option. This file may not be copied, modified, or distributed except according to those terms.