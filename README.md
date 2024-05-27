<div align="center">

# Camellia

**Fast and accurate phonemizer[^1], built in Rust**

</div>

`Camellia` (pronounced /kəˈmɛliə/ or /kəˈmiːliə/) is a Rust crate that turns a sentence of natural language into its phoneme transcript automatically. It supports English (amepd), Japanese (romaji), Mandarin (pinyin), French and German (prosodylab), with language-specific data (stress, accent and tones).

**🚧 WIP, DO NOT USE 🚧**

## Overview

This section briefly introduces the phonemization process for each language.

#### English

1. Look up words in `amepd` for spelling and stress.
2. For words that have multiple spellings, use the POS tag provided by `amepd` and a `Averaged Perceptron Tagger` to disambiguate them.
3. For OOV (out-of-vocabulary) words, predict the spelling with a g2p[^2] model.

#### Japanese

1. Retrieve the full context label from the text with `openjtalk-rs`.
2. Parse the context label, retrieve accent phrases and their mora boundary information.
4. We ignore OOV words for the UTF code doesn't contain any information of spelling.

#### Mandarin Chinese

1. Segment text with `jieba-rs` into words.
2. Look up words in `CC-CEDICT` for pinyin and tones.
3. For words that have multiple spellings, use a CRF model to disambiguate them.
4. We ignore OOV words for the UTF code doesn't contain any information of spelling.

#### French & German

Thanks for the orthography, French & German generally don't have the disambiguation (i.e. one word, multiple spelling) problem that is commonly seen in the languages above.  

1. Look up words in `CC-CEDICT` for pinyin and tones.
2. Predict the spelling with a g2p[^2] model.

### Languages

If you only use a subset of these languages, you can prune unused languages for the binary size. Some complex languages like Japanese takes some huge size.

## Motivation

We have alternatives like `espeak-ng`, but they lack some essential features, such as pitch accent prediction in Japanese and homograph disambiguation in Chinese. When actually building a production-ready TTS application like `Anthe`, we have to build our wheel, in which we design different approaches for each language. This results in much less flexibility but also better accuracy at an acceptable size increase.

## License

`Camellia` is dual-licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) or the [MIT license](http://opensource.org/licenses/MIT), at your option. This file may not be copied, modified, or distributed except according to those terms.

## About the name

`Camellia` is an open source component of `Anthe` the TTS application, which is still under development and is not published yet.

The name `Anthe` is derived from Saturn XLIX, a very small natural satellite of Saturn. It's named after one of the Alkyonides, this name means flowery.

Therefore, we use flower names for each component of `Anthe`. `Camellia`<sup>[wikipedia](https://en.wikipedia.org/wiki/Camellia)</sup> is a genus of flowering plants in the family Theaceae, native to eastern and southern Asia.

![Camellia](https://upload.wikimedia.org/wikipedia/commons/6/6b/Semi-double_Camelia_cultivar.jpg)

[^1]: `Phonemize` here refers to the procedure of transforming one or more word(s) into a phoneme sequence:  
"hello, world" -> "hh ax l ow1 _ w er1 l d" # Yes  
"world" -> "w er1 l d" # Yes

[^2]: `G2P`/`g2p` here refers to the procedure of transforming one single word into a phoneme sequence:  
"hello, world" -> "hh ax l ow1 _ w er1 l d" # No  
"world" -> "w er1 l d" # Yes