# Pack your Dictionary, G2P model and POS tagger

This crate help you pack the assets for `celosia`. Apparently each language needs a different set of assets.

There's usually no need to pack them yourself, this crate is here for the maintainers and a prof that a user can reproduce the same assets.

## English

English needs a dictionary from `amped`, a G2P model from the `g2p` sub crate and a POS tagger from the `nltk` python package.

### Dictionary

You can download the dictionary from [amepd's Github Page](https://github.com/rhdunn/amepd). Clone the repo and copy the `cmudict` file.

### G2P model

You can train the g2p model with [train](/train/README.md) python script, or download from releases.

### POS tagger

You can get the POS tagger from `nltk`'s [data](https://github.com/nltk/nltk_data), it's located at [here](https://github.com/nltk/nltk_data/blob/gh-pages/packages/taggers/averaged_perceptron_tagger_eng.zip). Be noted that we're using the "average**d** perceptron tagger", not the "average perceptron tagger".

### Pack

Put them in a directory, they should look like this:

```text
cmudict # the dictionary
en.npz # the G2P model
averaged_perceptron_tagger.pickle # the POS tagger
```

Then run:

```bash
cargo run --release -- --lang en --assets /path/to/your/directory
```

And you'll get a `en.pack.zst` file in the same directory, or you can specify it with `--out` or `-o`.

## Japanese

UnImplemented

## Chinese

UnImplemented