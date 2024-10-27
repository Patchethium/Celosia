# Pack your Dictionary, G2P model and POS tagger

This crate help you pack the assets for `celosia`. Apparently each language needs a different set of assets.

There's usually no need to pack them yourself, this crate is here for the maintainers and a prof that a user can reproduce the same assets.

## English

English needs a dictionary from `amped`, a G2P model from the `g2p` sub crate and a POS tagger from the `nltk` python package.

Put them in a directory, they should look like this:

```text
cmudict # the dictionary
en.npz # the G2P model
average_perceptron_tagger.pickle # the POS tagger
```

Then run:

```bash
cargo run --release -- pack --lang en --dir /path/to/your/directory
```

And you'll get a `en.pack.zst` file in the same directory, or specify it with `--out` or `-o`.

## Japanese

TODO!

## Chinese

TODO!