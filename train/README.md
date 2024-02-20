# Training with PyTorch

This directory contains the code for training a g2p model in English and French (WIP).

It's just a seq2seq gru with attention, don't wanna make things too complex.

## Environment

```bash
pip install -r requirements.txt
```

### English

Prepare the `amepd` dict

```bash
cd train

mkdir data

cd data

wget https://github.com/rhdunn/amepd/archive/refs/tags/amepd-0.2.zip

unzip amepd-0.2.zip

mv ./amepd-amepd-0.2/cmudict ./amepd.txt

cd ..

python preprocess.py
```

Then you can run the training

```bash
python train.py
```

You'll find checkpoints in `train/ckpt` and config the training with `train/config/en.yaml`.

### French

`WIP`

### Note:
- This g2p model is intended to be used on OOVs with average length (3-15), use the lexicon dictionary for other words.
- This model is uncased, which means it won't see a difference between `english` and `English`.
- This model makes attention failure, Anthelia combines repeated phonemes to affiliate it. Help in improving robustness is welcomed.