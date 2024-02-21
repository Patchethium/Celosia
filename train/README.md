# Training with PyTorch

This directory contains the code for training a g2p model in English and French (WIP).

It's just a seq2seq gru with attention, don't wanna make things too complex.

## Environment

```bash
pip install -r requirements.txt
```

## Data preparation

```bash
cd train

mkdir data

cd data
```

### English

Prepare the `amepd` dict

```bash
wget https://github.com/rhdunn/amepd/archive/refs/tags/amepd-0.2.zip

unzip amepd-0.2.zip

mv ./amepd-amepd-0.2/cmudict ./en.dict

cd ..

python preprocess.py en
```

### French

Prepare `prosodylab`'s dictionary

```bash
wget https://github.com/prosodylab/prosodylab.dictionaries/archive/refs/heads/master.zip

unzip master.zip

mv ./prosodylab.dictionaries-master/fr.dict ./fr.dict

cd ..

python preprocess.py fr
```

## Training

### English

```bash
python train.py en
```

### French

```bash
python train.py fr
```

## Config

You'll find checkpoints in `train/ckpt` and config the training with `train/config`.

### Note:
- This g2p model is intended to be used on OOVs with average length (3-15), use the lexicon dictionary for other words.
- This model is uncased, which means it won't see a difference between `english` and `English`.
- This model makes attention failure, Anthelia combines repeated phonemes to affiliate it. Help in improving robustness is welcomed.