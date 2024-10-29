# Training G2P Models with PyTorch

This directory contains the code for training a g2p model in English, French and German.

It's a simple seq2seq transformer model, just as described in the paper [Attention is All You Need](https://arxiv.org/abs/1706.03762).

May change from NN to FST (Finite State Transducer) in the future.

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

### German

Prepare `prosodylab`'s dictionary

```bash
wget https://github.com/prosodylab/prosodylab.dictionaries/archive/refs/heads/master.zip

unzip master.zip

mv ./prosodylab.dictionaries-master/de.dict ./de.dict

cd ..

python preprocess.py de
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

### German

```bash
python train.py de
```

## Config

You'll find checkpoints in `train/ckpt` and config the training with `train/config`.

## Export

To use the checkpoints into weight files `Celosia` can pack, run:
```bash
python export.py {checkpoint_path} {weight_path}

# example
python export.py ./ckpt/en-ckpt-epoch-5.pth ./ckpt/en.npz 
```

## Evaluate

We randomly sample 2.5% from the dataset as the test set to calculate the Phoneme WER (Shown below as `PER`).

```bash
python eval.py {lang} {checkpoint_path}

# example
python eval.py en ./ckpt/en-epoch-5.pth
```

## Checkpoints

We provide pretrained weights in binary form, stored in fp16 format to save space. The performance of f32 and f16 is exactly the same.

| lang | code | epochs | PER |
| --- | --- | --- | --- |
| English | en | 50 | 8.91% |
| French | fr | 20 | 0.77% |
| German | de | 15 | 0.57% |

### Note:
- This g2p model is intended to be used on OOVs with average length (>3), please use lexicon dictionary for 1~2 character words.
- This model is uncased, which means it doesn't see a difference between `english` and `English`.
- English needs more epochs to converge, while French and German converge faster.