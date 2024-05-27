import torch
from torcheval.metrics import WordErrorRate
from train import PhDataset, G2P
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import argparse
from torch.utils.data import random_split


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(lang: str, checkpoint: str):
    conf = OmegaConf.load(f"./config/{lang}.yaml")
    alphabets = conf.specials + conf.alphabets
    phonemes = conf.specials + conf.phonemes
    ds = PhDataset(f"./data/{lang}.txt", conf)
    _train_ds, eval_ds = random_split(ds, [0.9, 0.1])
    model = G2P(
        conf.d_model,
        conf.d_special + conf.d_alphabet,
        conf.d_special + conf.d_phoneme,
        0.0,
    )
    model.to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    wer = WordErrorRate()
    for w, p in tqdm(eval_ds):
        _, ph = model.forward(w.unsqueeze(0), p.unsqueeze(0))
        ph = torch.argmax(ph, dim=-1)
        predict = " ".join([phonemes[int(i)] for i in ph.squeeze()[:-2]])
        ref = " ".join([phonemes[int(i)] for i in p.squeeze()[1:-1]])
        # print(predict, ref)
        wer.update([predict], [ref])

    print(f"WER: {wer.compute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    main(args.lang, args.checkpoint)