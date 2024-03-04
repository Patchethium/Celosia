import torch
from torchtext.data.metrics import bleu_score
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
    _train_ds, eval_ds = random_split(ds, [0.99, 0.01])
    model = G2P(
        conf.d_special + conf.d_alphabet,
        conf.d_special + conf.d_phoneme,
        conf.d_model,
        0.0,
    )
    model.to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    predicted = []
    reference = []
    for w, p in tqdm(eval_ds):
        _, ph = model.forward(w.unsqueeze(0), p.unsqueeze(0))
        ph = torch.argmax(ph, dim=-1)
        predict = [phonemes[int(i)] for i in ph.squeeze()[:-2]]
        ref = [[phonemes[int(i)] for i in p.squeeze()[1:-1]]]
        predicted.append(predict)
        reference.append(ref)
    print(predicted[0], reference[0])

    print(f"Bleu Score: {bleu_score(predicted, reference)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    main(args.lang, args.checkpoint)