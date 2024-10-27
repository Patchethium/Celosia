import torch
from model import G2P
from omegaconf import OmegaConf
import argparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(lang: str, checkpoint: str, word: str):
    conf = OmegaConf.load(f"./config/{lang}.yaml")
    alphabets = conf.specials + conf.alphabets
    phonemes = conf.specials + conf.phonemes
    model = G2P(
        conf.d_model,
        conf.d_special + conf.d_alphabet,
        conf.d_special + conf.d_phoneme,
        conf.n_layers,
        conf.n_heads,
        conf.d_ffn,
        conf.max_len,
        conf.dropout
    )
    model.to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    word = word.lower()
    word_indices = [alphabets.index(c) for c in word]
    word_indices = [conf.sos_idx] + word_indices + [conf.eos_idx]
    word_indices = torch.tensor(word_indices).unsqueeze(0).to(DEVICE)
    ph = model.inference(word_indices, 100, conf.sos_idx, conf.eos_idx)
    ph = torch.argmax(ph, dim=-1)
    ph = ph.squeeze().cpu().tolist()
    i = 0
    while ph[i] != conf.eos_idx:
        i += 1
    ph = ph[:i]
    predict = " ".join([phonemes[int(i)] for i in ph])
    print(predict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang")
    parser.add_argument("checkpoint")
    parser.add_argument("word")
    args = parser.parse_args()
    main(args.lang, args.checkpoint, args.word)