"""
beam search, greedy search, nucleus sampling
"""

import torch
from torch import Tensor

from omegaconf import OmegaConf

from model import G2P
from train import PhDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(lang: str, ckpt: str):
    conf = OmegaConf.load(f"./config/{lang}.yaml")
    alphabets = conf.specials + conf.alphabets
    phonemes = conf.specials + conf.phonemes
    eval_ds = PhDataset(f"./data/{lang}-test.txt", conf, DEVICE)
    model = G2P(
        conf.d_model,
        conf.d_special + conf.d_alphabet,
        conf.d_special + conf.d_phoneme,
        conf.n_layers,
        0.0,
    ).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    pass

def greedy_search(model: G2P, word: Tensor, max_len: int, sos_idx: int, eos_idx: int) -> Tensor:
    """
    Greedy search for G2P model
    """
    enc_out, (h, c) = model.forward_enc(word)
    prev = torch.LongTensor([sos_idx]).to(word.device).unsqueeze(0)
    for t in range(max_len):
        pred = model.forward_dec_step((enc_out, (h, c)), prev)
        pred = torch.argmax(pred, dim=-1)
        prev = torch.cat([prev, pred[:, -1].unsqueeze(0)], dim=-1)
        if pred[:, -1] == eos_idx:
            break
    return prev