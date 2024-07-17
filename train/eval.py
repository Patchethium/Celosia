import torch
from torcheval.metrics import WordErrorRate
from train import PhDataset
from model import G2P
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import argparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(lang: str, checkpoint: str):
    conf = OmegaConf.load(f"./config/{lang}.yaml")
    alphabets = conf.specials + conf.alphabets
    phonemes = conf.specials + conf.phonemes
    eval_ds = PhDataset(f"./data/{lang}-test.txt", conf, DEVICE)
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
    wer = WordErrorRate()
    for w, p in tqdm(eval_ds):
        ph = model.inference(w.unsqueeze(0), p.shape[0], conf.sos_idx, conf.eos_idx)
        ph = torch.argmax(ph, dim=-1)
        predict = " ".join([phonemes[int(i)] for i in ph.squeeze()[:-2]])
        ref = " ".join([phonemes[int(i)] for i in p.squeeze()[1:-1]])
        # print(predict, ref)
        wer.update([predict], [ref])

    print(f"PER: {wer.compute()}") # WER calculated on phonemes is called PER

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    main(args.lang, args.checkpoint)