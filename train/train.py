import argparse
import os
from random import randint

import torch
from omegaconf import OmegaConf
from torch import Tensor, nn, optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter

from torcheval.metrics import WordErrorRate as WER

from datetime import datetime
from model import G2P

DEVICE = None
CONF = None

if not os.path.exists("./ckpt"):
    os.mkdir("./ckpt")


class PhDataset(Dataset):
    def __init__(self, dict_path: str, conf, device) -> None:
        super().__init__()
        alphabets = conf.specials + conf.alphabets
        phonemes = conf.specials + conf.phonemes
        self.data = []
        self.cached = {}
        self.device = device
        f = open(dict_path, "r")
        lines = f.readlines()
        alps = {c: i for i, c in enumerate(alphabets)}
        phs = {p: i for i, p in enumerate(phonemes)}
        for line in lines:
            line = line.rstrip()
            if len(line) < 2:
                continue
            word, phoneme = line.split("  ")
            word_t = torch.LongTensor(
                [conf.sos_idx] + [alps[c] for c in word] + [conf.eos_idx]
            )
            ph_t = torch.LongTensor(
                [conf.sos_idx] + [phs[p] for p in phoneme.split(" ")] + [conf.eos_idx]
            )

            self.data.append((word_t, ph_t))
            # self.data.append((word, phoneme))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple[str, str]:
        # cache the tensor on the device
        cached = self.cached.get(index)
        if cached is not None:
            return cached
        else:
            t = self.data[index]
            t = (t[0].to(self.device), t[1].to(self.device))
            self.cached[index] = t
            return t


def collate_fn(batch: list[tuple[Tensor, Tensor]]):
    # use left padding, flip and flip back after padding
    # word_list = [t[0].flip(dims=[0]) for t in batch]  # list[S_text,]
    # [B,S_text]
    word_batch = pad_sequence(
        [t[0] for t in batch], padding_value=0, batch_first=True
    )

    ph_batch = pad_sequence(
        [t[1] for t in batch], padding_value=0, batch_first=True
    )
    return word_batch, ph_batch


def train(lang: str, device: str):
    global CONF
    global DEVICE

    CONF = OmegaConf.load(f"./config/{lang}.yaml")
    try:
        if torch.cuda.is_available():
            DEVICE = torch.device(f"cuda:{device}")
    except Exception as e:
        print(e)
        DEVICE = torch.device("cpu")

    torch.manual_seed(CONF.seed)

    alphabets = CONF.specials + CONF.alphabets
    phonemes = CONF.specials + CONF.phonemes

    d_model = CONF.d_model
    d_alphabet = CONF.d_special + CONF.d_alphabet
    d_phoneme = CONF.d_special + CONF.d_phoneme

    train_ds = PhDataset(f"./data/{lang}-train.txt", CONF, DEVICE)
    val_ds = PhDataset(f"./data/{lang}-valid.txt", CONF, DEVICE)
    test_ds = PhDataset(f"./data/{lang}-test.txt", CONF, DEVICE)
    train_dl = DataLoader(
        train_ds, batch_size=CONF.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dl = DataLoader(
        val_ds, batch_size=CONF.batch_size, shuffle=False, collate_fn=collate_fn
    )
    model = G2P(d_model, d_alphabet, d_phoneme, CONF.n_layers, CONF.n_heads, CONF.d_ffn, CONF.max_len, CONF.dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=CONF.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=CONF.epochs, eta_min=1e-5)
    loss_func = nn.CrossEntropyLoss(ignore_index=CONF.pad_idx).to(DEVICE)
    writer = SummaryWriter(f"./logs/{lang.upper()}_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    global_step = 0
    padding = (
        torch.LongTensor([CONF.pad_idx]).repeat(CONF.batch_size).unsqueeze(1).to(DEVICE)
    )
    for e in range(CONF.epochs):
        for w, p in train_dl:
            optimizer.zero_grad()
            o = model.forward(w, p)
            # shift the label 1 token left so that the decoder can learn next token prediction
            p = p[:, 1:]
            p = torch.cat([p, padding[: p.shape[0], :]], dim=-1)
            o = o.permute(0, 2, 1)
            loss = loss_func.forward(o, p)
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            writer.add_scalar("Train loss", loss, global_step=global_step)
            global_step += 1
        else:
            with torch.no_grad():
                model.eval()
                # validate into tensorboard
                val_loss = 0
                val_batches = 0
                for w, p in val_dl:
                    o = model.inference(w, p.shape[1], CONF.sos_idx, CONF.eos_idx)
                    # same reason as in training
                    p = p[:, 1:]
                    p = torch.cat([p, padding[: p.shape[0], :]], dim=-1)
                    o = o.permute(0, 2, 1)
                    val_loss += loss_func.forward(o, p)
                    val_batches += 1
                writer.add_scalar(
                    "Val loss", val_loss / val_batches, global_step=global_step
                )

                # test, print a random sample from test set
                print(
                    f"Epoch {e+1}, train loss: {loss}, val loss: {val_loss / val_batches}"
                )
                test_case = test_ds[randint(0, len(test_ds))]
                w, p = test_case
                o = model.inference(w.unsqueeze(0), p.shape[0], CONF.sos_idx, CONF.eos_idx)
                o = torch.argmax(o, dim=-1).squeeze()
                p = p[1:]
                p = torch.cat([p, padding[0, :]], dim=-1)
                word = "".join([alphabets[int(c)] for c in w])
                tgt_phoneme = " ".join([phonemes[int(c)] for c in p])
                pred_phoenme = " ".join([phonemes[int(c)] for c in o])
                print(
                    f"Test result:\n\t Source: {word},\n\t Target: {tgt_phoneme},\n\t Pred:   {pred_phoenme}"
                )

                # WER
                wer = WER()
                for i in range(min(500, len(test_ds))):
                    w, p = test_ds[i]
                    o = model.inference(w.unsqueeze(0), p.shape[0], CONF.sos_idx, CONF.eos_idx)
                    o = torch.argmax(o, dim=-1).squeeze()
                    predict = " ".join([phonemes[int(i)] for i in o[:-2]])
                    ref = " ".join([phonemes[int(i)] for i in p[1:-1]])
                    wer.update(predict, ref)
                writer.add_scalar("Test WER", wer.compute(), global_step=global_step)
                model.train()
            if (e + 1) % 5 == 0:
                torch.save(model.state_dict(), f"./ckpt/{lang}-epoch-{e+1}.pth")
            scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lang", default="en")
    parser.add_argument("-d", "--device", default="0", required=False)
    args = parser.parse_args()
    train(args.lang, args.device)