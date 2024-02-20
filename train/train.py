from random import randint

import torch
from torch import Tensor, nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard.writer import SummaryWriter

from omegaconf import OmegaConf
from model import G2P

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONF = OmegaConf.load("./config/en.yaml")

alphabets = CONF.specials + CONF.alphabets
phonemes = CONF.specials + CONF.phonemes


class EnDataset(Dataset):
    def __init__(self, dict_path: str) -> None:
        super().__init__()
        self.data = []
        f = open(dict_path, "r")
        lines = f.readlines()
        als = {c: i for i, c in enumerate(alphabets)}
        phs = {p: i for i, p in enumerate(phonemes)}
        for line in lines:
            line = line.rstrip()
            if len(line) < 2:
                continue
            word, phoneme = line.split("  ")
            word_t = torch.LongTensor(
                [CONF.sos_idx] + [als[c] for c in word] + [CONF.eos_idx]
            )
            ph_t = torch.LongTensor(
                [CONF.sos_idx] + [phs[p] for p in phoneme.split(" ")] + [CONF.eos_idx]
            )

            self.data.append((word_t.to(DEVICE), ph_t.to(DEVICE)))
            # self.data.append((word, phoneme))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple[str, str]:
        return self.data[index]


def collate_fn(batch: list[tuple[Tensor, Tensor]]):
    word_batch = pad_sequence(
        [t[0] for t in batch], padding_value=CONF.pad_idx, batch_first=True
    )
    ph_batch = pad_sequence(
        [t[1] for t in batch], padding_value=CONF.pad_idx, batch_first=True
    )
    return word_batch, ph_batch


def train():
    ds = EnDataset("./data/en.txt")
    train_ds, val_ds, test_ds = random_split(ds, [0.9, 0.09, 0.01])
    train_dl = DataLoader(
        train_ds, batch_size=CONF.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_dl = DataLoader(
        val_ds, batch_size=CONF.batch_size, shuffle=False, collate_fn=collate_fn
    )
    model = G2P(CONF).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=CONF.lr)
    scheduler = ExponentialLR(optimizer, gamma=CONF.lr_decay)
    loss_func = nn.CrossEntropyLoss(ignore_index=CONF.pad_idx)
    writer = SummaryWriter()
    global_step = 0
    for e in range(CONF.epochs):
        for w, p in train_dl:
            optimizer.zero_grad()
            _, o = model.forward(w, p)
            o = o.permute(0, 2, 1)
            loss = loss_func.forward(o, p)
            loss.backward()
            optimizer.step()
            writer.add_scalar("Train loss", loss, global_step=global_step)
            global_step += 1
        else:
            with torch.no_grad():
                model.eval()
                # valdate into tensorboard
                val_loss = 0
                val_batches = 0
                for w, p in val_dl:
                    _, o = model.forward(w, p)
                    o = o.permute(0, 2, 1)
                    val_loss += loss_func.forward(o, p)
                    val_batches += 1
                writer.add_scalar(
                    "Val loss", val_loss / val_batches, global_step=global_step
                )

                # test, print a random sample from test set
                print(f"Epoch {e+1}, train loss: {loss}, val loss: {val_loss / val_batches}")
                test_case = test_ds[randint(0, len(test_ds))]
                w, p = test_case
                attn, o = model.forward(w.unsqueeze(0), p.unsqueeze(0))  # [1,S,N]
                writer.add_image(
                    "Attention Matrix",
                    attn.squeeze(),
                    global_step=global_step,
                    dataformats="HW",
                )
                o = torch.argmax(o, dim=-1).squeeze()
                word = "".join([alphabets[int(c)] for c in w])
                tgt_phoneme = " ".join([phonemes[int(c)] for c in p])
                pred_phoenme = " ".join([phonemes[int(c)] for c in o])
                print(
                    f"Test result:\n\t Source: {word},\n\t Target: {tgt_phoneme},\n\t Pred: {pred_phoenme}"
                )

                # calculate bleu score

                model.train()

        torch.save(model.state_dict(), f"./ckpt/en-ckpt-epoch-{e}.pth")
        scheduler.step()


if __name__ == "__main__":
    train()
