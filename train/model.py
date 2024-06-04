import torch
from torch import nn, Tensor
from torch.nn import functional as F

from random import random


class Attention(nn.Module):
    """
    Attention mechanism
    """

    def __init__(self, d_model: int, dropout: float = 0.3) -> None:
        super(Attention, self).__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        q: [B,T_q,D]
        k: [B,T_k,D]
        v: [B,T_k,D]
        """
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        # attention score
        score = torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        score = self.dropout(score)
        score = F.softmax(score, dim=-1)
        # weighted sum
        out = torch.bmm(score, v)
        return out


class G2P(nn.Module):
    """
    Seq2Seq LSTM with attention
    """

    def __init__(
        self, d_model: int, d_src: int, d_tgt: int, n_layers: int, tf_ratio: float
    ) -> None:
        super(G2P, self).__init__()
        self.d_src = d_src
        self.d_tgt = d_tgt
        self.n_layers = n_layers
        # encoder
        self.enc_emb = nn.Embedding(d_src, d_model)
        self.enc_rnn = nn.LSTM(
            d_model,
            d_model,
            batch_first=True,
            num_layers=n_layers,
            dropout=0.0 if n_layers == 1 else 0.3,
        )
        # attn
        self.attn = Attention(d_model)
        self.attn_post = nn.Linear(2 * d_model, d_model)
        # decoder
        self.dec_emb = nn.Embedding(d_tgt, d_model)
        self.dec_rnn = nn.LSTM(
            d_model,
            d_model,
            batch_first=True,
            num_layers=n_layers,
            dropout=0.0 if n_layers == 1 else 0.3,
        )
        self.post = nn.Linear(d_model, d_tgt)

        self.dropout = nn.Dropout(0.3)
        self.tf_ratio = tf_ratio

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        src: [B,T_src]
        tgt: [B,T_tgt]
        """
        src_emb = self.enc_emb(src)
        src_emb = self.dropout(src_emb)
        _, (h, c) = self.enc_rnn(src_emb)

        tgt_emb = self.dec_emb(tgt)
        tgt_emb = self.dropout(tgt_emb)

        tf = random() < self.tf_ratio and self.training  # teacher forcing
        if tf:
            attn = self.attn(tgt_emb, src_emb, src_emb)
            tgt_emb = torch.cat([tgt_emb, attn], dim=-1)
            tgt_emb = self.attn_post(tgt_emb)
            out, _ = self.dec_rnn(tgt_emb, (h, c))
            out = self.post(out)
        else:
            out = torch.empty([tgt.shape[0], tgt.shape[1], self.d_tgt]).to(src.device)
            prev = tgt_emb[:, 0, :].unsqueeze(1)  # [B,1,D], <sos> token
            for t in range(tgt.shape[1]):
                attn = self.attn(prev, src_emb, src_emb)
                prev = torch.cat([prev, attn], dim=-1)
                prev = self.attn_post(prev)
                out_t, (h, c) = self.dec_rnn(prev, (h, c))
                out_t = self.post(out_t)
                out[:, t, :] = out_t.squeeze(1)
                pred = out_t.argmax(dim=-1)
                prev = self.dropout(self.dec_emb(pred))
        return out



# simple test code
if __name__ == "__main__":
    model = G2P(512, 100, 100, 2, 0.5)
    src = torch.randint(0, 100, (32, 10))
    tgt = torch.randint(0, 100, (32, 10))
    out = model(src, tgt)
    print(out.size())
    model.tf_ratio = 1.0
    out = model(src, tgt)
    print(out.size())
    # compare time with training(non-tf) and inference
    import time

    start = time.time()
    for _ in range(100):
        out = model(src, tgt)
    print(f"Training time: {(time.time()-start) / 100}")
    start = time.time()
    for _ in range(100):
        out = model.inference(src, 10, 1)
    print(f"Inference time: {(time.time()-start) / 100}")
