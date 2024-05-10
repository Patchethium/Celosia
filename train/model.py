import torch
from torch import nn, Tensor
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, d_model: int) -> None:
        super(Attention, self).__init__()
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)
        self.scale = d_model ** -0.5

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = self.dropout(attn)
        attn = F.softmax(attn, dim=-1)
        return self.wo(torch.matmul(attn, v)), attn

class G2P(nn.Module):
    def __init__(self, d_model: int, d_src: int, d_tgt: int, tf_ratio: float) -> None:
        super(G2P, self).__init__()
        # encoder
        self.enc_emb = nn.Embedding(d_src, d_model)
        self.enc_rnn = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
        self.c_post = nn.Linear(2 * d_model, d_model)
        self.h_post = nn.Linear(2 * d_model, d_model)
        self.enc_post = nn.Linear(2 * d_model, d_model)

        # decoder
        self.dec_emb = nn.Embedding(d_tgt, d_model)
        self.dec_rnn = nn.LSTM(2 * d_model, d_model, batch_first=True)
        self.dec_post = nn.Linear(d_model, d_tgt)

        # attention
        self.attention = Attention(d_model)
        self.dropout = nn.Dropout(0.1)

        self.tf_ratio = tf_ratio

        self.d_tgt = d_tgt


    def forward(self, src: Tensor, tgt: Tensor) -> tuple[Tensor, Tensor]:
        """
        src: [B,T_src]
        tgt: [B,T_tgt]
        """
        B, T_src = src.size()
        B, T_tgt = tgt.size()
        # encoder
        enc_emb = self.enc_emb(src)
        enc_emb = self.dropout(enc_emb)
        enc_out, (h, c) = self.enc_rnn(enc_emb)
        enc_out = self.enc_post(enc_out) # 2N -> N
        h = torch.cat([h[0,:,:], h[1,:,:]], dim=-1)
        h = torch.tanh(self.h_post(h)).unsqueeze(0)
        c = torch.cat([c[0,:,:], c[1,:,:]], dim=-1)
        c = torch.tanh(self.c_post(c)).unsqueeze(0)

        # tf = torch.rand(1).item() < self.tf_ratio
        tf = self.training and torch.rand(1).item() < self.tf_ratio
        dec_emb = self.dec_emb(tgt)
        dec_emb = self.dropout(dec_emb)

        if tf:
            # if teacher forcing, we calculate the attention all at once
            attn, score = self.attention(dec_emb, enc_out, enc_out)
            dec_in = torch.cat([dec_emb, attn], dim=-1)
            dec_out, _ = self.dec_rnn(dec_in, (h, c))
            out = self.dec_post(dec_out)
            return score, out
        else:
            # if not, we do it one by one
            res = torch.empty(B, T_tgt, self.d_tgt)
            attn_res = torch.empty(B, T_tgt, T_src)
            prev_emb = self.dec_emb(tgt[:,0]).unsqueeze(1) # [B,1,d_tgt]
            for i in range(T_tgt):
                attn, score = self.attention(prev_emb, enc_out, enc_out)
                dec_in = torch.cat([prev_emb, attn], dim=-1)
                dec_out, (h, c) = self.dec_rnn(dec_in, (h, c))
                out = self.dec_post(dec_out)
                prev_emb = self.dec_emb(out.argmax(dim=-1))
                res[:,i,:] = out.squeeze(1)
                attn_res[:,i,:] = score.squeeze(1)
            return attn_res, res


# simple test code
if __name__ == "__main__":
    model = G2P(128, 100, 100, 0.0)
    src = torch.randint(0, 100, (32, 10))
    tgt = torch.randint(0, 100, (32, 10))
    out = model(src, tgt)
    print(out.size())
    model.tf_ratio = 1.0
    out = model(src, tgt)
    print(out.size())