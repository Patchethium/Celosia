import torch
from torch import nn, Tensor
from torch.nn import functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return self.pe[:, : x.size(1)].detach()


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(LearnedPositionalEncoding, self).__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.pos_emb(torch.arange(x.size(1), device=x.device)).unsqueeze(0)


class MHSA(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(MHSA, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        B, S_q, _ = q.size()
        B, S_k, _ = k.size()
        B, S_v, _ = v.size()
        q = self.q(q).view(B, S_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(k).view(B, S_k, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(v).view(B, S_v, self.n_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = (
            torch.matmul(scores, v)
            .transpose(1, 2)
            .contiguous()
            .view(B, S_q, self.d_model)
        )
        return self.o(out)


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout):
        super(FFN, self).__init__()
        self.l1 = nn.Linear(d_model, d_ffn)
        self.l2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.l2(self.dropout(F.relu(self.l1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, dropout):
        super(EncoderLayer, self).__init__()
        self.mhsa = MHSA(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, d_ffn, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.ln1(x + self.mhsa(x, x, x, mask))
        x = self.ln2(x + self.ffn(x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn, dropout):
        super(DecoderLayer, self).__init__()
        self.mhsa1 = MHSA(d_model, n_heads, dropout)
        self.mhsa2 = MHSA(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, d_ffn, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, enc: Tensor, mask: Tensor, mask_enc: Tensor) -> Tensor:
        x = self.ln1(x + self.mhsa1(x, x, x, mask))
        x = self.ln2(x + self.mhsa2(x, enc, enc, mask_enc))
        x = self.ln3(x + self.ffn(x))
        return x


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ffn, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)]
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ffn, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ffn, dropout) for _ in range(n_layers)]
        )

    def forward(self, x: Tensor, enc: Tensor, mask: Tensor, mask_enc: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, enc, mask, mask_enc)
        return x


class G2P(nn.Module):
    def __init__(
        self,
        d_model,
        d_alphabet,
        d_phoneme,
        n_layers,
        n_heads,
        d_ffn,
        dropout,
        pad_idx=0,
    ):
        super(G2P, self).__init__()
        self.d_model = d_model
        self.d_alphabet = d_alphabet
        self.d_phoneme = d_phoneme
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pad_idx = pad_idx

        self.enc_pos_emb = LearnedPositionalEncoding(d_model)
        self.dec_pos_emb = LearnedPositionalEncoding(d_model)
        self.enc_pos_norm = nn.LayerNorm(d_model)
        self.dec_pos_norm = nn.LayerNorm(d_model)

        self.src_emb = nn.Embedding(d_alphabet, d_model)
        self.tgt_emb = nn.Embedding(d_phoneme, d_model)

        self.encoder = Encoder(d_model, n_layers, n_heads, d_ffn, dropout)
        self.decoder = Decoder(d_model, n_layers, n_heads, d_ffn, dropout)

        self.fc = nn.Linear(d_model, d_phoneme)

    def get_src_mask(self, src: Tensor, src_pad_idx: int) -> Tensor:
        mask = src != src_pad_idx
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)  # [B, S, S]
        mask = mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)  # [B, H, S, S]
        return mask

    def get_tgt_mask(self, tgt: Tensor, tgt_pad_idx: int) -> Tensor:
        pad_mask = tgt != tgt_pad_idx
        pad_mask = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)  # [B, S, S]
        casual_mask = (
            torch.tril(torch.ones(tgt.size(1), tgt.size(1))).bool().to(tgt.device)
        )
        tgt_mask = pad_mask & casual_mask  # [B, S, S]
        # expand to heads
        tgt_mask = tgt_mask.unsqueeze(1).expand(
            -1, self.n_heads, -1, -1
        )  # [B, H, S, S]
        return tgt_mask

    def get_cross_mask(
        self, src: Tensor, tgt: Tensor, src_pad_idx: int, tgt_pad_idx: int
    ) -> Tensor:
        # in cross attention, we mask the padding of source and padding of target
        src_mask = src != src_pad_idx  # [B, S_src]
        tgt_mask = tgt != tgt_pad_idx  # [B, S_tgt]
        mask = src_mask.unsqueeze(1) & tgt_mask.unsqueeze(2)  # [B, S_src, S_tgt]
        mask = mask.unsqueeze(1).expand(
            -1, self.n_heads, -1, -1
        )  # [B, H, S_src, S_tgt]
        return mask

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src_mask = self.get_src_mask(src, self.pad_idx)
        tgt_mask = self.get_tgt_mask(tgt, self.pad_idx)
        cross_mask = self.get_cross_mask(src, tgt, self.pad_idx, self.pad_idx)

        src = self.src_emb(src)
        tgt = self.tgt_emb(tgt)

        src = src + self.enc_pos_emb(src)
        tgt = tgt + self.dec_pos_emb(tgt)

        src = self.enc_pos_norm(src)
        tgt = self.dec_pos_norm(tgt)

        enc = self.encoder(src, src_mask)
        dec = self.decoder(tgt, enc, tgt_mask, cross_mask)

        return self.fc(dec)

    def inference(
        self, src: Tensor, max_len: int, sos_idx: int, eos_idx: int
    ) -> Tensor:
        src_mask = self.get_src_mask(src, self.pad_idx)
        src_emb = self.src_emb(src)
        src_emb = src_emb + self.enc_pos_emb(src_emb)
        src_emb = self.enc_pos_norm(src_emb)
        enc = self.encoder(src_emb, src_mask)

        # [B, max_len], filled with pad_idx
        context = torch.full((src.shape[0], max_len), self.pad_idx, device=src.device)
        context[:, 0] = sos_idx
        logits = torch.zeros(src.shape[0], max_len, self.d_phoneme, device=src.device)
        for t in range(1, max_len):
            local_context = context[:, :t]
            local_emb = self.tgt_emb(local_context)
            local_emb = local_emb + self.dec_pos_emb(local_emb)
            local_emb = self.dec_pos_norm(local_emb)
            cross_mask = self.get_cross_mask(
                src, local_context, self.pad_idx, self.pad_idx
            )
            dec = self.decoder(local_emb, enc, None, cross_mask)
            out = self.fc(dec)[:, -1]
            logits[:, t - 1] = out
            context[:, t] = out.argmax(dim=-1)
            if (context[:, t] == eos_idx).all():
                break
        return logits


# simple test code
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = G2P(512, 30, 30, 6, 8, 2048, 0.1).to(device)
    src = torch.randint(0, 30, (32, 10)).to(device)
    tgt = torch.randint(0, 30, (32, 10)).to(device)
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
        out = model.inference(src, 10, 1, 2)
    print(f"Inference time: {(time.time()-start) / 100}")
