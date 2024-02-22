"""
Simple s2s gru with attention 
"""

import math
from random import random

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class Encoder(nn.Module):
    def __init__(self, d_alphabet: int, d_model: int):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(d_alphabet, d_model)
        self.rnn = nn.GRU(d_model, d_model, bidirectional=True, batch_first=True)
        self.h_post = nn.Linear(2 * d_model, d_model)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: Tensor):
        emb = self.dropout(self.emb(x))
        x, h = self.rnn(emb)
        h = F.tanh(
            self.h_post(torch.cat([h[-2, :, :], h[-1, :, :]], dim=-1))
        ).unsqueeze(0)
        return x, h


class Attention(nn.Module):
    """
    Scaled dot product attention
    """
    def __init__(self, d_model: int) -> None:
        super(Attention, self).__init__()
        self.scale = 1 / math.sqrt(d_model)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, q, k, v):
        attn = F.softmax(q @ k.transpose(1, 2) * self.scale, dim=-1)
        attn = self.dropout(attn)
        o = attn @ v
        return attn, o


class Decoder(nn.Module):
    def __init__(self, d_ph: int, d_model: int):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(d_ph, d_model)
        self.rnn = nn.GRU(2 * d_model, d_model, bidirectional=False, batch_first=True)
        self.post = nn.Linear(d_model, d_ph)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: Tensor, attn_o: Tensor, h: Tensor):
        emb = self.dropout(self.emb(x))
        dec_i = torch.cat([emb, attn_o], dim=-1)
        x, h = self.rnn(dec_i, h)
        o = self.post(x)
        return o, x, h


class G2P(nn.Module):
    def __init__(self, d_alphabet: int, d_phoneme: int, d_model: int, tf_ratio: float):
        super(G2P, self).__init__()
        self.d_alphabet = d_alphabet
        self.d_phoneme = d_phoneme
        self.d_model = d_model
        self.tf_ratio = tf_ratio

        self.enc = Encoder(d_alphabet, d_model)
        self.attn = Attention(d_model)
        self.dec = Decoder(d_phoneme, d_model)

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(2 * d_model, d_model)
        self.w_v = nn.Linear(2 * d_model, d_model)

    def forward(self, text: Tensor, tgt: Tensor):
        device = self.enc.emb.weight.device  # a trick to get the device on the fly
        B = text.shape[0]
        L_TEXT = text.shape[1]
        L_PH = tgt.shape[1]

        enc_o, h = self.enc(text)
        # pre-comute k and v for performance
        k = self.w_k(enc_o)
        v = self.w_v(enc_o)

        res = torch.empty([B, L_PH, self.d_phoneme]).to(device)
        attn = torch.empty([B, L_PH, L_TEXT]).to(device)
        attn_slice, attn_o = self.attn(h.transpose(0, 1), k, v)
        o = tgt[:, 0].unsqueeze(1).unsqueeze(1) #[B,1,1]

        for t in range(L_PH):
            attn[:, t, :] = attn_slice.squeeze(1)
            # teacher forcing
            if self.training and random() < self.tf_ratio:
                dec_i = tgt[:, t].unsqueeze(1)
            else:
                dec_i = torch.argmax(o, dim=-1)  # [B, 1, N]
            o, x, h = self.dec.forward(dec_i, attn_o, h)
            attn_slice, attn_o = self.attn(x, k, v)
            res[:, t, :] = o.squeeze(1)

        return attn, res