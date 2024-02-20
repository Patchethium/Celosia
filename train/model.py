"""
Simple s2s gru with attention 
"""

import math
from random import random

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class G2P(nn.Module):
    def __init__(self, conf):
        super(G2P, self).__init__()
        self.c = conf
        d_model = conf.d_model
        d_alphabet = conf.d_special + conf.d_alphabet
        d_phoneme = conf.d_special + conf.d_phoneme

        self.d_alphabet = conf.d_special + conf.d_alphabet
        self.d_phoneme = conf.d_special + conf.d_phoneme

        # encoder
        self.enc_emb = nn.Embedding(d_alphabet, d_model)
        self.enc_rnn = nn.GRU(d_model, d_model, bidirectional=True, batch_first=True)
        self.enc_h_post = nn.Linear(2 * d_model, d_model)

        # attn
        self.attn_q = nn.Linear(d_model, d_model)
        self.attn_k = nn.Linear(2 * d_model, d_model)
        self.attn_v = nn.Linear(2 * d_model, d_model)
        self.scaling = math.sqrt(d_model)

        # decoder
        self.dec_emb = nn.Embedding(d_phoneme, d_model)
        self.dec_rnn = nn.GRU(
            2 * d_model, d_model, bidirectional=False, batch_first=True
        )
        self.dec_post = nn.Linear(d_model, d_phoneme)

        self.dropout = nn.Dropout(p=self.c.dropout)

    def forward(self, word: Tensor, tgt: Tensor):
        # [B,S_text,] [B,S_phoneme]
        device = self.dec_post.weight.device
        enc_emb = self.dropout(self.enc_emb(word))
        enc_x, h = self.enc_rnn(enc_emb)
        h = F.tanh(
            self.enc_h_post(torch.cat([h[-2, :, :], h[-1, :, :]], dim=-1))
        ).unsqueeze(
            0
        )  # [1,B,N]

        # encoder output doesn't change afterwards, we can pre-compute k and v
        k = self.attn_k(enc_x).transpose(1, 2)
        v = self.attn_v(enc_x)  # [B,S_text,N]

        if self.training:
            tgt_emb = self.dec_emb(tgt)  # [B,S_ph,N]

        attn = F.softmax(
            torch.bmm(h.permute(1, 0, 2), k) / self.scaling, dim=-1
        )  # [B,1,S_text]
        attn = self.dropout(attn)
        attn_o = torch.bmm(attn, v)  # [B,1,N]

        # attn_o = h.permute(1,0,2)  # [B,1,N]

        dec_i = (
            torch.LongTensor([self.c.sos_idx])
            .repeat(tgt.shape[0])
            .unsqueeze(1)
        ).to(device)  # [B,1]

        res = torch.zeros([tgt.shape[0], tgt.shape[1], self.d_phoneme]).to(device)  # [B,S_ph,N]
        attn_res = torch.zeros(
            [tgt.shape[0], tgt.shape[1], word.shape[1]]
        ).to(device)  # [B,S_ph,S_text]

        for t in range(tgt.shape[1]):
            if random() < self.c.tf_ratio and self.training:
                dec_emb = self.dropout(tgt_emb[:, t, :].unsqueeze(1))
            else:
                dec_emb = self.dropout(self.dec_emb(dec_i))
            # decoder
            dec_rnn_i = torch.cat([dec_emb, attn_o], dim=-1)
            dec_x, h = self.dec_rnn(dec_rnn_i, h)
            # forward attention
            q = self.attn_q(h.permute(1, 0, 2))  # [B,1,N]
            attn = F.softmax(torch.bmm(q, k) / self.scaling, dim=-1)  # [B,1,S_text]
            attn_res[:, t, :] = attn.squeeze(1)
            attn_o = torch.bmm(attn, v)  # [B,1,N]
            # update results & hidden
            dec_o = self.dec_post(dec_x)  # [B,1,d_ph]
            res[:, t, :] = dec_o.squeeze(1)
            dec_i = torch.argmax(dec_o, dim=-1)  # [B,1,]
        return attn_res, res
