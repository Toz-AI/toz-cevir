"""Töz-Çevir-1 model tanımı: encoder-decoder, 74,1M parametre.

Parametre adları yayınlanan ağırlıkların anahtarlarıyla eşleşmek zorunda; hiçbir şey yeniden adlandırılmamalı.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK, BOS, EOS = 0, 1, 2, 3


@dataclass
class Ayar:
    vocab: int = 32000
    d: int = 512
    enc: int = 10
    dec: int = 6
    head: int = 8
    kv_head: int = 8
    ffn: int = 1408
    dropout: float = 0.1
    max_pos: int = 1024
    rope_taban: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        t = x.float()   # fp16'da taşmasın diye normalizasyon fp32
        t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + self.eps)
        return (t * self.weight.float()).to(x.dtype)


def rope_uygula(x, cos, sin):
    """x: (B, H, L, hd). Qwen tarzı (rotate_half, iç içe değil)."""
    L = x.shape[-2]
    c, s = cos[:L], sin[:L]
    xf = x.float()
    x1, x2 = xf.chunk(2, dim=-1)
    return (xf * c + torch.cat((-x2, x1), dim=-1) * s).to(x.dtype)


class Dikkat(nn.Module):
    def __init__(self, a: Ayar, rope: bool):
        super().__init__()
        self.h, self.kvh, self.hd = a.head, a.kv_head, a.d // a.head
        self.rope = rope
        self.q = nn.Linear(a.d, self.h * self.hd, bias=False)
        self.k = nn.Linear(a.d, self.kvh * self.hd, bias=False)
        self.v = nn.Linear(a.d, self.kvh * self.hd, bias=False)
        self.o = nn.Linear(self.h * self.hd, a.d, bias=False)
        self.qn, self.kn = RMSNorm(self.hd), RMSNorm(self.hd)   # QK-norm (fp16'da logit taşmasını önler)
        self.p = a.dropout

    def forward(self, x, kv=None, maske=None, causal=False, cs=None):
        B, L, _ = x.shape
        kv = x if kv is None else kv
        S = kv.shape[1]
        q = self.qn(self.q(x).view(B, L, self.h, self.hd).transpose(1, 2))
        k = self.kn(self.k(kv).view(B, S, self.kvh, self.hd).transpose(1, 2))
        v = self.v(kv).view(B, S, self.kvh, self.hd).transpose(1, 2)
        if self.rope:   # yalnız self-attention'da; cross-attention'da RoPE yok
            q, k = rope_uygula(q, *cs), rope_uygula(k, *cs)
        if self.kvh != self.h:
            k = k.repeat_interleave(self.h // self.kvh, dim=1)
            v = v.repeat_interleave(self.h // self.kvh, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=maske, is_causal=causal,
                                           dropout_p=self.p if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(B, L, self.h * self.hd))


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w13 = nn.Linear(d, 2 * h, bias=False)   # gate + up tek matriste
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        g, u = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(g) * u)


class EncKatman(nn.Module):
    def __init__(self, a: Ayar):
        super().__init__()
        self.n1, self.att = RMSNorm(a.d), Dikkat(a, rope=True)
        self.n2, self.ffn = RMSNorm(a.d), SwiGLU(a.d, a.ffn)
        self.drop = nn.Dropout(a.dropout)

    def forward(self, x, maske, cs):
        x = x + self.drop(self.att(self.n1(x), maske=maske, cs=cs))
        return x + self.drop(self.ffn(self.n2(x)))


class DecKatman(nn.Module):
    def __init__(self, a: Ayar):
        super().__init__()
        self.n1, self.att = RMSNorm(a.d), Dikkat(a, rope=True)
        self.n2, self.capraz = RMSNorm(a.d), Dikkat(a, rope=False)
        self.n3, self.ffn = RMSNorm(a.d), SwiGLU(a.d, a.ffn)
        self.drop = nn.Dropout(a.dropout)

    def forward(self, x, bellek, maske, cs):
        # Dolgu hep sonda + causal → hedef tarafında maskeye gerek yok (SDPA flash çekirdeğini seçebilir)
        x = x + self.drop(self.att(self.n1(x), causal=True, cs=cs))
        x = x + self.drop(self.capraz(self.n2(x), kv=bellek, maske=maske))
        return x + self.drop(self.ffn(self.n3(x)))


class TozCevir(nn.Module):
    def __init__(self, a: Ayar):
        super().__init__()
        self.a = a
        self.emb = nn.Embedding(a.vocab, a.d)        # encoder girişi + decoder girişi + çıkış: tek matris
        self.enc = nn.ModuleList(EncKatman(a) for _ in range(a.enc))
        self.dec = nn.ModuleList(DecKatman(a) for _ in range(a.dec))
        self.enc_norm, self.dec_norm = RMSNorm(a.d), RMSNorm(a.d)
        hd = a.d // a.head
        inv = 1.0 / a.rope_taban ** (torch.arange(0, hd, 2).float() / hd)
        f = torch.outer(torch.arange(a.max_pos).float(), inv)
        f = torch.cat((f, f), dim=-1)
        self.register_buffer("cos", f.cos(), persistent=False)
        self.register_buffer("sin", f.sin(), persistent=False)
        self._baslat()

    def _baslat(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
        olcek = 0.02 / math.sqrt(2 * (self.a.enc + self.a.dec))
        for ad, p in self.named_parameters():
            if ad.endswith((".o.weight", ".w2.weight")):   # artık dala yazan çıkış projeksiyonları
                nn.init.normal_(p, 0.0, olcek)

    def kodla(self, src, src_maske):
        m = src_maske[:, None, None, :]
        x = self.emb(src)
        for k in self.enc:
            x = k(x, m, (self.cos, self.sin))
        return self.enc_norm(x)

    def coz(self, tgt_in, bellek, src_maske):
        m = src_maske[:, None, None, :]
        x = self.emb(tgt_in)
        for k in self.dec:
            x = k(x, bellek, m, (self.cos, self.sin))
        return self.dec_norm(x)

    def forward(self, src, src_maske, tgt_in, tgt_out, yumusatma: float = 0.0):
        h = self.coz(tgt_in, self.kodla(src, src_maske), src_maske)
        logit = F.linear(h, self.emb.weight)
        return F.cross_entropy(logit.float().view(-1, logit.size(-1)), tgt_out.view(-1), ignore_index=PAD,
                               label_smoothing=yumusatma, reduction="sum")


def parametre_say(model):
    return sum(p.numel() for p in model.parameters())
