"""Töz-Çevir-1 ile çeviri: ağırlık/tokenizer yükleme, beam search, paragraf bölme."""
import os
import re

import torch
import torch.nn.functional as F

from .model import BOS, EOS, PAD, Ayar, TozCevir, parametre_say

HF_REPO = os.environ.get("TOZCEVIR_REPO", "TozAI/Toz-Cevir-1")
AGIRLIK_DOSYASI = "toz-cevir.pt"
TOKENIZER_DOSYASI = "tokenizer.model"

_BOL = re.compile(r"(?<=[.!?…])\s+(?=[\"'“‘(\[]*[A-ZÇĞİÖŞÜ0-9])")


@torch.no_grad()
def cevir_cumle(model, sp, etiket, cumle, hedef, beam=4, alfa=1.0):
    ids = sp.encode(cumle)[:254]
    if not ids:
        return ""
    dev = next(model.parameters()).device
    src = torch.tensor([[etiket[hedef]] + ids + [EOS]], device=dev)
    maske = src != PAD
    bellek = model.kodla(src, maske)
    maks = min(2 * len(ids) + 10, 254)
    ys = torch.full((1, 1), BOS, dtype=torch.long, device=dev)
    skor = torch.zeros(1, device=dev)
    bitmis = []
    for _ in range(maks):
        n = ys.shape[0]
        h = model.coz(ys, bellek.expand(n, -1, -1), maske.expand(n, -1))[:, -1]
        lp = F.log_softmax(F.linear(h, model.emb.weight).float(), dim=-1)
        lp[:, PAD] = -1e9
        top = (skor[:, None] + lp).view(-1).topk(2 * beam)
        yeni, yskor = [], []
        for sk, idx in zip(top.values.tolist(), top.indices.tolist()):
            b, t = divmod(idx, lp.shape[1])
            if t == EOS:
                bitmis.append((sk / (ys.shape[1] ** alfa), ys[b, 1:].tolist()))
            elif len(yeni) < beam:
                yeni.append(torch.cat((ys[b], torch.tensor([t], device=dev))))
                yskor.append(sk)
        if not yeni or len(bitmis) >= beam:
            break
        ys, skor = torch.stack(yeni), torch.tensor(yskor, device=dev)
    if not bitmis:
        bitmis = [(0.0, ys[0, 1:].tolist())]
    return sp.decode(max(bitmis, key=lambda x: x[0])[1])


def cevir_metin(model, sp, etiket, metin, hedef, beam=4, bol=True):
    satirlar = []
    for par in metin.split("\n"):
        par = par.strip()
        if not par:
            satirlar.append("")
            continue
        parcalar = _BOL.split(par) if bol else [par]
        satirlar.append(" ".join(cevir_cumle(model, sp, etiket, p, hedef, beam) for p in parcalar))
    return "\n".join(satirlar)


class Translator:
    """tozcevir.load() döndürür. translate("...", to="tr") → str; liste verilirse liste döner."""

    def __init__(self, model, sp, adim=None):
        self.model, self.sp, self.adim = model, sp, adim
        self.etiket = {"en": sp.piece_to_id("<2en>"), "tr": sp.piece_to_id("<2tr>")}

    @property
    def device(self):
        return next(self.model.parameters()).device

    def translate(self, text, to="tr", beam=4, split=True):
        """to: hedef dil ("tr" ya da "en"); kaynak dil söylenmez, modele hedef etiketi verilir.
        split=True: paragrafı cümlelere bölüp tek tek çevirir (model cümle düzeyinde eğitildi)."""
        if to not in self.etiket:
            raise ValueError(f"to='tr' ya da 'en' olmalı, verilen: {to!r}")
        if isinstance(text, (list, tuple)):
            return [self.translate(t, to, beam, split) for t in text]
        return cevir_metin(self.model, self.sp, self.etiket, text, to, beam, split)

    __call__ = translate


def _indir(repo, dosya, revision=None):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, dosya, revision=revision)


def load(path=None, tokenizer=None, device=None, repo=HF_REPO, revision=None, allow_pickle=False):
    """Modeli yükle.

    path       toz-cevir.pt yolu ya da içinde toz-cevir.pt ve tokenizer.model olan klasör; None → Hugging Face'ten indirir
    revision   Hub'daki sürüm etiketi (ör. "v1.0.0"); None → son sürüm
    tokenizer  tokenizer.model yolu; None → path'in yanında ya da Hub'da aranır
    allow_pickle  yalnız KENDİ eğitim checkpoint'in (ckpt/son.pt: optimizer + argparse içerir) için True yap;
                  yayınlanan toz-cevir.pt `weights_only=True` ile güvenle yüklenir.
    """
    import sentencepiece as spm

    if path is not None and os.path.isdir(path):
        klasor, path = path, os.path.join(path, AGIRLIK_DOSYASI)
        tokenizer = tokenizer or os.path.join(klasor, TOKENIZER_DOSYASI)
    if path is None:
        path = _indir(repo, AGIRLIK_DOSYASI, revision)
    if tokenizer is None:
        yanda = os.path.join(os.path.dirname(os.path.abspath(path)), TOKENIZER_DOSYASI)
        tokenizer = yanda if os.path.exists(yanda) else _indir(repo, TOKENIZER_DOSYASI, revision)

    try:
        ck = torch.load(path, map_location="cpu", weights_only=not allow_pickle)
    except Exception as e:
        if allow_pickle:
            raise
        raise RuntimeError(f"{path} güvenli modda okunamadı ({e!r}). Bu bir eğitim checkpoint'iyse (optimizer içerir) "
                           "ve kaynağına güveniyorsan allow_pickle=True ver.") from e
    sp = spm.SentencePieceProcessor(model_file=tokenizer)
    model = TozCevir(Ayar(**ck["ayar"]))
    model.load_state_dict(ck["model"])
    model.eval()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return Translator(model, sp, (ck.get("durum") or {}).get("adim"))


__all__ = ["HF_REPO", "Translator", "load", "cevir_cumle", "cevir_metin", "parametre_say"]
