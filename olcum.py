"""Töz-Çevir-1 ölçümü: FLORES-200 devtest / WMT24++ / NTREX-128 üzerinde OPUS-MT ile yan yana (chrF++ ve BLEU).
Windows / Linux / Mac; GPU varsa kullanır. Test setleri kendiliğinden indirilir, çıktılar önbelleğe alınır.

  pip install -e ".[eval]"
  python olcum.py                          # ağırlıkları Hub'dan indirir, 500 cümle/set, OPUS-MT base + tc-big
  python olcum.py --n 0                    # setlerin tamamı (FLORES 1012, WMT24++ 960, NTREX 1997)
  python olcum.py --rivals none            # yalnız Töz-Çevir
  python olcum.py --weights toz-cevir.pt --tokenizer tokenizer.model
"""
import argparse
import contextlib
import json
import os
import sys
import tarfile
import time
import urllib.request

import torch

if hasattr(sys.stdout, "reconfigure"):          # Windows konsolunda Türkçe karakterler
    sys.stdout.reconfigure(encoding="utf-8")

from tozcevir import HF_REPO, load as toz_yukle  # noqa: E402
from tozcevir.translate import cevir_cumle  # noqa: E402

AYNI = os.path.dirname(os.path.abspath(__file__))
VERI = os.path.join(AYNI, "olcum_veri")
ONBELLEK = os.path.join(AYNI, "olcum_onbellek")
RAKIPLER = {  # opus-mt-en-tr HF'de erişilemiyor (401) → en→tr için aynı kuruluşun opus-tatoeba-en-tr modeli
    "base": ("OPUS-MT base", "Helsinki-NLP/opus-tatoeba-en-tr", "Helsinki-NLP/opus-mt-tr-en"),
    "tcbig": ("OPUS-MT tc-big", "Helsinki-NLP/opus-mt-tc-big-en-tr", "Helsinki-NLP/opus-mt-tc-big-tr-en"),
}
SETLER = {  # kısa ad: (görünen ad, yönler) — WMT24++ ve NTREX İngilizce kökenli: yalnız en→tr dürüst
    "flores": ("FLORES-200 devtest", ("en→tr", "tr→en")),
    "wmt24pp": ("WMT24++", ("en→tr",)),
    "ntrex": ("NTREX-128", ("en→tr",)),
}


# ───────────────────────── test setleri ─────────────────────────
def indir(url, ad):
    os.makedirs(VERI, exist_ok=True)
    yol = os.path.join(VERI, ad)
    if not os.path.exists(yol):
        print(f"   indiriliyor: {url}", flush=True)
        urllib.request.urlretrieve(url, yol)
    return yol


def satirlar(b):
    return b.decode("utf-8").rstrip("\n").split("\n")


def veri_yukle(ad):
    """(İngilizce satırlar, Türkçe satırlar)"""
    if ad == "flores":
        tar = indir("https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz", "flores200.tar.gz")
        with tarfile.open(tar) as t:
            uye = {m.name.lstrip("./"): m for m in t.getmembers()}
            ac = lambda d: satirlar(t.extractfile(uye[f"flores200_dataset/devtest/{d}.devtest"]).read())
            return ac("eng_Latn"), ac("tur_Latn")
    if ad == "wmt24pp":
        yol = indir("https://huggingface.co/datasets/google/wmt24pp/resolve/main/en-tr_TR.jsonl", "wmt24pp_en-tr.jsonl")
        satir = [json.loads(x) for x in open(yol, encoding="utf-8") if x.strip()]
        satir = [r for r in satir if not r.get("is_bad_source")]          # kötü kaynaklar + canary
        return [r["source"] for r in satir], [r["target"] for r in satir]
    if ad == "ntrex":
        raw = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128/newstest2019-{}.txt"
        en = satirlar(open(indir(raw.format("src.eng"), "ntrex_en.txt"), "rb").read())
        tr = satirlar(open(indir(raw.format("ref.tur"), "ntrex_tr.txt"), "rb").read())
        return en, tr
    raise ValueError(ad)


# ───────────────────────── önbellek ─────────────────────────
def onbellekli(anahtar, kaynaklar, fn):
    os.makedirs(ONBELLEK, exist_ok=True)
    yol = os.path.join(ONBELLEK, anahtar.replace("/", "_").replace("→", "-").replace(" ", "_") + ".json")
    if os.path.exists(yol):
        d = json.load(open(yol, encoding="utf-8"))
        if d["kaynak"] == kaynaklar:
            print("      (önbellekten)", flush=True)
            return d["cikti"]
    cikti = fn(kaynaklar)
    json.dump(dict(kaynak=kaynaklar, cikti=cikti), open(yol, "w", encoding="utf-8"), ensure_ascii=False)
    return cikti


# ───────────────────────── sistemler ─────────────────────────
@contextlib.contextmanager
def anonim_hf():
    """OPUS-MT açık modeller; private repo için kullandığımız token'ı onlara göndermeyelim."""
    kayit = {k: os.environ.pop(k) for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") if k in os.environ}
    try:
        yield
    finally:
        os.environ.update(kayit)


_RK = {}


def rakip(ad, kaynaklar, cihaz, bs=32):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    if ad not in _RK:
        with anonim_hf():
            _RK[ad] = (AutoTokenizer.from_pretrained(ad), AutoModelForSeq2SeqLM.from_pretrained(ad).to(cihaz).eval())
    tok, m = _RK[ad]
    sirali = sorted(range(len(kaynaklar)), key=lambda i: len(kaynaklar[i]))
    cik = [None] * len(kaynaklar)
    with torch.no_grad():
        for b in range(0, len(sirali), bs):
            idx = sirali[b:b + bs]
            enc = tok([kaynaklar[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                      max_length=256).to(cihaz)
            out = m.generate(**enc, num_beams=4, max_length=256)
            for i, t in zip(idx, tok.batch_decode(out, skip_special_tokens=True)):
                cik[i] = t
    return cik


def bizim(model, sp, etiket, kaynaklar, hedef, beam):
    cik = []
    for i, s in enumerate(kaynaklar):
        cik.append(cevir_cumle(model, sp, etiket, s, hedef, beam))
        if (i + 1) % 100 == 0:
            print(f"      Töz-Çevir {i + 1}/{len(kaynaklar)}", flush=True)
    return cik


# ───────────────────────── ana akış ─────────────────────────
def main():
    p = argparse.ArgumentParser(description="Töz-Çevir-1 ölçümü")
    p.add_argument("--dosya", "--weights", default=None, help="toz-cevir.pt yolu; verilmezse Hub'dan indirilir")
    p.add_argument("--hf_repo", default=HF_REPO)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--allow_pickle", action="store_true", help="yalnız kendi eğitim checkpoint'in için")
    p.add_argument("--sayi", "--n", type=int, default=500, help="set/yön başına cümle (0 = hepsi)")
    p.add_argument("--setler", "--sets", default="flores,wmt24pp,ntrex")
    p.add_argument("--rakipler", "--rivals", default="base,tcbig", help="base,tcbig | none")
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--cikti", "--out", default="karsilastirma.csv")
    a = p.parse_args()

    import pandas as pd
    import sentencepiece as spm
    from sacrebleu.metrics import BLEU, CHRF

    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    if cihaz == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    print(f"cihaz: {cihaz}" + (f" ({torch.cuda.get_device_name(0)})" if cihaz == "cuda" else ""))

    t = toz_yukle(a.dosya, a.tokenizer, cihaz, a.hf_repo, allow_pickle=a.allow_pickle)
    model, sp, etiket, adim = t.model, t.sp, t.etiket, t.adim or "?"
    print(f"model: adım {adim}, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parametre", flush=True)

    rakipler = [] if a.rakipler in ("yok", "none") else [RAKIPLER[k] for k in a.rakipler.split(",") if k in RAKIPLER]
    chrf, bleu = CHRF(word_order=2), BLEU()
    satir, ornek, t_basla = [], {}, time.time()
    for set_kisa in [x for x in a.setler.split(",") if x in SETLER]:
        set_ad, yonler = SETLER[set_kisa]
        print(f"\n■ {set_ad}", flush=True)
        en, tr = veri_yukle(set_kisa)
        for yon in yonler:
            kay, ref = (en, tr) if yon == "en→tr" else (tr, en)
            n = a.sayi or len(kay)
            adim_ = max(1, len(kay) // n)
            ids = list(range(0, len(kay), adim_))[:n]
            kay, ref = [kay[i] for i in ids], [ref[i] for i in ids]
            hedef = "tr" if yon == "en→tr" else "en"
            print(f"▶ {yon} ({len(kay)} cümle)", flush=True)
            sistemler = {"Töz-Çevir": (f"toz_{adim}_{set_kisa}_{yon}_{len(kay)}",
                                       lambda k: bizim(model, sp, etiket, k, hedef, a.beam))}
            for ad, m_entr, m_tren in rakipler:
                mid = m_entr if yon == "en→tr" else m_tren
                sistemler[ad] = (f"{mid}_{set_kisa}_{yon}_{len(kay)}", lambda k, mid=mid: rakip(mid, k, cihaz))
            for sad, (anahtar, fn) in sistemler.items():
                t = time.time()
                try:
                    hyp = onbellekli(anahtar, kay, fn)
                except Exception as e:
                    print(f"   !!! {sad} başarısız: {e!r}"[:300])
                    continue
                c, b = chrf.corpus_score(hyp, [ref]).score, bleu.corpus_score(hyp, [ref]).score
                satir.append({"set": set_ad, "yön": yon, "sistem": sad, "chrF++": round(c, 1), "BLEU": round(b, 1),
                              "n": len(kay)})
                ornek[(set_ad, yon, sad)] = list(zip(kay, ref, hyp))
                print(f"   {sad:<16} chrF++ {c:5.1f}  BLEU {b:5.1f}  ({time.time() - t:.0f} sn)", flush=True)

    df = pd.DataFrame(satir)
    sira = ["Töz-Çevir"] + [r[0] for r in rakipler]
    print(f"\n══════════ KARŞILAŞTIRMA (adım {adim}, toplam {(time.time() - t_basla) / 60:.1f} dk) ══════════")
    for metrik in ("chrF++", "BLEU"):
        tablo = df.pivot_table(index=["set", "yön"], columns="sistem", values=metrik)
        tablo = tablo[[s for s in sira if s in tablo.columns]]
        print(f"\n{metrik}\n{tablo.to_string()}")
    df.to_csv(a.cikti, index=False, encoding="utf-8")
    print(f"\nkaydedildi: {a.cikti}")


if __name__ == "__main__":
    main()
