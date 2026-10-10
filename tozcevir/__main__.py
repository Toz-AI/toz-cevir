"""python -m tozcevir --to tr "I love you."      ya da metinsiz: etkileşimli (:tr / :en yönü değiştirir, :q çıkış)"""
import argparse

from . import __version__, load
from .translate import parametre_say


def main():
    p = argparse.ArgumentParser(prog="tozcevir", description="Töz-Çevir-1 en↔tr çeviri")
    p.add_argument("metin", nargs="?", help="çevrilecek metin; verilmezse etkileşimli mod")
    p.add_argument("--to", default="tr", choices=["tr", "en"], help="hedef dil")
    p.add_argument("--path", default=None, help="toz-cevir.pt ya da içeren klasör (yoksa Hugging Face'ten iner)")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--device", default=None, help="cuda | cpu (varsayılan: GPU varsa GPU)")
    p.add_argument("--no-split", action="store_true", help="paragrafı cümlelere bölme")
    p.add_argument("--version", action="version", version=f"tozcevir {__version__}")
    a = p.parse_args()

    t = load(a.path, a.tokenizer, a.device)
    bol = not a.no_split
    if a.metin is not None:
        print(t.translate(a.metin, a.to, a.beam, bol))
        return
    print(f"{parametre_say(t.model) / 1e6:.1f}M parametre, cihaz: {t.device}. Hedef: {a.to}. Komutlar: :tr  :en  :q")
    hedef = a.to
    while True:
        try:
            satir = input(f"[→{hedef}] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if satir in (":q", ":quit"):
            break
        if satir in (":tr", ":en"):
            hedef = satir[1:]
        elif satir:
            print(t.translate(satir, hedef, a.beam, bol), flush=True)


if __name__ == "__main__":
    main()
