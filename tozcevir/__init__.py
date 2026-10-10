"""Töz-Çevir-1: 74M parametreli en↔tr çeviri modeli.

    import tozcevir
    t = tozcevir.load()
    t.translate("The weather is lovely today.", to="tr")
"""
from .translate import HF_REPO, Translator, load

__version__ = "1.0.0"
__all__ = ["HF_REPO", "Translator", "load", "__version__"]
