# Töz-Çevir-1

**Töz-Çevir-1** is [Töz AI](https://huggingface.co/TozAI)'s English ↔ Turkish translation model, trained from scratch. 74.1M parameters, one model for both directions, Apache-2.0. It is the sibling of [Töz-1](https://huggingface.co/TozAI/Toz-1).

It was trained by one person on Kaggle's free 2×T4 GPUs, without any non-commercial data and without distilling from another translation model. On the benchmarks below it is ahead of the similarly sized OPUS-MT base model everywhere, and on English→Turkish it is level with OPUS-MT tc-big, which has about three times as many parameters. Turkish→English is where it still falls short.

## Results

chrF++ / BLEU on 500 sentences from each test set, beam size 4. Higher is better; best chrF++ per row in bold.

| Test set | Direction | **Töz-Çevir-1** (74M) | OPUS-MT base (~75M) | OPUS-MT tc-big (~235M) |
|---|---|---|---|---|
| FLORES-200 devtest | en→tr | **60.0** / 32.6 | 55.5 / 26.0 | 59.2 / 30.9 |
| FLORES-200 devtest | tr→en | 59.1 / 35.5 | 56.7 / 31.7 | **62.1** / 38.0 |
| WMT24++ | en→tr | **53.1** / 24.5 | 46.8 / 18.4 | 50.8 / 22.4 |
| NTREX-128 | en→tr | 48.3 / 18.7 | 46.7 / 16.5 | **48.8** / 19.0 |

How to read this:

- The 500 sentences are a fixed subset (every ⌊N/500⌋-th sentence, first 500 taken), the same for every system. The full sets have 1012 (FLORES), 960 (WMT24++) and 1997 (NTREX) sentences, so differences of a few tenths of a point are within noise.
- Scores come from `olcum.py` in this repository, with sacrebleu (chrF++ = chrF with word order 2, BLEU with default tokenization). All systems were run by us with the same settings.
- "OPUS-MT base" is `Helsinki-NLP/opus-tatoeba-en-tr` for en→tr (the older `opus-mt-en-tr` is no longer downloadable) and `Helsinki-NLP/opus-mt-tr-en` for tr→en. "tc-big" is `opus-mt-tc-big-en-tr` / `opus-mt-tc-big-tr-en`. Their parameter counts are estimates from checkpoint size.
- WMT24++ and NTREX were written in English, so only en→tr is reported for them.
- The checkpoint is the one from step 120,000. Between step 118,675 and 120,000 the scores moved by 0.0–0.4 chrF++.

## Versions

| version | date | notes |
|---|---|---|
| **v1.0.0** | October 2026 | first release: 120k training steps, scores above |

Older versions stay downloadable with `tozcevir.load(revision="v1.0.0")` (a Hub tag).

## Quick start

```bash
pip install git+https://github.com/Toz-AI/toz-cevir.git
```

```python
import tozcevir

t = tozcevir.load()          # downloads toz-cevir.pt (~300 MB) and tokenizer.model from the Hub
print(t.translate("The weather is lovely today, so we decided to walk to the park.", to="tr"))
print(t.translate("Bugün hava çok güzel, bu yüzden parka yürüyerek gitmeye karar verdik.", to="en"))
```

You never tell the model the source language, only the target (`to="tr"` or `to="en"`). Longer text is split into sentences and translated one by one, because the model was trained on sentence pairs.

Command line:

```bash
python -m tozcevir --to tr "Could you please send me the report by Friday?"
python -m tozcevir --to en            # interactive; :tr / :en switch direction, :q quits
```

It uses the GPU if there is one and runs on CPU otherwise. Beam search is not batched, so on an RTX 3060 Ti it takes roughly 0.2 s per sentence. Local files work too: `tozcevir.load("path/to/folder")` where the folder has `toz-cevir.pt` and `tokenizer.model`.

## Model

Encoder-decoder transformer, 74.1M parameters.

| | |
|---|---|
| Layers | 10 encoder, 6 decoder |
| Width | d = 512, 8 heads, SwiGLU feed-forward (1408) |
| Normalization | RMSNorm, QK-norm |
| Positions | RoPE in self-attention only |
| Embeddings | one 32k matrix shared by encoder, decoder and output |
| Tokenizer | SentencePiece unigram, 32k, byte fallback, digits split |
| Direction | a `<2tr>` / `<2en>` tag as the first source token |
| Max input | 254 tokens per sentence |

## Training

- **Data:** about 170M raw English–Turkish pairs from [OPUS](https://opus.nlpl.eu/) and HPLT, reduced to 95M after filtering. Filters: length and ratio rules, script and language ID, number consistency, LaBSE similarity, and the LASER scores that come with NLLB. Test sentences were removed by exact match on either side (normalized); there is no n-gram overlap check yet.
- **Sources:** NLLB (the CCMatrix mining), HPLT v3, OpenSubtitles, MaCoCu, GoURMET, wikimedia, WikiMatrix, SETIMES, Tatoeba, Wikipedia, KDE4, translatewiki, infopankki, Bianet, GlobalVoices, tldr-pages, XLEnt, LinguaTools-WikiTitles. Web-mined text and subtitles make up most of the tokens; clean human translations are a small share and were sampled more often.
- **Left out on purpose:** TED/QED/Tanzil (non-commercial licenses) and WMT news (it is the test data). No synthetic data and no output of another translation model was added, so nothing is distilled. The mined web corpora themselves can still contain some machine-translated pairs.
- **Run:** 120k steps of about 49k tokens (roughly 6B tokens, each pair seen in a random direction), AdamW, peak learning rate 7e-4 with warmup and a linear decay over the last 15%, label smoothing 0.1, fp16 on 2×T4. About 40 hours of wall-clock time, spread over several Kaggle sessions.

This repository has the inference and evaluation code only. The training code is not published.

## Limitations

- Sentence-level model. It has no document context, so pronouns, formality and terminology can drift across sentences.
- Idioms and wordplay are often translated literally.
- Very short inputs (one to three words, such as greetings) are a weak spot. With beam search the output is sometimes repeated ("I love you." → "Seni seviyorum. Seni seviyorum.") or padded with filler like "Oh,". Longer inputs don't show this; if you translate short phrases, give a little context or check the result.
- The training text leans towards web pages and subtitles. Formal, legal, medical or very technical text has not been tested and should be checked by a person.
- Web-mined data can contain wrong or machine-translated pairs, and the model can repeat their mistakes and biases. Names and numbers can be wrong, so check the ones that matter.
- Turkish→English is about 3 chrF++ behind tc-big.
- Inputs over 254 tokens are truncated.

## Reproducing the numbers

```bash
git clone https://github.com/Toz-AI/toz-cevir.git && cd toz-cevir
pip install -e ".[eval]"
python olcum.py                    # downloads the weights from the Hub
```

The script downloads FLORES-200, WMT24++ and NTREX-128 itself, caches the outputs, and compares against the OPUS-MT models. `--n 0` uses the full test sets, `--rivals none` skips the other systems, `--weights path/to/toz-cevir.pt` uses a local file.

## License and credits

Code and weights are released under Apache-2.0 (see `LICENSE`). The training data comes from many sources with their own licenses (several are CC-BY or CC-BY-SA; OpenSubtitles has no explicit license). Please look at the OPUS and HPLT pages for the details before relying on the data itself. We thank the maintainers of OPUS, HPLT, MaCoCu, GoURMET, Tatoeba and Wikimedia, and the authors of FLORES-200, NTREX and WMT24++ for the test data.

---

## Türkçe

**Töz-Çevir-1**, sıfırdan eğitilmiş küçük bir İngilizce ↔ Türkçe çeviri modeli: 74M parametre, iki yön için tek model, Apache-2.0. Kaggle'ın ücretsiz 2×T4 GPU'larında, ticari olmayan (NC) veri kullanmadan ve başka bir çeviri modelinden damıtma yapmadan eğitildi.

FLORES-200 en→tr'de chrF++ 60,0 ile yaklaşık üç kat büyük OPUS-MT tc-big'in (59,2) önünde. WMT24++'ta da önde (53,1 / 50,8). NTREX'te biraz geride (48,3 / 48,8). tr→en'de tc-big'in yaklaşık 3 puan altında (59,1 / 62,1). Benzer boydaki OPUS-MT base'i her sette geçiyor. Sayılar her setten 500 cümleyle ölçüldü; tam setlerde küçük farklar oynayabilir.

Kullanım:

```bash
pip install git+https://github.com/Toz-AI/toz-cevir.git
python -m tozcevir --to en "Yarın okula gideceğim ama İstanbul'u çok özledim."    # kaynak dili söylemene gerek yok, yalnız hedef
```

Bilinen eksikler: cümle düzeyinde çalışır (belge bağlamı yok), deyimleri çoğu zaman kelimesi kelimesine çevirir, bir-üç kelimelik çok kısa girdilerde çıktıyı bazen tekrarlar ("Seni seviyorum. Seni seviyorum."), eğitim metni ağırlıkla web ve altyazı olduğundan resmî, hukuki, tıbbi metinlerde mutlaka kontrol edilmeli. Bu depoda yalnızca çıkarım ve ölçüm kodu var; eğitim kodu yayınlanmadı.
