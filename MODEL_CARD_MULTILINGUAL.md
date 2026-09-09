<!-- This file is published verbatim as the model card at
     https://huggingface.co/aquiro1994/naics-github-classifier-multilingual
     Keep the two in sync: MODEL_CARD_MULTILINGUAL.md is the source. -->

---
license: mit
language:
- multilingual
- en
- es
- fr
- pt
- ru
- zh
library_name: transformers
tags:
- text-classification
- naics
- industry-classification
- github
- bge-m3
- xlm-roberta
- multilingual
datasets:
- custom
metrics:
- f1
- accuracy
pipeline_tag: text-classification
base_model: BAAI/bge-m3
---

# NAICS GitHub Repository Classifier, multilingual

A fine-tuned [BGE-M3](https://huggingface.co/BAAI/bge-m3) that classifies GitHub
repositories into **19 NAICS industry sectors** from their name, description,
topics and README.

It is the multilingual counterpart of
[`aquiro1994/naics-github-classifier`](https://huggingface.co/aquiro1994/naics-github-classifier),
which is RoBERTa-large and English-only. Same task, same 19 classes, same
training data, same recipe; the difference is the encoder underneath.

## Which one should you use?

| | this model | `naics-github-classifier` |
|---|---|---|
| Encoder | BGE-M3 (XLM-RoBERTa, 568M) | RoBERTa-large (355M) |
| Vocabulary | 250,002 tokens, 100+ languages | 50,265 tokens, English |
| Test accuracy | **86.95%** | 86.72% |
| Weighted F1 | **86.39%** | 86.33% |
| Macro F1 | 83.06% | 82.95% |
| Same sector for a README and its own English translation | **77%** | 19% |
| Calibrated as trained | yes, `T` = 1.04 | yes, `T` = 1.10 |

On English text the two are indistinguishable: a 0.2-point difference on a
1,318-row test set is well inside run-to-run noise. **Use this one when the
corpus is not all English**, which for public GitHub is about 12% of
repositories: on a sample of 821,929, that is the share whose language is
confidently detected as something other than English. Use the RoBERTa model
when the corpus is English and you want the smaller, faster network.

## Important: the training data is English

This model reads other languages through cross-lingual transfer, not because it
was trained on them. The 6,588 labelled repositories are English. What BGE-M3
brings is a shared multilingual representation space from its own pre-training,
so a Spanish or Chinese README lands near its English equivalent and the
classification head, trained on English, still applies.

That transfer is real but imperfect, and it is what the 77% above measures: 93
non-English repositories were classified twice, once as written and once from a
human-quality English translation, and the two answers agreed 77% of the time.
RoBERTa-large agrees with itself 19% of the time on the same test, which is what
having no vocabulary for the text looks like.

| | FR (31) | ES (28) | RU (14) | PT (11) | ZH (9) | all (93) |
|---|---:|---:|---:|---:|---:|---:|
| this model | 84% | 68% | 100% | 73% | 56% | **77%** |
| RoBERTa-large | 16% | 11% | 21% | 18% | 56% | 19% |

Three caveats worth stating plainly. **Agreement is not accuracy**: when the two
readings differ, neither is known to be right. **The per-language cells are tiny**
— 9 to 31 repositories each, so a single flipped repository moves Chinese by 11
points and the Russian 100% rests on 14 cases. Only the 93-repository total is
worth quoting. And the figure varies between training runs more than the English
metrics do — a second run of the same recipe measured 81%, against an English F1
of 85.2. Treat 77% as one measurement, not a constant.

Training on multilingual labelled data would be the real upgrade, and nothing
here measures what it would add.

## Usage

```python
import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)

# truncation is required: without it any input over 512 tokens raises
# "index 514 is out of bounds"
classifier = pipeline(
    "text-classification",
    model="aquiro1994/naics-github-classifier-multilingual",
    device=device,
    truncation=True,
    max_length=512,
)

text = "Repository: fincas-app | Description: Gestión de alquileres y contratos de arrendamiento | README: Aplicación web para administradores de fincas..."
print(classifier(text))
# [{'label': '53', 'score': 0.97}]  # Real Estate and Rental
```

### Build the input with the package, not by hand

The model was fine-tuned on text that goes through a cleaning step, so a
hand-assembled string is raw markdown it never saw in training. Use the builder
from [`naics-github-train`](https://github.com/alexanderquispe/naics-github-train),
which is the same function the training pipeline calls:

```python
from src.text_format import format_model_input

text = format_model_input(
    repo_name="fincas-app",
    description="Gestión de alquileres y contratos de arrendamiento",
    topics=["real-estate", "property-management"],
    readme="# Fincas\n\nAplicación web para **administradores de fincas**...",
)
```

The format is `Repository: … | Description: … | Topics: a; b | README: …`, with
missing fields dropped along with their separator.

### What the cleaning does, including two surprises

Badges removed, licence headers removed, URLs reduced to their domain, markdown
stripped, whitespace collapsed. Two behaviours are not what the code comments
suggest, and the training corpus was built with them, so they are load-bearing:

1. **Everything after the first `pip install`, `npm install` or `git clone` is
   dropped**, not just that line. On raw READMEs this costs 12.7% of
   repositories more than half their text.
2. **Code block bodies survive.** Fences are stripped; the code inside reaches
   the model.

Neither should be changed without retraining.

## Thresholding

The model comes out of training calibrated (`T` = 1.04, expected calibration
error 0.028), so its score means roughly what it says and no temperature scaling
is needed. Filtering at `score >= 0.8` keeps **81.8%** of the test set and is
right on **92.6%** of what it keeps.

## NAICS Classes

| Label | Code | Sector | Label | Code | Sector |
|---|---|---|---|---|---|
| 0 | 11 | Agriculture, Forestry, Fishing and Hunting | 10 | 53 | Real Estate and Rental |
| 1 | 21 | Mining, Quarrying, Oil and Gas Extraction | 11 | 54 | Professional, Scientific, Technical Services |
| 2 | 22 | Utilities | 12 | 56 | Administrative and Support Services |
| 3 | 23 | Construction | 13 | 61 | Educational Services |
| 4 | 31-33 | Manufacturing | 14 | 62 | Health Care and Social Assistance |
| 5 | 42 | Wholesale Trade | 15 | 71 | Arts, Entertainment, and Recreation |
| 6 | 44-45 | Retail Trade | 16 | 72 | Accommodation and Food Services |
| 7 | 48-49 | Transportation and Warehousing | 17 | 81 | Other Services |
| 8 | 51 | Information | 18 | 92 | Public Administration |
| 9 | 52 | Finance and Insurance | | | |

NAICS code 55 (Management of Companies) is absent from the training data, so the
model cannot predict it.

## Training

Trained with
[`naics-github-train`](https://github.com/alexanderquispe/naics-github-train),
which regenerates this checkpoint:

```bash
python scripts/train.py --model bge-m3 \
    --data data/raw/train_data_gpt_ab8_score_with_code.parquet \
    --epochs 8 --batch-size 8 --gradient-accumulation-steps 8 \
    --min-samples 80 --eval-steps 100
```

| Parameter | Value |
|---|---|
| Base model | `BAAI/bge-m3` (568M parameters) |
| Data | 6,588 labelled repositories, 19 classes |
| Split | 4,611 train / 659 validation / 1,318 test, stratified, seed 42 |
| Sequence length | 512 tokens |
| Batch | 8 with 8 gradient accumulation steps: effective 64 |
| Epochs | 8 (584 optimizer steps); best checkpoint by validation F1, at step 300 |
| Learning rate | 1.5e-5, polynomial decay, 15% warm-up |
| Weight decay | 0.02 |
| Optimizer | AdamW |
| Hardware | Apple M5 Max, fp32, 151 minutes |

A 1,024-token window was tried under an earlier recipe and scored 0.5 points
lower than its 512-token counterpart: the signal is in the opening of the
README, not in its length.

The settings above are recorded in `training_config.json` in this repository, and
`scripts/evaluate.py` reads them so evaluation cannot silently use a different
split or sequence length.

## Limitations

- **The training data is English.** See the section above.
- **Agreement across languages is not accuracy**, it rests on 93 repositories,
  and it varies between runs (77% here, 81% in a second run of the same recipe).
  The per-language breakdown has 9 to 31 repositories per cell.
- **The labels are GPT-4.1 judgements**, not human annotation.
- **The model cannot abstain.** The training data contains only positives, so
  every repository is assigned some sector. On a production corpus of public
  GitHub repositories, roughly half of what the model labels confidently
  arguably belongs to no sector at all: coursework, portfolios, exercises,
  generic tooling. Filtering by score helps but does not solve this.
- **Sector 55 cannot be predicted**, and rare sectors are weak: Wholesale Trade
  (42) has 24 test examples and its F1 is the lowest of the nineteen.
- **One run.** Seed-to-seed variation on a 1,318-row test set is about ±1 point.

## Citation

```bibtex
@misc{naics-github-classifier-multilingual,
  author = {Quispe, Alexander and Xu, Kevin},
  title  = {NAICS GitHub Repository Classifier, multilingual},
  year   = {2026},
  url    = {https://huggingface.co/aquiro1994/naics-github-classifier-multilingual}
}
```

## Repository

Training and inference code:
[github.com/alexanderquispe/naics-github-train](https://github.com/alexanderquispe/naics-github-train)
