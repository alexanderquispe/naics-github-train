<!-- This file is published verbatim as the model card at
     https://huggingface.co/aquiro1994/naics-github-classifier
     Keep the two in sync: scripts/../MODEL_CARD.md is the source. -->

---
license: mit
language:
- en
library_name: transformers
tags:
- text-classification
- naics
- industry-classification
- github
- roberta
datasets:
- custom
metrics:
- f1
- accuracy
pipeline_tag: text-classification
---

# NAICS GitHub Repository Classifier

A fine-tuned RoBERTa-large model that classifies GitHub repositories into **19 NAICS (North American Industry Classification System)** industry sectors based on repository metadata.

## Model Description

This model takes GitHub repository information (name, description, topics, README) and predicts the most likely industry sector the repository belongs to.

- **Model:** `roberta-large` (355M parameters)
- **Task:** Multi-class text classification (19 classes)
- **Language:** English
- **Training Data:** 6,588 labeled GitHub repositories

## Intended Use

- Classifying GitHub repositories by industry sector
- Analyzing open-source software ecosystem by industry
- Research on technology adoption across industries

## NAICS Classes

| Label | NAICS Code | Industry Sector |
|-------|------------|-----------------|
| 0 | 11 | Agriculture, Forestry, Fishing and Hunting |
| 1 | 21 | Mining, Quarrying, Oil and Gas Extraction |
| 2 | 22 | Utilities |
| 3 | 23 | Construction |
| 4 | 31-33 | Manufacturing |
| 5 | 42 | Wholesale Trade |
| 6 | 44-45 | Retail Trade |
| 7 | 48-49 | Transportation and Warehousing |
| 8 | 51 | Information |
| 9 | 52 | Finance and Insurance |
| 10 | 53 | Real Estate and Rental |
| 11 | 54 | Professional, Scientific, Technical Services |
| 12 | 56 | Administrative and Support Services |
| 13 | 61 | Educational Services |
| 14 | 62 | Health Care and Social Assistance |
| 15 | 71 | Arts, Entertainment, and Recreation |
| 16 | 72 | Accommodation and Food Services |
| 17 | 81 | Other Services |
| 18 | 92 | Public Administration |

## Usage

### Quick Start

```python
import torch
from transformers import pipeline

# "mps" is the Apple Silicon GPU; it is not selected automatically, and
# leaving it out makes inference ~40x slower on a Mac. See the section below.
device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)

# truncation is required: without it any input over 512 tokens raises
# "index 514 is out of bounds for dimension 1 with size 514"
classifier = pipeline(
    "text-classification",
    model="aquiro1994/naics-github-classifier",
    device=device,
    truncation=True,
    max_length=512,
)

text = "Repository: bank-api | Description: REST API for banking transactions | README: A secure API for financial operations"
result = classifier(text)
print(result)
# [{'label': '52', 'score': 0.86}]  # Finance and Insurance
```

### Full Example

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    "aquiro1994/naics-github-classifier"
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("aquiro1994/naics-github-classifier")

# Format input
text = "Repository: mediscan | Description: AI diagnostic tool for radiology | Topics: healthcare; medical-imaging; deep-learning | README: MediScan uses computer vision to assist radiologists..."

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
with torch.no_grad():
    outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()

# Map to NAICS code
id2label = model.config.id2label
print(f"Predicted NAICS: {id2label[predicted_class]}")  # 62 (Health Care)
```

## Running on Apple Silicon (Mac)

The model runs on the Mac GPU through Metal (`mps`). PyTorch does not select it
automatically, so pass the device explicitly — otherwise inference falls back to
CPU and is ~40x slower.

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "mps" else torch.float32

model = AutoModelForSequenceClassification.from_pretrained(
    "aquiro1994/naics-github-classifier", dtype=dtype
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained("aquiro1994/naics-github-classifier")

def classify(texts, batch_size=32):
    # Sort by length so each batch pads to a short common length
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = [None] * len(texts)
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        batch = tokenizer([texts[j] for j in idx], padding=True, truncation=True,
                          max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            # softmax in fp32: fp16 loses precision on near-uniform logits
            probs = torch.softmax(model(**batch).logits.float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        for k, j in enumerate(idx):
            out[j] = (model.config.id2label[int(pred[k])], float(conf[k]))
    return out
```

Throughput on an Apple M5 Max (batch 32, 512 tokens):

| Device | Precision | rows/s |
|--------|-----------|-------:|
| CPU | fp32 | 4.3 |
| MPS | fp32 | 47.5 |
| **MPS** | **fp16** | **177** |

Notes:

- **fp16 is safe here.** On a 2,000-repo sample, labels above the `0.8` confidence
  threshold matched fp32 **100%** of the time. Disagreements appear only below
  `score < 0.4`, on inputs such as `Repository: ajax | README: \n`, where the model
  spreads probability almost uniformly over the 19 classes and any numerical noise
  flips the argmax.
- **Batch size 32-64 is the sweet spot**; larger batches are *slower*, not faster.
  Peak memory was 6.5 GB.
- Sorting by length before batching is worth 2-5x on mixed-length inputs, because
  otherwise every batch pads to its longest member.

## Batch or repeated inference

`from_pretrained` revalidates the cached files against the Hub on every call, so
each run makes HTTP requests even when the model is already on disk (measured: 8
per model load, 0 with the flag below). Over a job split into chunks this adds up,
and it inflates this model's download counter. Load once, then stay local:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "aquiro1994/naics-github-classifier",
    dtype=dtype,
    local_files_only=True,   # after the first run has cached the model
).to(device).eval()
```

`HF_HUB_OFFLINE=1` does the same for any script.

**On memory:** out-of-memory errors on the Mac GPU come from untruncated README
text, not from batch size — inputs can reach megabytes before truncation. Cap the
README (3,000 characters is what the published datasets use) rather than shrinking
the batch. With inputs capped, fp16 at batch 64 peaks at 3.2 GB on an M5 Max.

## Input Format

The model expects the four fields joined like this:

```
Repository: {repo_name} | Description: {description} | Topics: {topics} | README: {readme_content}
```

| Field | Required | Description |
|-------|----------|-------------|
| Repository | No | Repository name |
| Description | No | Short description |
| Topics | No | Semicolon-separated tags |
| README | No | README content |

Fields that are missing are dropped along with their separator.

### Build it with the package, not by hand

The model was fine-tuned on text that goes through a cleaning step, so a
hand-assembled string is raw markdown it never saw in training. Use the builder
from [`naics-github-train`](https://github.com/alexanderquispe/naics-github-train),
which is the same function the training pipeline calls:

```python
from src.text_format import format_model_input

text = format_model_input(
    repo_name="mediscan",
    description="AI diagnostic tool for radiology",
    topics=["healthcare", "medical-imaging"],
    readme="# MediScan\n\nMedical **imaging** analysis for radiologists...",
)
```

Measured over 25,000 GitHub repositories with this model's multilingual
sibling, cleaned and uncleaned input give the same sector for 99.4% of the
repositories it is confident about, and retention at a 0.8 threshold moves by
0.1 points. The difference is small but it is free to get right, and on five
short probes three moved across the 0.8 threshold that downstream pipelines
filter on.

### What the cleaning does, including two surprises

Badges and shields removed, licence headers removed, URLs reduced to their
domain, markdown headers and emphasis stripped, whitespace collapsed, and a few
technology names normalised (`js` to `javascript`, `py` to `python`).

Two behaviours are worth knowing because they are not what the code comments
suggest, and because the training data was built with them:

1. **Everything after the first `pip install`, `npm install` or `git clone` is
   dropped**, not just that line. On raw READMEs this costs 12.7% of
   repositories more than half their text. The training corpus was truncated the
   same way: over the 1,077 training repositories whose raw README contains an
   install command, the median goes from 4,072 to 1,445 characters and none of
   the install commands survive. The model therefore expects it.
2. **Code block bodies survive.** The fences are stripped but the code inside
   reaches the model; the rule that was meant to replace a block with its
   language name never fires.

Neither should be "fixed" without retraining and republishing this model.

## Training Details

### Training Data

- **Source:** GitHub repositories labeled with NAICS codes
- **Size:** 6,588 examples
- **Classes:** 19 NAICS sectors
- **Split:** 70% train / 10% validation / 20% test

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | `roberta-large` |
| Batch Size | 32, with 2 gradient accumulation steps: effective 64 |
| Learning Rate | 1.5e-5 |
| Epochs | 8, which is 584 optimizer steps; the recipe keeps the best checkpoint by validation F1 rather than the last |
| Warmup Ratio | 0.15 |
| LR Schedule | polynomial decay |
| Max Sequence Length | 512 |
| Optimizer | AdamW |
| Weight Decay | 0.02 |
| Early Stopping Patience | 5 evaluations, every 100 steps |
| Seed | 42 (see the note on reproducibility below) |

### Preprocessing

Text preprocessing, applied to the joined string:
- Removal of markdown badges and formatting
- URL cleaning (keep domain names)
- License header removal
- Whitespace normalization
- Technology term normalization (js → javascript, py → python)
- Everything after the first install command is dropped (see Input Format)
- Code fences are stripped; the code inside them is kept

See the Input Format section for the two behaviours that differ from what the
code comments claim, and why they must not be changed.

## Limitations

- Trained on English repositories only. The tokenizer has no vocabulary for
  other languages, so a README in Spanish, French or Chinese is not read: on 93
  such repositories, this model gives the original and its own English
  translation the same sector only 19% of the time.
- May not generalize to non-software repositories.
- NAICS code 55 (Management of Companies) is absent from the training data, so
  the model cannot predict it. The 19 classes above are all it knows.
- The training labels came from GPT-4.1 judgements, not human annotation.
- Performance varies for repositories with minimal README content. On the
  production corpus, roughly half of the repositories the model labels
  confidently arguably belong to no sector at all (coursework, portfolios,
  exercises); the training data contains only positives, so the model has no way
  to abstain.
- **Not reproducible.** The training script did not seed before initialising the
  classification head, so these exact weights cannot be regenerated. The code
  has since been fixed
  ([issue #4](https://github.com/alexanderquispe/naics-github-train/issues/4)),
  and runs from the current version are reproducible; this checkpoint predates
  that fix.

## Citation

```bibtex
@misc{naics-github-classifier,
  author = {{GitHub, Inc.} and Xu, Kevin and Quispe, Alexander},
  title = {NAICS GitHub Repository Classifier},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/aquiro1994/naics-github-classifier}
}
```

## Repository

Training code and data preparation: [github.com/alexanderquispe/naics-github-train](https://github.com/alexanderquispe/naics-github-train)
