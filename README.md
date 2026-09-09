# NAICS GitHub Repository Classifier

Fine-tuning transformer models to classify GitHub repositories into NAICS (North American Industry Classification System) codes based on repository metadata.

## Model Performance

| Model | Test F1 | Test Accuracy | Training Time |
|-------|---------|---------------|---------------|
| **BGE-M3** (multilingual) | **86.39%** | **86.95%** | ~90 min (M5 Max) |
| **RoBERTa-large** (English) | **86.33%** | **86.72%** | ~8 min (A100) |

Both are on the Hub; the next section says which to use.

## Two published models

| | English | Multilingual |
|---|---|---|
| Model | [`aquiro1994/naics-github-classifier`](https://huggingface.co/aquiro1994/naics-github-classifier) | [`aquiro1994/naics-github-classifier-multilingual`](https://huggingface.co/aquiro1994/naics-github-classifier-multilingual) |
| Encoder | RoBERTa-large (355M) | BGE-M3 (568M) |
| Vocabulary | 50,265 tokens, English | 250,002 tokens, 100+ languages |
| Test accuracy | 86.72% | 86.95% |
| Weighted F1 | 86.33% | 86.39% |
| Macro F1 | 82.95% | 83.06% |
| A README and its own translation get the same sector | 19% | 77% |
| Train with | `--model roberta-large` | `--model bge-m3` |

On English text they are indistinguishable; the difference is that only the
second reads a README that is not in English, which on public GitHub is about
12% of repositories. Both come out of training calibrated, so `score >= 0.8`
means what it says.

The multilingual model's **training data is still English**. What BGE-M3 adds is
a shared representation space from its own pre-training, so a Spanish or Chinese
README lands near its English equivalent and the head trained on English still
applies. Training on multilingual labelled data would be the next step, and
nothing here measures what it would add.

## Quick Start

### Use Pre-trained Model (Recommended)

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="aquiro1994/naics-github-classifier"
)

text = "Repository: bank-api | Description: REST API for banking transactions | README: Secure financial API"
result = classifier(text)
print(result)
# [{'label': '52', 'score': 0.9368}]  # Finance and Insurance
```

### Train Your Own Model

```bash
# Clone and install
git clone https://github.com/alexanderquispe/naics-github-train.git
cd naics-github-train
pip install -e .          # or: pip install -r requirements.txt

# Train RoBERTa-large (~8 min on A100)
python scripts/train.py \
    --model roberta-large \
    --data data/raw/train_data_gpt_ab8_score_with_code.parquet \
    --batch-size 32 \
    --epochs 8
```

### Google Colab

```python
!git clone https://github.com/alexanderquispe/naics-github-train.git
%cd naics-github-train
!pip install -q transformers datasets accelerate scikit-learn

!python scripts/train.py \
    --model roberta-large \
    --data data/raw/train_data_gpt_ab8_score_with_code.parquet \
    --batch-size 32 \
    --epochs 8
```

## Overview

This project trains transformer models (RoBERTa, ModernBERT, DeBERTa) to automatically classify GitHub repositories into industry categories using the NAICS coding system. The classifier uses repository metadata including:

- Repository name
- Description
- Topics/tags
- README content

## Training Details

| Property | Value |
|----------|-------|
| **Dataset** | 6,588 GitHub repositories |
| **Train/Val/Test Split** | 70% / 10% / 20% (4,611 / 659 / 1,318) |
| **Classes** | 19 NAICS industry sectors |
| **Base Model** | RoBERTa-large (355M parameters) |
| **Batch Size** | 32 |
| **Learning Rate** | 1.5e-05 |
| **Epochs** | 8 |
| **Max Sequence Length** | 512 |
| **Hardware** | NVIDIA A100 40GB |

## Project Structure

```
naics-github-train/
├── README.md                    # This file
├── MODEL_CARD.md                # Hugging Face model card
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration settings
├── data/
│   └── raw/
│       └── train_data_gpt_ab8_score_with_code.parquet
├── src/
│   ├── text_format.py           # The single input builder: training + inference
│   ├── data_loader.py           # Data loading & preprocessing
│   ├── trainer.py               # Model training pipeline
│   ├── inference.py             # Prediction functions
│   ├── naics_mapping.py         # NAICS code mappings
│   └── metrics.py               # Evaluation metrics
├── tests/
│   └── test_text_format.py      # Pins the training/inference input contract
├── scripts/
│   ├── train.py                 # CLI training script
│   ├── evaluate.py              # Evaluation script
│   ├── predict.py               # Prediction script
│   ├── inference_batch.py       # Batch inference on parquet files
│   └── plot_industry_adoption.py # Industry adoption visualizations
├── notebooks/
│   └── inference_demo.ipynb     # Demo notebook
└── models/                      # Saved model checkpoints
```

## Usage

### Training Options

```bash
# RoBERTa-large (best performance)
python scripts/train.py --model roberta-large --batch-size 32 --epochs 8 \
    --data data/raw/train_data_gpt_ab8_score_with_code.parquet

# RoBERTa-base (faster training)
python scripts/train.py --model roberta-base --batch-size 16 --epochs 8 \
    --data data/raw/train_data_gpt_ab8_score_with_code.parquet

# With gradient checkpointing (for limited GPU memory)
python scripts/train.py --model roberta-large --batch-size 8 --gradient-checkpointing \
    --data data/raw/train_data_gpt_ab8_score_with_code.parquet
```

Available models:
- `roberta-large` - RoBERTa large (recommended)
- `roberta-base` - RoBERTa base
- `modernbert-base` - ModernBERT base
- `modernbert-large` - ModernBERT large
- `deberta-v3-base` - DeBERTa v3 base
- `deberta-v3-large` - DeBERTa v3 large

### Making Predictions

Build the input with `format_repository_input`. It applies the same
preprocessing the model was fine-tuned on; assembling the string by hand feeds
the model raw markdown it never saw in training.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from src.inference import format_repository_input

model = AutoModelForSequenceClassification.from_pretrained("aquiro1994/naics-github-classifier")
tokenizer = AutoTokenizer.from_pretrained("aquiro1994/naics-github-classifier")

text = format_repository_input(
    repo_name="mediscan",
    description="AI diagnostic tool for radiology",
    topics=["healthcare", "medical-imaging"],
    readme="# MediScan\n\nMedical **imaging** analysis for radiologists...",
)

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()

id2label = model.config.id2label
print(f"Predicted NAICS: {id2label[predicted_class]}")
```

### One input format, everywhere

`src/text_format.py` holds the only function that builds the model's input,
`format_model_input`. Training (`data_loader.prepare_text_input`), single
prediction (`inference.format_repository_input`, `scripts/predict.py`) and batch
inference (`scripts/inference_batch.py`) all delegate to it, so fine-tuning and
prediction cannot disagree about what the model reads. The format is:

```
Repository: <name> | Description: <description> | Topics: <a; b> | README: <readme>
```

with `clean_readme_text` applied to the joined string: badges, licence headers,
markdown decoration and code blocks removed, URLs reduced to their domain,
whitespace collapsed. Missing fields are dropped along with their separator, a
topics list becomes `a; b`, and a description serialised as a Python bytes repr
(`b'inform\xc3\xa1tica'`) is decoded back to text.

`tests/test_text_format.py` asserts that the three entry points produce the same
string, character for character.

> **Changed in 1.1.** Before this version `format_repository_input` and the
> snippet above did **not** clean the text, while training did, so
> `scripts/predict.py` fed the model raw markdown. Measured on 25,000 GitHub
> repositories the effect is small -- the two formats give the same sector for
> 99.4% of the repositories the model is confident about, and retention at a
> 0.8 threshold moves by 0.1 points -- but the mismatch was real. Pass
> `clean_text=False` for the old behaviour.

### Batch Inference

Run predictions on a parquet file with repository data:

```bash
# Basic usage
python scripts/inference_batch.py \
    --input data/repos.parquet \
    --output predictions.parquet

# With options
python scripts/inference_batch.py \
    --input data/repos.parquet \
    --output predictions.parquet \
    --batch-size 32 \
    --limit 1000  # For testing
```

The input parquet should have columns: `name` (or `name_repo`), `description`, `topics`, `readme` (or `readme_content`).

### Hardware

The device is auto-detected in this order: CUDA, then MPS (Apple Silicon), then CPU.
Half precision is enabled by default on CUDA and MPS. Override either one if needed:

```bash
python scripts/inference_batch.py -i data.parquet -o out.parquet --device mps
python scripts/inference_batch.py -i data.parquet -o out.parquet --no-fp16
```

Throughput on an Apple M5 Max (RoBERTa-large, batch 32, 512 tokens):

| Device | Precision | rows/s |
|--------|-----------|-------:|
| CPU | fp32 | 4.3 |
| MPS | fp32 | 47.5 |
| **MPS** | **fp16** | **177** |

fp16 does not change the predictions this model is used for: on a 2,000-repo
sample, labels above the `0.8` confidence threshold matched fp32 **100%** of the
time. Disagreements only appear below `score < 0.4`, on inputs such as
`Repository: ajax | README: \n`, where the model spreads probability almost
uniformly over the 19 classes and any numerical noise flips the argmax.

Batch size 32-64 is the sweet spot on Apple Silicon; larger batches are *slower*,
not faster. Peak memory was 6.5 GB.

### Large or repeated runs

`from_pretrained` revalidates the cached files against the Hub on every call, so
each run makes ~16 HTTP requests even when the model is already on disk. Over a
job split into chunks that adds up, and it inflates the model's download counter
on the Hub. Download once, then run offline:

```bash
# first run: downloads and caches the model
python scripts/inference_batch.py -i data.parquet -o out.parquet --limit 1

# subsequent runs: no Hub requests at all
python scripts/inference_batch.py -i data.parquet -o out.parquet --local-files-only
```

`HF_HUB_OFFLINE=1` has the same effect for any script that loads the model.

Memory is not usually the constraint: on an M5 Max, fp16 with batch 64 peaks at
3.2 GB. If you do hit an out-of-memory error on the GPU, the cause is almost
always untruncated README text — cap it with `--max-readme-chars` rather than
shrinking the batch.

### Reproducing the published NAICS datasets

The production pipeline that generated the published datasets does **not** apply
`clean_readme_text`, and truncates the README to 3,000 characters. Neither is
what training does, so the defaults here no longer reproduce it: `--max-readme-chars`
now defaults to no truncation, matching training. To reproduce the published
output exactly, pass both flags:

```bash
python scripts/inference_batch.py \
    -i data.parquet -o out.parquet \
    --no-clean-text --max-readme-chars 3000
```

Measured against the published predictions on a 400-repo sample, restricted to
the `score >= 0.8` rows that the analysis actually keeps: **100.00%** label
agreement with these flags, **97.35%** with the defaults. Use them for
replication only; for new work the defaults match training and are the right
choice.

### Tests

```bash
python -m pytest tests/ -q
```

`tests/test_text_format.py` locks the training/inference input contract.

### Industry Adoption Visualization

Generate adoption charts by industry:

```bash
python scripts/plot_industry_adoption.py

# With custom data directory
python scripts/plot_industry_adoption.py --raw-data-dir /path/to/data
```

## Input Format

The model expects text in this format:

```
Repository: {repo_name} | Description: {description} | Topics: {topics} | README: {readme_content}
```

| Field | Required | Description |
|-------|----------|-------------|
| Repository | Yes | Repository name |
| Description | No | Short description |
| Topics | No | Semicolon-separated tags |
| README | No | README content |

## Data Format

Training data should be a parquet file with these columns:

| Column | Description | Required |
|--------|-------------|----------|
| `code` | NAICS code (2-digit) | Yes |
| `name_repo` | Repository name | Yes |
| `description` | Repository description | Recommended |
| `topics` | GitHub topics/tags | Optional |
| `readme_content` | README text | Optional |

## NAICS Codes (19 Classes)

| Code | Sector |
|------|--------|
| 11 | Agriculture, Forestry, Fishing and Hunting |
| 21 | Mining, Quarrying, and Oil and Gas Extraction |
| 22 | Utilities |
| 23 | Construction |
| 31-33 | Manufacturing |
| 42 | Wholesale Trade |
| 44-45 | Retail Trade |
| 48-49 | Transportation and Warehousing |
| 51 | Information |
| 52 | Finance and Insurance |
| 53 | Real Estate and Rental and Leasing |
| 54 | Professional, Scientific, and Technical Services |
| 56 | Administrative and Support Services |
| 61 | Educational Services |
| 62 | Health Care and Social Assistance |
| 71 | Arts, Entertainment, and Recreation |
| 72 | Accommodation and Food Services |
| 81 | Other Services |
| 92 | Public Administration |

*Note: Code 55 (Management of Companies) excluded due to insufficient training samples (<80).*

## Troubleshooting

### CUDA Out of Memory

```bash
# Use gradient checkpointing
python scripts/train.py --model roberta-large --batch-size 8 --gradient-checkpointing \
    --data data/raw/train_data_gpt_ab8_score_with_code.parquet

# Or reduce batch size
python scripts/train.py --model roberta-large --batch-size 4
```

### Slow Training

- Use A100 or similar GPU for best performance
- Increase batch size if memory allows
- Enable BF16 (default on supported GPUs)

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

## License

MIT License




## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [NAICS Association](https://www.naics.com/)
