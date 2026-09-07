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

## Model Details

| Property | Value |
|----------|-------|
| **Model** | `roberta-large` (355M parameters) |
| **Task** | Multi-class text classification |
| **Classes** | 19 NAICS industry sectors |
| **Language** | English |
| **Training Data** | 6,588 labeled GitHub repositories |

## Performance

| Metric | Validation | Test |
|--------|------------|------|
| **F1 Score** | 0.8678 | **0.8633** |
| **Accuracy** | 0.8741 | 0.8672 |
| **Precision** | 0.8678 | 0.8630 |
| **Recall** | 0.8741 | 0.8672 |

Generalization gap: 0.0045 (good generalization)

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
from transformers import pipeline

# truncation is required: without it any README over 512 tokens raises
# "index 514 is out of bounds"
classifier = pipeline(
    "text-classification",
    model="aquiro1994/naics-github-classifier",
    truncation=True,
    max_length=512,
)

text = "Repository: bank-api | Description: REST API for banking transactions | README: A secure API for financial operations"
result = classifier(text)
print(result)
# [{'label': '52', 'score': 0.9368}]  # Finance and Insurance
```

**Build the input with the package, not by hand.** The model was fine-tuned on
text that goes through `clean_readme_text`; a hand-assembled string is raw
markdown the model never saw in training. Measured over five probes, the
difference does not change the label but moves the score across the 0.8
threshold in three of them.

```python
from src.text_format import format_model_input

text = format_model_input(
    repo_name="bank-api",
    description="REST API for banking transactions",
    topics=["fintech", "payments"],
    readme="# bank-api\n\nA **secure** API for financial operations...",
)
```

### Full Example

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("aquiro1994/naics-github-classifier")
tokenizer = AutoTokenizer.from_pretrained("aquiro1994/naics-github-classifier")

# Format input
# same builder the model was trained with
from src.text_format import format_model_input

text = format_model_input(
    repo_name="mediscan",
    description="AI diagnostic tool for radiology",
    topics=["healthcare", "medical-imaging", "deep-learning"],
    readme="MediScan uses computer vision to assist radiologists...",
)

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()

# Map to NAICS code
id2label = model.config.id2label
print(f"Predicted NAICS: {id2label[predicted_class]}")  # 62 (Health Care)
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
| README | No | README content (can be truncated) |

## Training Details

### Training Data

- **Source:** GitHub repositories labeled with NAICS codes via GPT-4 classification
- **File:** `train_data_gpt_ab8_score_with_code.parquet`
- **Total Examples:** 6,588
- **Classes:** 19 NAICS sectors. Code 55 (Management of Companies) does not appear in the training data at all; the `--min-samples 80` filter excluded nothing on this corpus.

### Data Splits

| Split | Examples | Percentage |
|-------|----------|------------|
| Train | 4,611 | 70% |
| Validation | 659 | 10% |
| Test | 1,318 | 20% |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | `roberta-large` |
| Batch Size | 32 x 2 gradient accumulation = 64 effective |
| Learning Rate | 1.5e-05 |
| Epochs | 8 |
| Max Sequence Length | 512 |
| Optimizer | AdamW |
| Weight Decay | 0.02 |
| Warmup Ratio | 0.15 |
| Early Stopping Patience | 5 |
| Precision | BF16 |

### Training Environment

- **Hardware:** NVIDIA A100 40GB
- **Training Time:** ~8 minutes
- **Framework:** Hugging Face Transformers

### Preprocessing

Text preprocessing includes:
- Removal of markdown badges and formatting
- URL cleaning (keep domain names)
- License header removal
- Code block removal (keep language indicators)
- Technology term normalization (js → javascript, py → python)
- Whitespace normalization

## Limitations

- Trained primarily on English repositories
- May not generalize well to non-software repositories
- NAICS code 55 (Management of Companies) is absent from the training data, so the model cannot predict it
- Performance may vary for repositories with minimal README content
- Text length stats: min=80, max=212,260, avg=3,555 characters

## Training Code

Full training pipeline available at: [github.com/alexanderquispe/naics-github-train](https://github.com/alexanderquispe/naics-github-train)

### Reproduce Training

```bash
git clone https://github.com/alexanderquispe/naics-github-train.git
cd naics-github-train

python scripts/train.py \
    --model roberta-large \
    --data data/raw/train_data_gpt_ab8_score_with_code.parquet \
    --batch-size 32 \
    --epochs 8
```

## Citation

```bibtex
@misc{naics-github-classifier,
  author = {Alexander Quispe},
  title = {NAICS GitHub Repository Classifier},
  year = {2025},
  publisher = {Hugging Face},
  url = {https://huggingface.co/aquiro1994/naics-github-classifier}
}
```

## License

MIT License
