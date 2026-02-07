# NAICS GitHub Repository Classifier

Fine-tuning transformer models to classify GitHub repositories into NAICS (North American Industry Classification System) codes based on repository metadata.

## Model Performance

| Model | Test F1 | Test Accuracy | Training Time |
|-------|---------|---------------|---------------|
| **RoBERTa-large** | **86.33%** | **86.72%** | ~8 min (A100) |

**Pre-trained model available:** [huggingface.co/alexanderquispe/naics-github-classifier](https://huggingface.co/alexanderquispe/naics-github-classifier)

## Quick Start

### Use Pre-trained Model (Recommended)

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="alexanderquispe/naics-github-classifier"
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
pip install -r requirements.txt

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
│   ├── data_loader.py           # Data loading & preprocessing
│   ├── trainer.py               # Model training pipeline
│   ├── inference.py             # Prediction functions
│   └── metrics.py               # Evaluation metrics
├── scripts/
│   ├── train.py                 # CLI training script
│   ├── evaluate.py              # Evaluation script
│   └── predict.py               # Prediction script
├── notebooks/
│   └── inference_demo.ipynb     # Demo notebook
└── models/                      # Saved model checkpoints
```

## Usage

### Training Options

```bash
# RoBERTa-large (best performance)
python scripts/train.py --model roberta-large --batch-size 32 --epochs 8

# RoBERTa-base (faster training)
python scripts/train.py --model roberta-base --batch-size 16 --epochs 8

# With gradient checkpointing (for limited GPU memory)
python scripts/train.py --model roberta-large --batch-size 8 --gradient-checkpointing
```

Available models:
- `roberta-large` - RoBERTa large (recommended)
- `roberta-base` - RoBERTa base
- `modernbert-base` - ModernBERT base
- `modernbert-large` - ModernBERT large
- `deberta-v3-base` - DeBERTa v3 base
- `deberta-v3-large` - DeBERTa v3 large

### Making Predictions

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model = AutoModelForSequenceClassification.from_pretrained("alexanderquispe/naics-github-classifier")
tokenizer = AutoTokenizer.from_pretrained("alexanderquispe/naics-github-classifier")

text = "Repository: mediscan | Description: AI diagnostic tool for radiology | README: Medical imaging analysis..."

inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()

id2label = model.config.id2label
print(f"Predicted NAICS: {id2label[predicted_class]}")
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
python scripts/train.py --model roberta-large --batch-size 8 --gradient-checkpointing

# Or reduce batch size
python scripts/train.py --model roberta-large --batch-size 4
```

### Slow Training

- Use A100 or similar GPU for best performance
- Increase batch size if memory allows
- Enable BF16 (default on supported GPUs)

## License

MIT License

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [NAICS Association](https://www.naics.com/)
