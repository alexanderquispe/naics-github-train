# NAICS GitHub Repository Classifier using Transformers

Fine-tuning transformer models to classify GitHub repositories into NAICS (North American Industry Classification System) codes based on repository metadata.

## Overview

This project trains transformer models (ModernBERT, DeBERTa, RoBERTa) to automatically classify GitHub repositories into industry categories using the NAICS coding system. The classifier uses repository metadata including:

- Repository name
- Description
- Topics/tags
- README content

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended for training)

### Setup

```bash
# Clone the repository
git clone https://github.com/alexanderquispe/naics-github-train.git
cd naics-github-train

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start with Pre-trained Model

If you want to use the pre-trained model for inference (without training your own):

### Option 1: Train Your Own Model (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (~4 minutes on GPU)
python scripts/train.py --model roberta-base --epochs 8 --batch-size 16
```

### Option 2: Download Pre-trained Model

The trained model files are too large for GitHub. Download from:
- **Google Drive:** [Contact repository maintainer for link]
- **Hugging Face Hub:** [Coming soon]

After downloading, place the model folder in `models/roberta-base-naics-classifier/`.

### Option 3: Use the Inference Notebook

Open `notebooks/inference_demo.ipynb` in Jupyter for an interactive demo with examples.

```bash
jupyter notebook notebooks/inference_demo.ipynb
```

## Project Structure

```
naics-github-train/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration settings
├── .gitignore                   # Git ignore rules
├── data/
│   ├── raw/                     # Original data files
│   │   └── train_data_gpt_ab8score.parquet
│   └── processed/               # Processed data
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Data loading & preprocessing
│   ├── naics_mapping.py         # NAICS code descriptions
│   ├── trainer.py               # Model training pipeline
│   ├── inference.py             # Prediction functions
│   ├── metrics.py               # Evaluation metrics
│   └── visualization.py         # Plotting functions
├── scripts/
│   ├── train.py                 # CLI training script
│   ├── evaluate.py              # Evaluation script
│   └── predict.py               # Prediction script
├── notebooks/
│   └── inference_demo.ipynb     # Demo notebook for inference
├── models/                      # Saved model checkpoints
└── outputs/                     # Training logs & outputs
```

## Usage

### Training a Model

Train a classifier using the default ModernBERT model:

```bash
python scripts/train.py --model modernbert-base --data data/raw/train_data_gpt_ab8score.parquet
```

Training with custom parameters:

```bash
python scripts/train.py \
    --model deberta-v3-base \
    --epochs 10 \
    --batch-size 8 \
    --learning-rate 2e-5 \
    --output models/deberta-naics
```

Available models:
- `modernbert-base` - ModernBERT base (recommended, supports long sequences)
- `modernbert-large` - ModernBERT large
- `deberta-v3-base` - DeBERTa v3 base
- `deberta-v3-large` - DeBERTa v3 large
- `roberta-base` - RoBERTa base
- `roberta-large` - RoBERTa large

### Evaluating a Model

```bash
python scripts/evaluate.py \
    --model models/modernbert-base-naics-classifier \
    --test-data data/raw/train_data_gpt_ab8score.parquet \
    --plot
```

### Making Predictions

Single prediction:

```bash
python scripts/predict.py \
    --model models/modernbert-base-naics-classifier \
    --input "Repository: bank-api | Description: Banking API for financial transactions"
```

From repository components:

```bash
python scripts/predict.py \
    --model models/modernbert-base-naics-classifier \
    --repo-name bank-api \
    --description "Banking API for financial transactions" \
    --show-description \
    --show-confidence
```

Batch prediction from file:

```bash
python scripts/predict.py \
    --model models/modernbert-base-naics-classifier \
    --input-file repos.csv \
    --output predictions.csv
```

### Python API

```python
from src.data_loader import load_parquet_data, prepare_naics_dataset
from src.trainer import setup_model, train_model
from src.inference import load_trained_model, predict_naics

# Load and prepare data
raw_data = load_parquet_data("data/raw/train_data_gpt_ab8score.parquet")
processed_df, label2id, id2label = prepare_naics_dataset(raw_data)

# Load trained model for inference
model, tokenizer, label_mappings = load_trained_model("models/classifier")

# Make prediction
result = predict_naics(
    text="Repository: bank-api | Description: Banking API",
    model=model,
    tokenizer=tokenizer,
    label_mappings=label_mappings,
)
print(f"Predicted NAICS: {result['predicted_naics']}")
```

## Data Format

The training data should be a parquet file with the following columns:

| Column | Description | Required |
|--------|-------------|----------|
| `code` | NAICS code (2-digit) | Yes |
| `name_repo` or `repo` | Repository name | Recommended |
| `description` | Repository description | Recommended |
| `topics` | GitHub topics/tags | Optional |
| `readme_content` | README text | Optional |

## NAICS Codes Reference

The classifier predicts 2-digit NAICS sector codes:

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
| 55 | Management of Companies and Enterprises |
| 56 | Administrative and Support Services |
| 61 | Educational Services |
| 62 | Health Care and Social Assistance |
| 71 | Arts, Entertainment, and Recreation |
| 72 | Accommodation and Food Services |
| 81 | Other Services (except Public Administration) |
| 92 | Public Administration |

## Configuration

Edit `config.py` to customize:

- **Model settings**: Default model, max sequence length
- **Training parameters**: Epochs, batch size, learning rate, etc.
- **Data paths**: Input/output directories
- **Supported models**: Add new model architectures

Key configuration options:

```python
from config import ModelConfig, TrainingConfig, DataConfig

# Model configuration
model_config = ModelConfig(
    model_name="modernbert-base",
    max_seq_length=2048,
)

# Training configuration
training_config = TrainingConfig(
    num_epochs=8,
    batch_size=8,
    learning_rate=1.5e-5,
    early_stopping_patience=2,
)
```

## Model Performance

Performance on the test set (8 epochs, batch size 16):

| Model | F1 Score | Accuracy | Training Time |
|-------|----------|----------|---------------|
| RoBERTa-base | **79.3%** | **81.3%** | ~4 min (GPU) |

*Performance varies based on data quality, hyperparameters, and training data size.*

### Training Details

- **Dataset:** 2,538 GitHub repositories with NAICS labels
- **Train/Val/Test Split:** 70% / 10% / 20%
- **Classes:** 19 NAICS industry sectors
- **Hardware:** NVIDIA RTX 3080 (16GB)

## Development

### Running Tests

```bash
# Test data loading
python -c "from src.data_loader import load_parquet_data; print(load_parquet_data('data/raw/train_data_gpt_ab8score.parquet').head())"

# Test NAICS mapping
python -c "from src.naics_mapping import get_naics_description; print(get_naics_description('52'))"
```

### Adding New Models

1. Add the model to `SUPPORTED_MODELS` in `config.py`:
   ```python
   SUPPORTED_MODELS["new-model"] = "huggingface/model-id"
   ```

2. Adjust `max_seq_length` in `ModelConfig.__post_init__()` if needed.

## Troubleshooting

### CUDA Out of Memory

- Reduce `--batch-size` (try 4 or 2)
- Reduce `--max-seq-length` (try 1024 or 512)
- Use gradient accumulation (increase `gradient_accumulation_steps` in config)

### Slow Training

- Enable bf16 training (default if GPU supports it)
- Use fused optimizer (default)
- Increase `--batch-size` if memory allows

### Poor Model Performance

- Increase training epochs
- Adjust learning rate
- Check data quality and class balance
- Try different model architectures

## License

MIT License

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
- [NAICS Association](https://www.naics.com/)
