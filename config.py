"""
Configuration settings for NAICS GitHub Repository Classifier.

This module centralizes all configurable parameters for data paths,
model settings, and training hyperparameters.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Project paths
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# Supported models
SUPPORTED_MODELS: Dict[str, str] = {
    "modernbert-base": "answerdotai/ModernBERT-base",
    "modernbert-large": "answerdotai/ModernBERT-large",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "deberta-v3-large": "microsoft/deberta-v3-large",
    "roberta-base": "roberta-base",
    "roberta-large": "roberta-large",
}


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    model_name: str = "modernbert-base"
    max_seq_length: int = 2048

    @property
    def model_id(self) -> str:
        """Get the Hugging Face model ID."""
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{self.model_name}' not supported. "
                f"Choose from: {list(SUPPORTED_MODELS.keys())}"
            )
        return SUPPORTED_MODELS[self.model_name]

    def __post_init__(self):
        # Adjust max_seq_length based on model capabilities
        if "modernbert" in self.model_name:
            # ModernBERT supports up to 8192 tokens
            self.max_seq_length = min(self.max_seq_length, 8192)
        elif "deberta" in self.model_name or "roberta" in self.model_name:
            # DeBERTa and RoBERTa typically support up to 512 tokens
            self.max_seq_length = min(self.max_seq_length, 512)


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    # Training parameters
    num_epochs: int = 8
    batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 1.5e-5
    weight_decay: float = 0.02
    warmup_ratio: float = 0.15

    # Optimization
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "polynomial"

    # Evaluation and saving
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 5
    metric_for_best_model: str = "f1"

    # Early stopping
    early_stopping_patience: int = 2
    early_stopping_threshold: float = 0.001

    # Hardware
    use_bf16: bool = True
    use_fused_optimizer: bool = True

    # Data split ratios
    test_size: float = 0.2
    val_size: float = 0.1

    # Reproducibility
    seed: int = 42


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # Column names
    target_column: str = "code"
    text_columns: Dict[str, str] = field(default_factory=lambda: {
        "repo_name": "name_repo",
        "description": "description",
        "topics": "topics",
        "readme": "readme_content",
    })

    # Text processing
    max_readme_words: Optional[int] = None  # No limit by default
    clean_text: bool = True

    # Class filtering
    min_samples_per_class: int = 80  # Exclude classes with fewer samples

    # Default data file
    default_data_file: str = "train_data_naics_github.parquet"

    @property
    def default_data_path(self) -> Path:
        """Get the default data file path."""
        return RAW_DATA_DIR / self.default_data_file


def get_default_config() -> tuple:
    """Get default configuration objects."""
    return ModelConfig(), TrainingConfig(), DataConfig()


def validate_config(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_config: DataConfig
) -> bool:
    """Validate configuration settings."""

    # Check model name
    if model_config.model_name not in SUPPORTED_MODELS:
        logger.error(f"Invalid model: {model_config.model_name}")
        return False

    # Check training parameters
    if training_config.batch_size < 1:
        logger.error("Batch size must be at least 1")
        return False

    if training_config.learning_rate <= 0:
        logger.error("Learning rate must be positive")
        return False

    if not 0 < training_config.test_size < 1:
        logger.error("Test size must be between 0 and 1")
        return False

    # Check data config
    if not data_config.target_column:
        logger.error("Target column must be specified")
        return False

    logger.info("Configuration validated successfully")
    return True
