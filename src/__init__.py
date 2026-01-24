"""
NAICS GitHub Repository Classifier

A package for fine-tuning transformer models to classify GitHub repositories
into NAICS (North American Industry Classification System) codes.
"""

from .naics_mapping import NAICS_CODE_TO_DESCRIPTION, get_naics_description, get_all_naics_codes
from .data_loader import load_parquet_data, prepare_naics_dataset, create_dataset_splits
from .metrics import compute_metrics, generate_classification_report
from .trainer import setup_model, get_training_args, train_model, save_model
from .inference import load_trained_model, predict_naics, batch_predict

__version__ = "1.0.0"
__all__ = [
    "NAICS_CODE_TO_DESCRIPTION",
    "get_naics_description",
    "get_all_naics_codes",
    "load_parquet_data",
    "prepare_naics_dataset",
    "create_dataset_splits",
    "compute_metrics",
    "generate_classification_report",
    "setup_model",
    "get_training_args",
    "train_model",
    "save_model",
    "load_trained_model",
    "predict_naics",
    "batch_predict",
]
