"""
Data Loading and Preprocessing for NAICS Classification.

This module provides functions for loading, cleaning, and preparing
GitHub repository data for NAICS classification model training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from .text_format import (  # noqa: F401  (re-exported for backwards compatibility)
    clean_readme_text,
    clean_topics,
    decode_description,
    format_model_input,
)

logger = logging.getLogger(__name__)


def load_parquet_data(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load data from a parquet file.

    Args:
        file_path: Path to the parquet file

    Returns:
        pandas DataFrame with the loaded data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be read as parquet
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        raise ValueError(f"Failed to load parquet file: {e}")


def prepare_text_input(
    row: pd.Series,
    clean_text: bool = True,
    max_readme_words: Optional[int] = None,
) -> str:
    """
    Format repository data into a single text input for the model.

    Thin wrapper over `text_format.format_model_input`, which is also what every
    inference path uses, so training and prediction cannot drift apart.

    Args:
        row: DataFrame row with repository data
        clean_text: Whether to clean the text
        max_readme_words: Maximum number of words to include from README

    Returns:
        Formatted text string
    """
    readme = row.get("readme_content")
    if max_readme_words and pd.notna(readme):
        readme = " ".join(str(readme).split()[:max_readme_words])

    return format_model_input(
        repo_name=row.get("name_repo") if pd.notna(row.get("name_repo")) else row.get("repo"),
        description=row.get("description"),
        topics=row["topics"] if "topics" in row.index else None,
        readme=readme,
        clean_text=clean_text,
    )


def prepare_naics_dataset(
    raw_data: pd.DataFrame,
    target_column: str = "code",
    clean_text: bool = True,
    max_readme_words: Optional[int] = None,
    min_samples_per_class: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Prepare the NAICS dataset for model training.

    Args:
        raw_data: Raw DataFrame with repository data
        target_column: Column containing NAICS codes
        clean_text: Whether to clean text content
        max_readme_words: Maximum README words to include
        min_samples_per_class: Minimum samples required per class.
            Classes with fewer samples will be excluded.

    Returns:
        Tuple of (processed DataFrame, label2id mapping, id2label mapping)
    """
    logger.info("Preparing NAICS dataset")
    logger.info(f"Original dataset: {len(raw_data)} examples")

    # Clean data
    df = raw_data.copy()
    df = df.dropna(subset=[target_column])
    df[target_column] = df[target_column].astype(str)

    logger.info(f"After cleaning: {len(df)} examples")
    logger.info(f"Unique NAICS codes: {df[target_column].nunique()}")

    # Filter out classes with too few samples
    if min_samples_per_class is not None and min_samples_per_class > 0:
        class_counts = df[target_column].value_counts()
        valid_classes = class_counts[class_counts >= min_samples_per_class].index.tolist()
        excluded_classes = class_counts[class_counts < min_samples_per_class]

        if len(excluded_classes) > 0:
            logger.info(f"Excluding {len(excluded_classes)} classes with < {min_samples_per_class} samples:")
            for code, count in excluded_classes.items():
                logger.info(f"  - Code {code}: {count} samples")

        df = df[df[target_column].isin(valid_classes)]
        logger.info(f"After filtering: {len(df)} examples, {len(valid_classes)} classes")

    # Topics are normalised inside format_model_input, once. Cleaning them here
    # as well used to rewrite commas twice, so a topic containing a comma came
    # out differently in training than at inference.

    # Create text inputs
    logger.info("Creating text inputs from repository data...")
    df["text"] = df.apply(
        lambda row: prepare_text_input(row, clean_text, max_readme_words),
        axis=1,
    )

    # Check text lengths
    text_lengths = df["text"].str.len()
    logger.info(
        f"Text length stats: min={text_lengths.min()}, "
        f"max={text_lengths.max()}, avg={text_lengths.mean():.0f}"
    )

    # Create label mappings
    unique_labels = sorted(df[target_column].unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    df["label"] = df[target_column].map(label2id)

    logger.info(f"Created {len(label2id)} label mappings")

    return df[["text", "label"]], label2id, id2label


def create_dataset_splits(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """
    Create stratified train/validation/test splits.

    Args:
        df: DataFrame with 'text' and 'label' columns
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test)
        seed: Random seed for reproducibility

    Returns:
        DatasetDict with train, validation, and test splits
    """
    logger.info("Creating dataset splits")

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=seed,
    )

    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        stratify=train_val_df["label"],
        random_state=seed,
    )

    logger.info(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    # Convert to Hugging Face datasets
    dataset_dict = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
            "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
            "test": Dataset.from_pandas(test_df.reset_index(drop=True)),
        }
    )

    return dataset_dict


def tokenize_dataset(
    dataset_dict: DatasetDict,
    tokenizer,
    max_length: int = 2048,
) -> DatasetDict:
    """
    Tokenize the dataset using the provided tokenizer.

    Args:
        dataset_dict: DatasetDict with text data
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length

    Returns:
        Tokenized DatasetDict
    """
    logger.info(f"Tokenizing dataset with max_length={max_length}")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    # Rename label column if needed
    if "label" in dataset_dict["train"].features.keys():
        dataset_dict = dataset_dict.rename_column("label", "labels")

    # Tokenize datasets
    tokenized_dataset = dataset_dict.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    logger.info(f"Tokenized features: {tokenized_dataset['train'].features.keys()}")

    return tokenized_dataset
