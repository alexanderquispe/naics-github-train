"""
Data Loading and Preprocessing for NAICS Classification.

This module provides functions for loading, cleaning, and preparing
GitHub repository data for NAICS classification model training.
"""

import re
import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

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


def clean_topics(topic_str) -> str:
    """
    Clean and normalize the topics field.

    Handles various formats: lists, string representations of lists,
    comma-separated strings, etc.

    Args:
        topic_str: Topics field value (various formats)

    Returns:
        Cleaned topics string with semicolon separation
    """
    # Handle None/NaN values
    if topic_str is None or (isinstance(topic_str, float) and pd.isna(topic_str)):
        return ""

    # Handle arrays/lists
    if isinstance(topic_str, (list, tuple, np.ndarray)):
        if len(topic_str) == 0:
            return ""
        return "; ".join([str(item) for item in topic_str if item])

    # Handle empty strings
    if isinstance(topic_str, str) and (topic_str == "" or topic_str == "[]"):
        return ""

    try:
        # If it's a string representation of a list
        if isinstance(topic_str, str):
            if topic_str.startswith("[") and topic_str.endswith("]"):
                try:
                    topic_list = ast.literal_eval(topic_str)
                    if isinstance(topic_list, list):
                        return "; ".join([str(item) for item in topic_list if item])
                except (ValueError, SyntaxError):
                    pass

            # Clean string format
            return topic_str.replace("[", "").replace("]", "").replace(",", ";").strip()

        # Convert other types to string
        return str(topic_str)

    except Exception as e:
        logger.warning(f"Error processing topic: {topic_str}, Error: {e}")
        return ""


def clean_readme_text(text: str) -> str:
    """
    Clean README text by removing markdown artifacts, code blocks, and noise.

    Args:
        text: Raw README content

    Returns:
        Cleaned text string
    """
    if not text or pd.isna(text):
        return ""

    text = str(text)

    # Remove badges and shields
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # ![badge](url)
    text = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", text)  # [![badge](url)](link)

    # Remove license/copyright headers
    text = re.sub(
        r"(MIT License|Apache License|GPL|BSD|Copyright.*?)(\n|$)",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Clean URLs but keep domain info
    text = re.sub(r"https?://([^/\s]+)[^\s]*", r"\1", text)

    # Remove excessive markdown formatting
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # Headers
    text = re.sub(r"[*_~`]{1,2}", "", text)  # Bold/italic/code markers

    # Remove code blocks but keep language info
    text = re.sub(r"```(\w+)?\n.*?\n```", r"code-\1", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"\1", text)  # Inline code

    # Normalize technology mentions
    text = re.sub(r"\b(javascript|js)\b", "javascript", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(python|py)\b", "python", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(react|reactjs)\b", "react", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(node|nodejs)\b", "nodejs", text, flags=re.IGNORECASE)

    # Clean excessive punctuation
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)
    text = re.sub(r"[.]{3,}", "...", text)

    # Normalize whitespace
    text = re.sub(r"\n\s*\n", " ", text)
    text = re.sub(r"\s+", " ", text)

    # Remove common installation noise
    text = re.sub(
        r"(npm install|pip install|git clone).*?(\n|$)", "", text, flags=re.IGNORECASE
    )

    return text.strip()


def prepare_text_input(
    row: pd.Series,
    clean_text: bool = True,
    max_readme_words: Optional[int] = None,
) -> str:
    """
    Format repository data into a single text input for the model.

    Args:
        row: DataFrame row with repository data
        clean_text: Whether to clean the text
        max_readme_words: Maximum number of words to include from README

    Returns:
        Formatted text string
    """
    components = []

    # Repository name
    if pd.notna(row.get("name_repo")):
        components.append(f"Repository: {row['name_repo']}")
    elif pd.notna(row.get("repo")):
        components.append(f"Repository: {row['repo']}")

    # Description
    if pd.notna(row.get("description")):
        components.append(f"Description: {row['description']}")

    # Topics
    if "topics" in row.index and pd.notna(row["topics"]) and str(row["topics"]).strip():
        components.append(f"Topics: {row['topics']}")

    # README content
    if pd.notna(row.get("readme_content")):
        readme_text = str(row["readme_content"])
        if max_readme_words:
            readme_words = readme_text.split()[:max_readme_words]
            readme_text = " ".join(readme_words)
        components.append(f"README: {readme_text}")

    # Combine components
    combined_text = " | ".join(components)

    # Clean if requested
    if clean_text:
        combined_text = clean_readme_text(combined_text)

    return combined_text


def prepare_naics_dataset(
    raw_data: pd.DataFrame,
    target_column: str = "code",
    clean_text: bool = True,
    max_readme_words: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """
    Prepare the NAICS dataset for model training.

    Args:
        raw_data: Raw DataFrame with repository data
        target_column: Column containing NAICS codes
        clean_text: Whether to clean text content
        max_readme_words: Maximum README words to include

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

    # Clean topics if available
    if "topics" in df.columns:
        df["topics"] = df["topics"].apply(clean_topics)

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
