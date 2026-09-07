"""
Inference Functions for NAICS Classification.

This module provides functions for loading trained models
and making predictions on new data.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

from .text_format import format_model_input

logger = logging.getLogger(__name__)


def resolve_device(device: Optional[str] = None) -> str:
    """CUDA, then Apple Silicon, then CPU. Same order as scripts/inference_batch.py."""
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_trained_model(
    model_path: Union[str, Path],
    device: Optional[str] = None,
) -> tuple:
    """
    Load a trained model and tokenizer.

    Args:
        model_path: Path to the saved model directory
        device: Device to load model on ('cuda', 'cpu', or None for auto)

    Returns:
        Tuple of (model, tokenizer, label_mappings)
    """
    # A Hugging Face id such as "aquiro1994/naics-github-classifier" is not a
    # path and must reach from_pretrained untouched. Anything that exists on
    # disk is local; anything else is accepted only if it has the shape of a
    # Hub id, so a mistyped directory still fails loudly.
    local = Path(model_path).expanduser()
    if local.exists():
        model_path = local
    elif re.fullmatch(r"[\w.-]+/[\w.-]+", str(model_path)):
        model_path = str(model_path)          # Hub id, let from_pretrained resolve it
    else:
        raise FileNotFoundError(
            f"Model not found at {model_path}. Give a local directory that "
            f"exists, or a Hugging Face id such as 'org/model-name'."
        )

    device = resolve_device(device)

    logger.info(f"Loading model from {model_path}")
    logger.info(f"Using device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(device)
    model.eval()

    # Load label mappings if available
    label_mappings = None
    # Only a local directory carries label_mappings.json; for a Hub id the
    # mapping comes from the model config, which from_pretrained already read.
    mappings_path = Path(model_path) / "label_mappings.json" if isinstance(model_path, Path) else None
    if mappings_path is not None and mappings_path.exists():
        with open(mappings_path, "r") as f:
            label_mappings = json.load(f)
        logger.info(f"Loaded label mappings with {len(label_mappings['label2id'])} labels")
    else:
        # Try to get from model config
        if hasattr(model.config, "id2label"):
            label_mappings = {
                "id2label": model.config.id2label,
                "label2id": model.config.label2id,
            }

    return model, tokenizer, label_mappings


def create_classifier_pipeline(
    model_path: Union[str, Path],
    device: Optional[str] = None,
    max_length: int = 512,
) -> pipeline:
    """
    Create a Hugging Face pipeline for text classification.

    Args:
        model_path: Path to the saved model, or a Hugging Face model id
        device: "cuda", "mps" or "cpu"; auto-detected in that order when None
        max_length: Tokens to keep. Inputs are truncated to this length, which
            the model's positional limit requires

    Returns:
        Text classification pipeline
    """
    if device is None:
        device = resolve_device()

    logger.info(f"Creating classifier pipeline from {model_path}")

    # truncation and max_length are required: without them any input over the
    # model's positional limit raises "index 514 is out of bounds".
    classifier = pipeline(
        task="text-classification",
        model=str(model_path),
        device=device,
        truncation=True,
        max_length=max_length,
    )

    return classifier


def predict_naics(
    text: str,
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    label_mappings: Optional[Dict] = None,
    device: Optional[str] = None,
    return_all_scores: bool = False,
) -> Dict:
    """
    Predict NAICS code for a single text input.

    Args:
        text: Input text (formatted repository data)
        model: Trained model
        tokenizer: Model tokenizer
        label_mappings: Label mappings dictionary
        device: Device for inference
        return_all_scores: Whether to return scores for all classes

    Returns:
        Dictionary with prediction results
    """
    if device is None:
        device = next(model.parameters()).device

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

    # Get predicted class
    predicted_class_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, predicted_class_idx].item()

    # Map to label name
    predicted_label = str(predicted_class_idx)
    if label_mappings and "id2label" in label_mappings:
        id2label = label_mappings["id2label"]
        # Handle both string and int keys
        if str(predicted_class_idx) in id2label:
            predicted_label = id2label[str(predicted_class_idx)]
        elif predicted_class_idx in id2label:
            predicted_label = id2label[predicted_class_idx]

    result = {
        "predicted_naics": predicted_label,
        "confidence": confidence,
        "class_index": predicted_class_idx,
    }

    if return_all_scores:
        all_scores = {}
        for idx, score in enumerate(probs[0].tolist()):
            label = str(idx)
            if label_mappings and "id2label" in label_mappings:
                id2label = label_mappings["id2label"]
                if str(idx) in id2label:
                    label = id2label[str(idx)]
                elif idx in id2label:
                    label = id2label[idx]
            all_scores[label] = score
        result["all_scores"] = all_scores

    return result


def batch_predict(
    texts: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    label_mappings: Optional[Dict] = None,
    device: Optional[str] = None,
    batch_size: int = 16,
) -> List[Dict]:
    """
    Predict NAICS codes for a batch of texts.

    Args:
        texts: List of input texts
        model: Trained model
        tokenizer: Model tokenizer
        label_mappings: Label mappings dictionary
        device: Device for inference
        batch_size: Batch size for processing

    Returns:
        List of prediction dictionaries
    """
    if device is None:
        device = next(model.parameters()).device

    results = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        # Process each prediction in batch
        predicted_classes = torch.argmax(probs, dim=-1)
        confidences = probs.max(dim=-1).values

        for j, (pred_idx, conf) in enumerate(zip(predicted_classes, confidences)):
            pred_idx = pred_idx.item()
            conf = conf.item()

            # Map to label name
            predicted_label = str(pred_idx)
            if label_mappings and "id2label" in label_mappings:
                id2label = label_mappings["id2label"]
                if str(pred_idx) in id2label:
                    predicted_label = id2label[str(pred_idx)]
                elif pred_idx in id2label:
                    predicted_label = id2label[pred_idx]

            results.append(
                {
                    "text": batch_texts[j][:100] + "..." if len(batch_texts[j]) > 100 else batch_texts[j],
                    "predicted_naics": predicted_label,
                    "confidence": conf,
                }
            )

    return results


def format_repository_input(
    repo_name: Optional[str] = None,
    description: Optional[str] = None,
    topics: Optional[str] = None,
    readme: Optional[str] = None,
    clean_text: bool = True,
) -> str:
    """
    Format repository data into model input format.

    Applies the same preprocessing as training (see `src/text_format.py`):
    the four fields are joined and `clean_readme_text` is applied to the result.
    Before version 1.1 this function did not clean the text, which fed the model
    raw markdown while it had been fine-tuned on cleaned text.

    Args:
        repo_name: Repository name
        description: Repository description
        topics: Repository topics (list, repr of a list, or separated string)
        readme: README content
        clean_text: Apply the training-time cleaning. Pass False only to
            reproduce the pre-1.1 behaviour.

    Returns:
        Formatted text string for model input
    """
    return format_model_input(
        repo_name=repo_name,
        description=description,
        topics=topics,
        readme=readme,
        clean_text=clean_text,
    )
