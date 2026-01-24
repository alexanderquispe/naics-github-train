"""
Inference Functions for NAICS Classification.

This module provides functions for loading trained models
and making predictions on new data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

logger = logging.getLogger(__name__)


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
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
    mappings_path = model_path / "label_mappings.json"
    if mappings_path.exists():
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
    device: Optional[int] = None,
) -> pipeline:
    """
    Create a Hugging Face pipeline for text classification.

    Args:
        model_path: Path to the saved model
        device: Device index (-1 for CPU, 0+ for GPU)

    Returns:
        Text classification pipeline
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    logger.info(f"Creating classifier pipeline from {model_path}")

    classifier = pipeline(
        task="text-classification",
        model=str(model_path),
        device=device,
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
) -> str:
    """
    Format repository data into model input format.

    Args:
        repo_name: Repository name
        description: Repository description
        topics: Repository topics (semicolon-separated)
        readme: README content

    Returns:
        Formatted text string for model input
    """
    components = []

    if repo_name:
        components.append(f"Repository: {repo_name}")

    if description:
        components.append(f"Description: {description}")

    if topics:
        components.append(f"Topics: {topics}")

    if readme:
        components.append(f"README: {readme}")

    return " | ".join(components)
