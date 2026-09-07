"""
Evaluation Metrics for NAICS Classification.

This module provides functions for computing and reporting
classification metrics during training and evaluation.
"""

import logging
from typing import Dict, Tuple, Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """
    Compute metrics for evaluation during training.

    This function is designed to be used as the compute_metrics
    argument in Hugging Face Trainer.

    Args:
        eval_pred: Tuple of (predictions logits, labels)

    Returns:
        Dictionary with f1, accuracy, precision, and recall
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    # Calculate metrics
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )

    # f1_macro is reported alongside the weighted score because the classes are
    # imbalanced 8:1 and a sector the model never predicts is invisible in the
    # weighted number.
    return {
        "f1": float(f1),
        "f1_macro": float(f1_macro),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
    }


def compute_detailed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Compute detailed classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: Averaging method ('weighted', 'macro', 'micro')

    Returns:
        Dictionary with detailed metrics
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id2label: Optional[Dict[int, str]] = None,
    output_dict: bool = False,
) -> str:
    """
    Generate a detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        id2label: Optional mapping from label IDs to names
        output_dict: If True, return dict instead of string

    Returns:
        Classification report as string or dict
    """
    target_names, labels = None, None
    if id2label:
        # Sort by integer id, not lexicographically: after a JSON round trip the
        # keys are strings, and sorted() would give 0, 1, 10, 11, ... 2, 3, so
        # every row from id 2 onwards would carry another sector's name.
        labels = sorted(int(k) for k in id2label)
        target_names = [str(id2label.get(i, id2label.get(str(i)))) for i in labels]

    # labels= is required: without it sklearn infers the classes present in the
    # data, and any sector absent from a slice makes the lengths disagree.
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0,
    )

    return report


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization mode ('true', 'pred', 'all', or None)

    Returns:
        Confusion matrix as numpy array
    """
    return confusion_matrix(y_true, y_pred, normalize=normalize)


def log_evaluation_results(
    metrics: Dict[str, float],
    split_name: str = "Evaluation",
) -> None:
    """
    Log evaluation results in a formatted way.

    Args:
        metrics: Dictionary of metric names to values
        split_name: Name of the evaluation split
    """
    logger.info(f"\n{'=' * 40}")
    logger.info(f"{split_name.upper()} RESULTS")
    logger.info(f"{'=' * 40}")

    for metric_name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{metric_name.capitalize()}: {value:.4f}")
        else:
            logger.info(f"{metric_name.capitalize()}: {value}")

    logger.info(f"{'=' * 40}\n")


def compare_results(
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    threshold: float = 0.05,
) -> Dict[str, Any]:
    """
    Compare validation and test results to detect overfitting.

    Args:
        val_metrics: Validation metrics
        test_metrics: Test metrics
        threshold: Gap threshold for overfitting warning

    Returns:
        Dictionary with comparison results
    """
    comparison = {
        "validation": val_metrics,
        "test": test_metrics,
        "gaps": {},
        "status": "good",
    }

    for metric in ["f1", "accuracy"]:
        if metric in val_metrics and metric in test_metrics:
            gap = val_metrics[metric] - test_metrics[metric]
            comparison["gaps"][metric] = gap

            if gap > threshold:
                comparison["status"] = "potential_overfitting"
                logger.warning(
                    f"Potential overfitting detected: {metric} gap = {gap:.4f}"
                )
            elif gap < -threshold:
                comparison["status"] = "unusual"
                logger.info(
                    f"Test performance better than validation (unusual): "
                    f"{metric} gap = {gap:.4f}"
                )

    return comparison
