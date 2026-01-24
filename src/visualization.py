"""
Visualization Functions for NAICS Classification.

This module provides functions for plotting label distributions,
confusion matrices, and training history.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_label_distribution(
    df: pd.DataFrame,
    target_column: str = "label",
    title: str = "NAICS Categories Distribution",
    figsize: tuple = (15, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot the distribution of NAICS categories.

    Args:
        df: DataFrame with label column
        target_column: Name of the target column
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=figsize)

    label_counts = df[target_column].value_counts().sort_index()

    # Bar plot
    plt.subplot(1, 2, 1)
    ax = label_counts.plot(kind="bar", color="steelblue", edgecolor="black")
    plt.title(f"{title} - Bar Chart")
    plt.xlabel("NAICS Code")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")

    # Add value labels on bars
    for i, v in enumerate(label_counts.values):
        ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=8)

    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(
        label_counts.values,
        labels=label_counts.index,
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.title(f"{title} - Pie Chart")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    plt.show()

    # Print summary statistics
    logger.info(f"Distribution: {len(label_counts)} categories")
    logger.info(f"Range: {label_counts.min()} to {label_counts.max()} examples per category")
    logger.info(f"Total: {label_counts.sum()} examples")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: tuple = (12, 10),
    normalize: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of label names
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize the matrix
        save_path: Optional path to save the figure
    """
    from sklearn.metrics import confusion_matrix

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels if labels else "auto",
        yticklabels=labels if labels else "auto",
        square=True,
    )

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = None,
    title: str = "Training History",
    figsize: tuple = (12, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot training history metrics.

    Args:
        history: Dictionary mapping metric names to lists of values
        metrics: Specific metrics to plot (default: all)
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    if metrics is None:
        metrics = list(history.keys())

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)

    if num_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric in history:
            values = history[metric]
            ax.plot(values, marker="o", markersize=3)
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Step/Epoch")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    plt.show()


def plot_class_performance(
    report_dict: Dict,
    metric: str = "f1-score",
    title: str = "Per-Class Performance",
    figsize: tuple = (14, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot per-class performance metrics.

    Args:
        report_dict: Classification report as dictionary
        metric: Metric to plot ('precision', 'recall', 'f1-score')
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    # Extract class-level metrics (exclude summary rows)
    classes = []
    values = []

    for class_name, metrics in report_dict.items():
        if isinstance(metrics, dict) and metric in metrics:
            if class_name not in ["accuracy", "macro avg", "weighted avg"]:
                classes.append(class_name)
                values.append(metrics[metric])

    # Sort by value
    sorted_pairs = sorted(zip(values, classes), reverse=True)
    values, classes = zip(*sorted_pairs)

    plt.figure(figsize=figsize)

    # Horizontal bar plot
    colors = plt.cm.RdYlGn([v for v in values])
    bars = plt.barh(range(len(classes)), values, color=colors, edgecolor="black")

    plt.yticks(range(len(classes)), classes)
    plt.xlabel(metric.replace("-", " ").title())
    plt.title(f"{title} - {metric.replace('-', ' ').title()}")
    plt.xlim(0, 1.0)

    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    plt.show()


def plot_text_length_distribution(
    df: pd.DataFrame,
    text_column: str = "text",
    title: str = "Text Length Distribution",
    figsize: tuple = (10, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot distribution of text lengths.

    Args:
        df: DataFrame with text column
        text_column: Name of the text column
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    lengths = df[text_column].str.len()

    plt.figure(figsize=figsize)

    plt.subplot(1, 2, 1)
    plt.hist(lengths, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Text Length (characters)")
    plt.ylabel("Frequency")
    plt.title(f"{title}")
    plt.axvline(lengths.mean(), color="red", linestyle="--", label=f"Mean: {lengths.mean():.0f}")
    plt.axvline(lengths.median(), color="green", linestyle="--", label=f"Median: {lengths.median():.0f}")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.boxplot(lengths, vert=True)
    plt.ylabel("Text Length (characters)")
    plt.title("Box Plot")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {save_path}")

    plt.show()

    # Print statistics
    logger.info(f"Text length statistics:")
    logger.info(f"  Min: {lengths.min()}")
    logger.info(f"  Max: {lengths.max()}")
    logger.info(f"  Mean: {lengths.mean():.0f}")
    logger.info(f"  Median: {lengths.median():.0f}")
    logger.info(f"  Std: {lengths.std():.0f}")
