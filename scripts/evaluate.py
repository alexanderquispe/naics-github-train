#!/usr/bin/env python
"""
Evaluation Script for NAICS GitHub Repository Classifier.

This script evaluates a trained model on test data and generates
detailed performance reports.

Usage:
    python scripts/evaluate.py --model models/modernbert-naics-classifier
    python scripts/evaluate.py --model models/classifier --test-data data/raw/test.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import RAW_DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from src.data_loader import (
    load_parquet_data,
    prepare_naics_dataset,
    create_dataset_splits,
    tokenize_dataset,
)
from src.inference import load_trained_model
from src.metrics import (
    compute_detailed_metrics,
    generate_classification_report,
    get_confusion_matrix,
    log_evaluation_results,
)
from src.visualization import (
    plot_confusion_matrix,
    plot_class_performance,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate NAICS classifier on test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default=str(RAW_DATA_DIR / "train_data_gpt_ab8score.parquet"),
        help="Path to test data (parquet file)",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="code",
        help="Column name containing NAICS codes",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Path to save evaluation report",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show evaluation plots",
    )
    parser.add_argument(
        "--save-plots",
        type=str,
        default=None,
        help="Directory to save plots",
    )

    return parser.parse_args()


def evaluate_on_dataset(model, tokenizer, dataset, batch_size=16):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        tokenizer: Model tokenizer
        dataset: Tokenized dataset
        batch_size: Batch size for evaluation

    Returns:
        Tuple of (predictions, labels)
    """
    device = next(model.parameters()).device
    model.eval()

    all_predictions = []
    all_labels = []

    # Process in batches
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        # Prepare inputs
        inputs = {
            "input_ids": torch.tensor(batch["input_ids"]).to(device),
            "attention_mask": torch.tensor(batch["attention_mask"]).to(device),
        }

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        labels = np.array(batch["labels"])

        all_predictions.extend(predictions)
        all_labels.extend(labels)

    return np.array(all_predictions), np.array(all_labels)


def main():
    """Main evaluation pipeline."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("NAICS GitHub Repository Classifier - Evaluation")
    logger.info("=" * 60)

    # Check model path
    model_path = Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)

    logger.info(f"Model: {model_path}")
    logger.info(f"Test data: {args.test_data}")

    # Load model
    logger.info("\n" + "=" * 40)
    logger.info("Loading Model")
    logger.info("=" * 40)

    model, tokenizer, label_mappings = load_trained_model(model_path)

    if label_mappings:
        id2label = label_mappings.get("id2label", {})
        label2id = label_mappings.get("label2id", {})
        logger.info(f"Loaded {len(id2label)} label mappings")
    else:
        logger.warning("No label mappings found, using indices")
        id2label = {}
        label2id = {}

    # Load and prepare test data
    logger.info("\n" + "=" * 40)
    logger.info("Loading Test Data")
    logger.info("=" * 40)

    test_data_path = Path(args.test_data)
    if not test_data_path.exists():
        logger.error(f"Test data not found: {test_data_path}")
        sys.exit(1)

    raw_data = load_parquet_data(test_data_path)
    logger.info(f"Loaded {len(raw_data)} examples")

    # Prepare dataset
    processed_df, data_label2id, data_id2label = prepare_naics_dataset(
        raw_data,
        target_column=args.target_column,
    )

    # Use data labels if model labels not available
    if not id2label:
        id2label = data_id2label
        label2id = data_label2id

    # Create test split
    dataset_dict = create_dataset_splits(
        processed_df,
        test_size=0.2,
        val_size=0.1,
    )

    # Tokenize
    tokenized_dataset = tokenize_dataset(
        dataset_dict,
        tokenizer,
        max_length=args.max_seq_length,
    )

    test_dataset = tokenized_dataset["test"]
    logger.info(f"Test set size: {len(test_dataset)}")

    # Evaluate
    logger.info("\n" + "=" * 40)
    logger.info("Running Evaluation")
    logger.info("=" * 40)

    predictions, labels = evaluate_on_dataset(
        model,
        tokenizer,
        test_dataset,
        batch_size=args.batch_size,
    )

    # Compute metrics
    logger.info("\n" + "=" * 40)
    logger.info("Computing Metrics")
    logger.info("=" * 40)

    metrics = compute_detailed_metrics(labels, predictions)
    log_evaluation_results(metrics, "Test")

    # Generate classification report
    logger.info("\n" + "=" * 40)
    logger.info("Classification Report")
    logger.info("=" * 40)

    report = generate_classification_report(labels, predictions, id2label)
    print(report)

    # Get report as dict for plotting
    report_dict = generate_classification_report(
        labels, predictions, id2label, output_dict=True
    )

    # Save report if requested
    if args.output_report:
        report_path = Path(args.output_report)
        with open(report_path, "w") as f:
            f.write("NAICS Classifier Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Test Data: {args.test_data}\n")
            f.write(f"Test Size: {len(test_dataset)}\n\n")
            f.write("Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")
            f.write("Classification Report:\n")
            f.write(report)
        logger.info(f"Report saved to: {report_path}")

    # Plot if requested
    if args.plot or args.save_plots:
        # Get label names
        label_names = [str(id2label.get(str(i), id2label.get(i, i))) for i in sorted(set(labels))]

        # Confusion matrix
        save_path = None
        if args.save_plots:
            save_dir = Path(args.save_plots)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "confusion_matrix.png"

        plot_confusion_matrix(
            labels,
            predictions,
            labels=label_names,
            title="NAICS Classification Confusion Matrix",
            save_path=save_path,
        )

        # Per-class performance
        if args.save_plots:
            save_path = save_dir / "class_performance.png"

        plot_class_performance(
            report_dict,
            metric="f1-score",
            title="Per-Class F1 Score",
            save_path=save_path,
        )

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test F1 Score: {metrics['f1']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
