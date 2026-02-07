#!/usr/bin/env python
"""
Training Script for NAICS GitHub Repository Classifier.

This script provides a CLI interface for training transformer models
to classify GitHub repositories into NAICS codes.

Usage:
    python scripts/train.py --model modernbert-base --data data/raw/train_data_naics_github.parquet
    python scripts/train.py --model deberta-v3-base --epochs 10 --batch-size 8
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    SUPPORTED_MODELS,
    RAW_DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    ModelConfig,
    TrainingConfig,
    DataConfig,
)
from src.data_loader import (
    load_parquet_data,
    prepare_naics_dataset,
    create_dataset_splits,
    tokenize_dataset,
)
from src.trainer import (
    setup_model,
    get_training_args,
    train_model,
    evaluate_model,
    save_model,
)
from src.visualization import plot_label_distribution

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUTS_DIR / "training.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NAICS classifier on GitHub repository data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="modernbert-base",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Model architecture to use",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length for tokenization",
    )

    # Data arguments
    parser.add_argument(
        "--data",
        type=str,
        default=str(RAW_DATA_DIR / "train_data_naics_github.parquet"),
        help="Path to training data (parquet file)",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="code",
        help="Column name containing NAICS codes",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=80,
        help="Minimum samples per class. Classes with fewer samples are excluded.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for test set",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Proportion of data for validation set",
    )

    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1.5e-5,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.02,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.15,
        help="Proportion of training for warmup",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Early stopping patience (evaluations)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage (allows larger batch sizes)",
    )

    # Output arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for model (default: models/<model-name>-naics-classifier)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show data distribution plots",
    )

    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bfloat16 training",
    )

    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("NAICS GitHub Repository Classifier - Training")
    logger.info("=" * 60)

    # Log configuration
    logger.info(f"Model: {args.model}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Min samples per class: {args.min_samples}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Early stopping patience: {args.early_stopping_patience}")
    logger.info(f"Gradient checkpointing: {args.gradient_checkpointing}")

    # Set output directory
    if args.output is None:
        output_dir = MODELS_DIR / f"{args.model}-naics-classifier"
    else:
        output_dir = Path(args.output)

    logger.info(f"Output directory: {output_dir}")

    # Load data
    logger.info("\n" + "=" * 40)
    logger.info("Loading Data")
    logger.info("=" * 40)

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    raw_data = load_parquet_data(data_path)
    logger.info(f"Loaded {len(raw_data)} examples")

    # Prepare dataset
    logger.info("\n" + "=" * 40)
    logger.info("Preparing Dataset")
    logger.info("=" * 40)

    processed_df, label2id, id2label = prepare_naics_dataset(
        raw_data,
        target_column=args.target_column,
        min_samples_per_class=args.min_samples,
    )

    logger.info(f"Processed {len(processed_df)} examples")
    logger.info(f"Number of classes: {len(label2id)}")

    # Plot distribution if requested
    if args.plot:
        plot_label_distribution(
            processed_df,
            target_column="label",
            title="NAICS Categories Distribution",
        )

    # Create splits
    logger.info("\n" + "=" * 40)
    logger.info("Creating Data Splits")
    logger.info("=" * 40)

    dataset_dict = create_dataset_splits(
        processed_df,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    # Setup model
    logger.info("\n" + "=" * 40)
    logger.info("Setting Up Model")
    logger.info("=" * 40)

    model_id = SUPPORTED_MODELS[args.model]
    num_labels = len(label2id)

    model, tokenizer = setup_model(
        model_id=model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Tokenize dataset
    logger.info("\n" + "=" * 40)
    logger.info("Tokenizing Dataset")
    logger.info("=" * 40)

    # Adjust max_seq_length based on model
    max_seq_length = args.max_seq_length
    if "modernbert" not in args.model:
        max_seq_length = min(max_seq_length, 512)
        logger.info(f"Adjusted max_seq_length to {max_seq_length} for {args.model}")

    tokenized_dataset = tokenize_dataset(
        dataset_dict,
        tokenizer,
        max_length=max_seq_length,
    )

    # Setup training arguments
    logger.info("\n" + "=" * 40)
    logger.info("Configuring Training")
    logger.info("=" * 40)

    training_args = get_training_args(
        output_dir=str(output_dir),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        use_bf16=not args.no_bf16,
        seed=args.seed,
    )

    # Train model
    logger.info("\n" + "=" * 40)
    logger.info("Training Model")
    logger.info("=" * 40)

    trainer, train_result = train_model(
        model=model,
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_dataset,
        training_args=training_args,
        early_stopping_patience=args.early_stopping_patience,
    )

    # Evaluate on validation set
    logger.info("\n" + "=" * 40)
    logger.info("Validation Evaluation")
    logger.info("=" * 40)

    val_results = evaluate_model(trainer, tokenized_dataset, split="validation")

    # Evaluate on test set
    logger.info("\n" + "=" * 40)
    logger.info("Test Evaluation")
    logger.info("=" * 40)

    test_results = evaluate_model(trainer, tokenized_dataset, split="test")

    # Compare results
    logger.info("\n" + "=" * 40)
    logger.info("Performance Summary")
    logger.info("=" * 40)

    logger.info(f"Validation F1: {val_results['eval_f1']:.4f}")
    logger.info(f"Test F1: {test_results['eval_f1']:.4f}")

    gap = val_results["eval_f1"] - test_results["eval_f1"]
    if gap > 0.05:
        logger.warning(f"Potential overfitting detected! Gap: {gap:.4f}")
    else:
        logger.info(f"Good generalization. Gap: {gap:.4f}")

    # Save model
    logger.info("\n" + "=" * 40)
    logger.info("Saving Model")
    logger.info("=" * 40)

    save_model(
        trainer=trainer,
        tokenizer=tokenizer,
        save_path=str(output_dir),
        label2id=label2id,
        id2label=id2label,
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Final Test F1: {test_results['eval_f1']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
