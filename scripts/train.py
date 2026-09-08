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
import json
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    SUPPORTED_MODELS,
    LONG_CONTEXT_MODELS,
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
    count_training_steps,
    find_last_checkpoint,
)
from src.visualization import plot_label_distribution
from transformers import set_seed

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
        default=512,
        help="Tokens per example. 512 is what every published model was trained "
             "with; ModernBERT and BGE-M3 accept more if you ask for it.",
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
        "--gradient-accumulation-steps",
        type=int,
        default=2,
        help="Micro-batches summed before each optimizer step. The effective "
             "batch is --batch-size times this.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the newest checkpoint in the output directory. A run "
             "interrupted by the machine sleeping or a crash leaves one every "
             "--eval-steps steps; without this the run starts over.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluate (and checkpoint) every N optimizer steps",
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

    # Seed before anything builds a model. The Trainer seeds too, but by then
    # setup_model has already drawn the classification head from an unseeded
    # generator, which is what made --seed decorative.
    set_seed(args.seed)

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
    logger.info(f"Seed: {args.seed}")
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

    # Resolve the sequence length. 512 unless the model is long-context AND the
    # caller asked for more, and never beyond what the positional embeddings
    # allow. The resolved value is written next to the model so evaluate.py
    # cannot silently use a different one.
    max_seq_length = args.max_seq_length
    if not any(k in args.model for k in LONG_CONTEXT_MODELS):
        max_seq_length = min(max_seq_length, 512)
    model_limit = getattr(model.config, "max_position_embeddings", None)
    if model_limit:
        usable = model_limit - 2 if model_limit <= 1024 else model_limit
        max_seq_length = min(max_seq_length, usable)
    logger.info(f"Sequence length: {max_seq_length} "
                f"(requested {args.max_seq_length}, model limit {model_limit})")

    tokenized_dataset = tokenize_dataset(
        dataset_dict,
        tokenizer,
        max_length=max_seq_length,
    )

    # Setup training arguments
    logger.info("\n" + "=" * 40)
    logger.info("Configuring Training")
    logger.info("=" * 40)

    total_steps = count_training_steps(
        num_examples=len(tokenized_dataset["train"]),
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.epochs,
    )
    logger.info(f"Optimizer steps: {total_steps} ({args.batch_size} x "
                f"{args.gradient_accumulation_steps} = effective batch "
                f"{args.batch_size * args.gradient_accumulation_steps})")

    training_args = get_training_args(
        output_dir=str(output_dir),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        use_bf16=not args.no_bf16,
        seed=args.seed,
        num_training_steps=total_steps,
    )

    # Train model
    logger.info("\n" + "=" * 40)
    logger.info("Training Model")
    logger.info("=" * 40)

    checkpoint = find_last_checkpoint(output_dir) if args.resume else None
    if args.resume and checkpoint is None:
        logger.warning(f"--resume given but no checkpoint found in {output_dir}; starting fresh")

    trainer, train_result = train_model(
        model=model,
        tokenizer=tokenizer,
        tokenized_dataset=tokenized_dataset,
        training_args=training_args,
        early_stopping_patience=args.early_stopping_patience,
        resume_from_checkpoint=checkpoint,
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

    # Record the settings that define the split and the input, so evaluate.py
    # can reuse them instead of guessing. Evaluating with a different
    # min_samples or seed silently scores the model on rows it trained on.
    training_config = {
        "model": args.model,
        "model_id": model_id,
        "data": str(data_path),
        "max_seq_length": max_seq_length,
        "min_samples_per_class": args.min_samples,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "eval_steps": args.eval_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "optimizer_steps": total_steps,
        "num_classes": len(label2id),
        "n_train": len(tokenized_dataset["train"]),
        "n_validation": len(tokenized_dataset["validation"]),
        "n_test": len(tokenized_dataset["test"]),
        "val_f1": val_results.get("eval_f1"),
        "test_f1": test_results.get("eval_f1"),
        "test_accuracy": test_results.get("eval_accuracy"),
    }
    (output_dir / "training_config.json").write_text(json.dumps(training_config, indent=2))
    logger.info(f"Training configuration saved to {output_dir / 'training_config.json'}")

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {output_dir}")
    logger.info(f"Final Test F1: {test_results['eval_f1']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
