"""
Model Training Pipeline for NAICS Classification.

This module provides functions for setting up and training
transformer models for NAICS classification.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import torch
from datasets import DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from .metrics import compute_metrics

logger = logging.getLogger(__name__)


def setup_model(
    model_id: str,
    num_labels: int,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Set up the model and tokenizer for training.

    Args:
        model_id: Hugging Face model identifier
        num_labels: Number of classification labels
        label2id: Mapping from labels to IDs
        id2label: Mapping from IDs to labels

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Setting up model: {model_id}")
    logger.info(f"Number of labels: {num_labels}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model
    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model, tokenizer


def get_training_args(
    output_dir: str = "naics-classifier",
    num_epochs: int = 8,
    batch_size: int = 8,
    eval_batch_size: int = 16,
    learning_rate: float = 1.5e-5,
    weight_decay: float = 0.02,
    warmup_ratio: float = 0.15,
    gradient_accumulation_steps: int = 2,
    max_grad_norm: float = 1.0,
    lr_scheduler_type: str = "polynomial",
    eval_steps: int = 100,
    save_steps: int = 100,
    save_total_limit: int = 5,
    use_bf16: bool = True,
    use_fused_optimizer: bool = True,
    seed: int = 42,
    **kwargs,
) -> TrainingArguments:
    """
    Create training arguments for the Trainer.

    Args:
        output_dir: Directory for saving checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size per device
        eval_batch_size: Evaluation batch size per device
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        warmup_ratio: Proportion of training for warmup
        gradient_accumulation_steps: Steps to accumulate gradients
        max_grad_norm: Maximum gradient norm for clipping
        lr_scheduler_type: Learning rate scheduler type
        eval_steps: Evaluation frequency in steps
        save_steps: Checkpoint save frequency in steps
        save_total_limit: Maximum checkpoints to keep
        use_bf16: Whether to use bfloat16 precision
        use_fused_optimizer: Whether to use fused AdamW optimizer
        seed: Random seed
        **kwargs: Additional TrainingArguments parameters

    Returns:
        TrainingArguments object
    """
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()

    # Determine optimizer
    optim = "adamw_torch_fused" if use_fused_optimizer and cuda_available else "adamw_torch"

    # Determine precision
    bf16 = use_bf16 and cuda_available

    training_args = TrainingArguments(
        output_dir=output_dir,
        # Training parameters
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        # Optimization
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=max_grad_norm,
        bf16=bf16,
        optim=optim,
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        # Logging
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
        # Reproducibility
        seed=seed,
        # Additional kwargs
        **kwargs,
    )

    logger.info(f"Training arguments configured:")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  BF16: {bf16}")

    return training_args


def get_callbacks(
    early_stopping_patience: int = 2,
    early_stopping_threshold: float = 0.001,
) -> list:
    """
    Get training callbacks.

    Args:
        early_stopping_patience: Number of evaluations with no improvement before stopping
        early_stopping_threshold: Minimum improvement to be considered significant

    Returns:
        List of callbacks
    """
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
        )
    ]
    return callbacks


def train_model(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    tokenized_dataset: DatasetDict,
    training_args: TrainingArguments,
    early_stopping_patience: int = 2,
    early_stopping_threshold: float = 0.001,
) -> Tuple[Trainer, Dict[str, Any]]:
    """
    Train the model using the Hugging Face Trainer.

    Args:
        model: The model to train
        tokenizer: The tokenizer
        tokenized_dataset: Tokenized dataset with train/validation/test splits
        training_args: Training configuration
        early_stopping_patience: Early stopping patience
        early_stopping_threshold: Early stopping threshold

    Returns:
        Tuple of (trainer, training_results)
    """
    logger.info("Creating Trainer...")

    # Get callbacks
    callbacks = get_callbacks(early_stopping_patience, early_stopping_threshold)

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Log dataset sizes
    logger.info(f"Training examples: {len(tokenized_dataset['train'])}")
    logger.info(f"Validation examples: {len(tokenized_dataset['validation'])}")
    if "test" in tokenized_dataset:
        logger.info(f"Test examples: {len(tokenized_dataset['test'])}")

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    logger.info("Training completed!")
    logger.info(f"Training loss: {train_result.training_loss:.4f}")

    return trainer, train_result


def evaluate_model(
    trainer: Trainer,
    tokenized_dataset: DatasetDict,
    split: str = "test",
) -> Dict[str, float]:
    """
    Evaluate the model on a dataset split.

    Args:
        trainer: Trained Trainer object
        tokenized_dataset: Dataset with the split to evaluate
        split: Dataset split to evaluate ('validation' or 'test')

    Returns:
        Dictionary of evaluation metrics
    """
    if split not in tokenized_dataset:
        raise ValueError(f"Split '{split}' not found in dataset")

    logger.info(f"Evaluating on {split} set...")
    results = trainer.evaluate(eval_dataset=tokenized_dataset[split])

    logger.info(f"{split.capitalize()} Results:")
    logger.info(f"  F1: {results['eval_f1']:.4f}")
    logger.info(f"  Accuracy: {results['eval_accuracy']:.4f}")
    logger.info(f"  Precision: {results['eval_precision']:.4f}")
    logger.info(f"  Recall: {results['eval_recall']:.4f}")

    return results


def save_model(
    trainer: Trainer,
    tokenizer: AutoTokenizer,
    save_path: str,
    label2id: Optional[Dict[str, int]] = None,
    id2label: Optional[Dict[int, str]] = None,
) -> None:
    """
    Save the trained model and tokenizer.

    Args:
        trainer: Trained Trainer object
        tokenizer: The tokenizer
        save_path: Path to save the model
        label2id: Optional label to ID mapping to save
        id2label: Optional ID to label mapping to save
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model to {save_path}")

    # Save model
    trainer.save_model(str(save_path))

    # Save tokenizer
    tokenizer.save_pretrained(str(save_path))

    # Save label mappings if provided
    if label2id and id2label:
        import json

        mappings = {"label2id": label2id, "id2label": id2label}
        mappings_path = save_path / "label_mappings.json"
        with open(mappings_path, "w") as f:
            json.dump(mappings, f, indent=2)
        logger.info(f"Label mappings saved to {mappings_path}")

    logger.info("Model saved successfully!")
