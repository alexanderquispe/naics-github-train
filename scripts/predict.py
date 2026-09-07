#!/usr/bin/env python
"""
Prediction Script for NAICS GitHub Repository Classifier.

This script makes predictions on new repository data using a trained model.

Usage:
    python scripts/predict.py --model models/modernbert-naics-classifier --input "Banking API for transactions"
    python scripts/predict.py --model models/classifier --repo-name bank-api --description "Banking API"
    python scripts/predict.py --model models/classifier --input-file repos.csv --output predictions.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import MODELS_DIR
from src.inference import (
    load_trained_model,
    predict_naics,
    batch_predict,
    format_repository_input,
)
from src.naics_mapping import get_naics_description

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict NAICS codes for GitHub repositories",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory",
    )

    # Input options (mutually exclusive groups)
    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument(
        "--input",
        type=str,
        help="Direct text input for prediction",
    )
    input_group.add_argument(
        "--repo-name",
        type=str,
        help="Repository name",
    )
    input_group.add_argument(
        "--description",
        type=str,
        help="Repository description",
    )
    input_group.add_argument(
        "--topics",
        type=str,
        help="Repository topics (comma-separated)",
    )
    input_group.add_argument(
        "--readme",
        type=str,
        help="README content or path to README file",
    )
    input_group.add_argument(
        "--input-file",
        type=str,
        help="Path to CSV/parquet file with repositories",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions (CSV or JSON)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of top predictions to return",
    )
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="Show confidence scores",
    )
    parser.add_argument(
        "--show-description",
        action="store_true",
        help="Show NAICS code descriptions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Compute device. Auto-detected (CUDA, then MPS, then CPU) when omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for file predictions",
    )

    return parser.parse_args()


def _readme_argument(value):
    """--readme takes README content or a path to a file holding it.

    Path.exists() raises OSError on anything over 255 bytes per path component,
    so the length has to be checked first: a real README is text, not a path.
    """
    if not value:
        return value
    if len(value) < 255 and "\n" not in value:
        candidate = Path(value)
        try:
            if candidate.is_file():
                return candidate.read_text()
        except OSError:
            pass
    return value


def _first_column(df, names):
    """First of `names` present in df, or None."""
    for name in names:
        if name in df.columns:
            return name
    return None


def predict_single(
    model,
    tokenizer,
    label_mappings,
    text: str,
    top_k: int = 1,
    show_confidence: bool = False,
    show_description: bool = False,
):
    """Make a single prediction and display results."""
    result = predict_naics(
        text=text,
        model=model,
        tokenizer=tokenizer,
        label_mappings=label_mappings,
        return_all_scores=(top_k > 1),
    )

    print("\n" + "=" * 50)
    print("Prediction Result")
    print("=" * 50)

    if top_k == 1:
        naics_code = result["predicted_naics"]
        print(f"NAICS Code: {naics_code}")

        if show_description:
            desc = get_naics_description(naics_code)
            if desc:
                print(f"Description: {desc}")

        if show_confidence:
            print(f"Confidence: {result['confidence']:.4f}")
    else:
        # Show top-k predictions
        if "all_scores" in result:
            sorted_scores = sorted(
                result["all_scores"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:top_k]

            print(f"Top {top_k} Predictions:")
            for i, (code, score) in enumerate(sorted_scores, 1):
                line = f"  {i}. NAICS {code}"
                if show_description:
                    desc = get_naics_description(code)
                    if desc:
                        line += f" - {desc}"
                if show_confidence:
                    line += f" (confidence: {score:.4f})"
                print(line)

    print("=" * 50)
    return result


def predict_from_file(
    model,
    tokenizer,
    label_mappings,
    input_path: str,
    output_path: Optional[str],
    batch_size: int,
    show_description: bool,
):
    """Make predictions on a file of repositories."""
    input_path = Path(input_path)

    # Load data
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    logger.info(f"Loaded {len(df)} repositories from {input_path}")

    # Accept the column names the README documents and the ones
    # scripts/inference_batch.py accepts, and fail loudly if the repository name
    # or the README cannot be found rather than classifying blank text.
    name_col = _first_column(df, ["name_repo", "repo", "name", "nwo", "repo_name"])
    readme_col = _first_column(df, ["readme_content", "readme", "readme_text"])
    desc_col = _first_column(df, ["description", "desc"])
    topics_col = _first_column(df, ["topics", "topic"])
    logger.info(
        f"Using columns: name={name_col}, description={desc_col}, "
        f"topics={topics_col}, readme={readme_col}"
    )
    if name_col is None and readme_col is None:
        logger.error(
            "Neither a repository-name column (name_repo, repo, name, nwo, "
            "repo_name) nor a README column (readme_content, readme, "
            "readme_text) was found. Columns present: %s", list(df.columns)
        )
        sys.exit(1)

    texts = []
    for _, row in df.iterrows():
        text = format_repository_input(
            repo_name=row.get(name_col) if name_col else None,
            description=row.get(desc_col) if desc_col else None,
            topics=row.get(topics_col) if topics_col else None,
            readme=row.get(readme_col) if readme_col else None,
        )
        texts.append(text)

    # Make predictions
    logger.info("Making predictions...")
    results = batch_predict(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        label_mappings=label_mappings,
        batch_size=batch_size,
    )

    # Add predictions to dataframe
    df["predicted_naics"] = [r["predicted_naics"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]

    if show_description:
        df["naics_description"] = df["predicted_naics"].apply(get_naics_description)

    # Save or display
    if output_path:
        output_path = Path(output_path)
        if output_path.suffix == ".csv":
            df.to_csv(output_path, index=False)
        elif output_path.suffix == ".json":
            df.to_json(output_path, orient="records", indent=2)
        elif output_path.suffix == ".parquet":
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)

        logger.info(f"Predictions saved to {output_path}")
    else:
        # Display summary
        print("\n" + "=" * 50)
        print("Prediction Summary")
        print("=" * 50)
        print(f"Total repositories: {len(df)}")
        print(f"\nNAICS Distribution:")
        print(df["predicted_naics"].value_counts())

        print("\nSample Predictions:")
        sample_cols = ["predicted_naics", "confidence"]
        if "name_repo" in df.columns:
            sample_cols = ["name_repo"] + sample_cols
        if show_description and "naics_description" in df.columns:
            sample_cols.append("naics_description")

        print(df[sample_cols].head(10).to_string())

    return df


def main():
    """Main prediction pipeline."""
    args = parse_args()

    # A local directory or a Hugging Face id; load_trained_model validates both.
    model_path = args.model

    logger.info(f"Loading model from {model_path}")
    model, tokenizer, label_mappings = load_trained_model(model_path, device=args.device)

    # Flags that only apply to one mode used to be accepted and dropped in
    # silence. Say so instead, so nobody waits for a file that is never written.
    if args.input_file and args.top_k > 1:
        logger.warning("--top-k applies to single predictions only; file mode writes the top label per row")
    if not args.input_file and args.output:
        logger.warning("--output applies to --input-file only; single predictions are printed, not written")

    # Determine input type and make predictions
    if args.input_file:
        # Batch prediction from file
        predict_from_file(
            model=model,
            tokenizer=tokenizer,
            label_mappings=label_mappings,
            input_path=args.input_file,
            output_path=args.output,
            batch_size=args.batch_size,
            show_description=args.show_description,
        )

    elif args.input:
        # Direct text input
        predict_single(
            model=model,
            tokenizer=tokenizer,
            label_mappings=label_mappings,
            # through the shared builder, so a bare sentence gets the same
            # scaffolding and cleaning the model was trained on
            text=format_repository_input(description=args.input),
            top_k=args.top_k,
            show_confidence=args.show_confidence,
            show_description=args.show_description,
        )

    elif args.repo_name or args.description or args.readme:
        # Build input from components
        readme_content = _readme_argument(args.readme)

        text = format_repository_input(
            repo_name=args.repo_name,
            description=args.description,
            topics=args.topics,
            readme=readme_content,
        )

        predict_single(
            model=model,
            tokenizer=tokenizer,
            label_mappings=label_mappings,
            text=text,
            top_k=args.top_k,
            show_confidence=args.show_confidence,
            show_description=args.show_description,
        )

    else:
        # Interactive mode
        print("\nNAICS Repository Classifier - Interactive Mode")
        print("Enter repository information (Ctrl+C to exit)")
        print("-" * 50)

        while True:
            try:
                print("\nEnter text to classify (or 'q' to quit):")
                text = input("> ").strip()

                if text.lower() == "q":
                    break

                if text:
                    predict_single(
                        model=model,
                        tokenizer=tokenizer,
                        label_mappings=label_mappings,
                        text=text,
                        top_k=args.top_k,
                        show_confidence=args.show_confidence,
                        show_description=args.show_description,
                    )
            except KeyboardInterrupt:
                print("\nExiting...")
                break

    return 0


if __name__ == "__main__":
    sys.exit(main())
