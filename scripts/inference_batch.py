"""
Batch inference script for NAICS classification.

Usage:
    python scripts/inference_batch.py --input data/raw/claude_repos_text_fields.parquet --output predictions.parquet
    python scripts/inference_batch.py --input data.parquet --output results.parquet --batch-size 32 --limit 1000
"""

import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# NAICS code descriptions
NAICS_DESCRIPTIONS = {
    "11": "Agriculture, Forestry, Fishing and Hunting",
    "21": "Mining, Quarrying, Oil and Gas Extraction",
    "22": "Utilities",
    "23": "Construction",
    "31-33": "Manufacturing",
    "42": "Wholesale Trade",
    "44-45": "Retail Trade",
    "48-49": "Transportation and Warehousing",
    "51": "Information",
    "52": "Finance and Insurance",
    "53": "Real Estate and Rental",
    "54": "Professional, Scientific, Technical Services",
    "56": "Administrative and Support Services",
    "61": "Educational Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
    "72": "Accommodation and Food Services",
    "81": "Other Services",
    "92": "Public Administration",
}


def clean_readme_text(text: str) -> str:
    """
    Clean README text by removing markdown artifacts, code blocks, and noise.
    This is the SAME cleaning function used during training.

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


def format_input_text(
    name: str,
    description: Optional[str] = None,
    topics: Optional[str] = None,
    readme: Optional[str] = None,
    max_readme_chars: int = 5000,
    clean_text: bool = True,
) -> str:
    """
    Format repository data into the expected input format for the model.
    Applies the SAME preprocessing as used during training.

    Args:
        name: Repository name
        description: Repository description
        topics: Topics/tags (can be string or list)
        readme: README content
        max_readme_chars: Maximum characters to include from README
        clean_text: Whether to apply text cleaning (should match training)

    Returns:
        Formatted text string
    """
    components = []

    # Repository name
    if name and str(name).strip():
        components.append(f"Repository: {name}")

    # Description
    if description and str(description).strip() and str(description) != "nan":
        components.append(f"Description: {description}")

    # Topics
    if topics and str(topics).strip() and str(topics) not in ["nan", "[]", ""]:
        # Handle list or string format
        if isinstance(topics, list):
            topics_str = "; ".join(str(t) for t in topics if t)
        else:
            topics_str = str(topics).replace("[", "").replace("]", "").replace(",", ";").replace("'", "")
        if topics_str.strip():
            components.append(f"Topics: {topics_str}")

    # README content (truncate if too long)
    if readme and str(readme).strip() and str(readme) != "nan":
        readme_text = str(readme)[:max_readme_chars]
        components.append(f"README: {readme_text}")

    # Combine components
    combined_text = " | ".join(components)

    # Apply same cleaning as training
    if clean_text:
        combined_text = clean_readme_text(combined_text)

    return combined_text


def load_model(model_name: str, device: Optional[str] = None, token: Optional[str] = None):
    """
    Load the model and tokenizer from Hugging Face.

    Args:
        model_name: Hugging Face model ID or local path
        device: Device to use ('cuda', 'cpu', or None for auto)
        token: Hugging Face token for private models

    Returns:
        Tuple of (model, tokenizer, device)
    """
    logger.info(f"Loading model: {model_name}")

    # Auto-detect device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, token=token)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded with {model.config.num_labels} labels")

    return model, tokenizer, device


def predict_batch(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    max_length: int = 512,
) -> List[dict]:
    """
    Run inference on a batch of texts.

    Args:
        texts: List of input texts
        model: The classification model
        tokenizer: The tokenizer
        device: Device to run on
        max_length: Maximum sequence length

    Returns:
        List of prediction dictionaries
    """
    # Tokenize
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        confidences = probs.max(dim=-1).values

    # Get label mappings
    id2label = model.config.id2label

    results = []
    for pred, conf in zip(predictions.cpu().numpy(), confidences.cpu().numpy()):
        naics_code = id2label[pred]
        results.append({
            "predicted_naics": naics_code,
            "confidence": float(conf),
            "naics_description": NAICS_DESCRIPTIONS.get(naics_code, "Unknown"),
        })

    return results


def run_inference(
    input_file: str,
    output_file: str,
    model_name: str = "aquiro1994/naics-github-classifier",
    batch_size: int = 16,
    max_length: int = 512,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    token: Optional[str] = None,
):
    """
    Run batch inference on a parquet file.

    Args:
        input_file: Path to input parquet file
        output_file: Path to output parquet file
        model_name: Hugging Face model ID
        batch_size: Batch size for inference
        max_length: Maximum sequence length
        limit: Limit number of rows (for testing)
        device: Device to use
        token: Hugging Face token for private models
    """
    # Load data
    logger.info(f"Loading data from: {input_file}")
    df = pd.read_parquet(input_file)

    if limit:
        df = df.head(limit)
        logger.info(f"Limited to {limit} rows for testing")

    logger.info(f"Total rows: {len(df)}")

    # Detect column names
    name_col = "name" if "name" in df.columns else "name_repo"
    readme_col = "readme" if "readme" in df.columns else "readme_content"

    logger.info(f"Using columns: name={name_col}, readme={readme_col}")

    # Load model
    model, tokenizer, device = load_model(model_name, device, token)

    # Prepare input texts
    logger.info("Formatting input texts...")
    texts = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Formatting"):
        text = format_input_text(
            name=row.get(name_col, ""),
            description=row.get("description", ""),
            topics=row.get("topics", ""),
            readme=row.get(readme_col, ""),
        )
        texts.append(text)

    # Run inference in batches
    logger.info(f"Running inference with batch_size={batch_size}...")
    all_results = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Inference"):
        batch_texts = texts[i:i + batch_size]
        batch_results = predict_batch(
            batch_texts, model, tokenizer, device, max_length
        )
        all_results.extend(batch_results)

    # Add predictions to dataframe
    df["predicted_naics"] = [r["predicted_naics"] for r in all_results]
    df["confidence"] = [r["confidence"] for r in all_results]
    df["naics_description"] = [r["naics_description"] for r in all_results]

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_file.endswith(".csv"):
        df.to_csv(output_file, index=False)
    else:
        df.to_parquet(output_file, index=False)

    logger.info(f"Results saved to: {output_file}")

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total predictions: {len(df)}")
    logger.info(f"Average confidence: {df['confidence'].mean():.4f}")
    logger.info(f"\nPrediction distribution:")
    for naics, count in df["predicted_naics"].value_counts().head(10).items():
        desc = NAICS_DESCRIPTIONS.get(naics, "Unknown")
        pct = count / len(df) * 100
        logger.info(f"  {naics} ({desc}): {count} ({pct:.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run batch inference for NAICS classification"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input parquet file path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path (parquet or csv)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="aquiro1994/naics-github-classifier",
        help="Hugging Face model ID (default: aquiro1994/naics-github-classifier)"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="Hugging Face token for private models (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=16,
        help="Batch size for inference (default: 16)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of rows (for testing)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (default: auto-detect)"
    )

    args = parser.parse_args()

    # Get token from args or environment variable
    import os
    token = args.token or os.environ.get("HF_TOKEN")

    run_inference(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        limit=args.limit,
        device=args.device,
        token=token,
    )


if __name__ == "__main__":
    main()
