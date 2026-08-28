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


def resolve_device(device: Optional[str] = None) -> str:
    """
    Pick the best available device.

    Order: explicit choice > CUDA > MPS (Apple Silicon) > CPU.
    """
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(device: str, fp16: Optional[bool] = None) -> torch.dtype:
    """
    Pick the inference precision.

    Half precision is enabled by default on CUDA and MPS: it is roughly 3x
    faster than fp32 and does not change predictions above the confidence
    thresholds this model is used with. CPU stays on fp32, where fp16 is slower.
    """
    if fp16 is None:
        fp16 = device in ("cuda", "mps")
    return torch.float16 if fp16 else torch.float32


def load_model(
    model_name: str,
    device: Optional[str] = None,
    token: Optional[str] = None,
    fp16: Optional[bool] = None,
):
    """
    Load the model and tokenizer from Hugging Face.

    Args:
        model_name: Hugging Face model ID or local path
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
        token: Hugging Face token for private models
        fp16: Force half precision on/off (None = auto per device)

    Returns:
        Tuple of (model, tokenizer, device)
    """
    logger.info(f"Loading model: {model_name}")

    device = resolve_device(device)
    dtype = resolve_dtype(device, fp16)
    logger.info(f"Using device: {device} ({str(dtype).replace('torch.', '')})")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, token=token, dtype=dtype
    )
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
        # float() first: softmax in fp16 loses precision on near-uniform logits
        probs = torch.softmax(outputs.logits.float(), dim=-1)
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
    batch_size: int = 32,
    max_length: int = 512,
    limit: Optional[int] = None,
    device: Optional[str] = None,
    token: Optional[str] = None,
    fp16: Optional[bool] = None,
    clean_text: bool = True,
    max_readme_chars: int = 5000,
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
    model, tokenizer, device = load_model(model_name, device, token, fp16)

    # Prepare input texts. Zipping the columns is an order of magnitude faster
    # than iterrows, which matters at hundreds of thousands of rows.
    logger.info("Formatting input texts...")

    def column_or_blank(name: str):
        return df[name].tolist() if name in df.columns else [""] * len(df)

    texts = [
        format_input_text(
            name=name,
            description=description,
            topics=topics,
            readme=readme,
            max_readme_chars=max_readme_chars,
            clean_text=clean_text,
        )
        for name, description, topics, readme in zip(
            column_or_blank(name_col),
            column_or_blank("description"),
            column_or_blank("topics"),
            column_or_blank(readme_col),
        )
    ]

    # Length-sorted batching: group texts of similar length so each batch pads
    # to a short common length instead of to the longest text in the file.
    # Same predictions, but several times less wasted computation.
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    results_in_order = [None] * len(texts)

    logger.info(f"Running inference with batch_size={batch_size}...")
    for i in tqdm(range(0, len(order), batch_size), desc="Inference"):
        idx = order[i:i + batch_size]
        batch_results = predict_batch(
            [texts[j] for j in idx], model, tokenizer, device, max_length
        )
        for j, result in zip(idx, batch_results):
            results_in_order[j] = result

    # Add predictions to dataframe
    df["predicted_naics"] = [r["predicted_naics"] for r in results_in_order]
    df["confidence"] = [r["confidence"] for r in results_in_order]
    df["naics_description"] = [r["naics_description"] for r in results_in_order]

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
        default=32,
        help="Batch size for inference (default: 32). On Apple Silicon 32-64 is "
             "fastest; larger batches are slower, not faster."
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
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: auto-detect; 'mps' is Apple Silicon)"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        default=None,
        help="Force half precision (default: on for CUDA and MPS, off for CPU)"
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Force full precision"
    )
    parser.add_argument(
        "--no-clean-text",
        dest="clean_text",
        action="store_false",
        default=True,
        help="Skip clean_readme_text. The production pipeline that generated the "
             "published NAICS datasets does NOT clean the text; use this flag "
             "together with --max-readme-chars 3000 to reproduce it."
    )
    parser.add_argument(
        "--max-readme-chars",
        type=int,
        default=5000,
        help="Truncate README to this many characters (default: 5000; the "
             "production pipeline uses 3000)"
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
        fp16=args.fp16,
        clean_text=args.clean_text,
        max_readme_chars=args.max_readme_chars,
    )


if __name__ == "__main__":
    main()
