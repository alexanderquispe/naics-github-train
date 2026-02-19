#!/usr/bin/env python
"""
Monthly Industry Adoption Charts for AI Agents.

This script creates visualizations showing when repos first adopted each AI agent,
broken down by industry (NAICS code).

Usage:
    python scripts/plot_industry_adoption.py
    python scripts/plot_industry_adoption.py --raw-data-dir /path/to/pr/data

Environment Variables:
    RAW_DATA_DIR: Path to directory containing PR JSONL files (copilot_prs.jsonl, codex_prs.jsonl)
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw data directory - configurable via environment variable or command line
DEFAULT_RAW_DATA_DIR = os.environ.get(
    "RAW_DATA_DIR",
    str(PROJECT_ROOT / "data" / "raw")  # Default to data/raw within project
)

# NAICS code descriptions for chart legends
NAICS_DESCRIPTIONS = {
    "11": "Agriculture",
    "21": "Mining",
    "22": "Utilities",
    "23": "Construction",
    "31-33": "Manufacturing",
    "42": "Wholesale Trade",
    "44-45": "Retail Trade",
    "48-49": "Transportation",
    "51": "Information",
    "52": "Finance",
    "53": "Real Estate",
    "54": "Professional Services",
    "56": "Admin Services",
    "61": "Education",
    "62": "Healthcare",
    "71": "Entertainment",
    "72": "Accommodation",
    "81": "Other Services",
    "92": "Public Admin",
}


def extract_first_use_from_prs(jsonl_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Extract first use date per repo from PR data.

    Args:
        jsonl_path: Path to the JSONL file with PR data
        output_path: Path to save the extracted first use dates

    Returns:
        DataFrame with columns ['nwo', 'first_use_date']
    """
    logger.info(f"Extracting first use dates from {jsonl_path}")

    # Read JSONL and get min date per repo
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            records.append({
                "repo_nwo": data["repo_nwo"],
                "created_at": data["created_at"]
            })

    df = pd.DataFrame(records)
    df["created_at"] = pd.to_datetime(df["created_at"])

    # Group by repo and get minimum date
    first_use = df.groupby("repo_nwo")["created_at"].min().reset_index()
    first_use.columns = ["nwo", "first_use_date"]

    logger.info(f"Found {len(first_use)} unique repos")

    # Save to parquet
    first_use.to_parquet(output_path, index=False)
    logger.info(f"Saved first use dates to {output_path}")

    return first_use


def load_claude_first_use() -> pd.DataFrame:
    """
    Load Claude first use dates from pre-computed parquet file.

    Returns:
        DataFrame with columns ['nwo', 'first_use_date']
    """
    path = PROCESSED_DIR / "first_claude_commits.parquet"
    logger.info(f"Loading Claude first use dates from {path}")

    df = pd.read_parquet(path)
    df = df.rename(columns={"repo_nwo": "nwo", "first_claude_commit": "first_use_date"})
    df["first_use_date"] = pd.to_datetime(df["first_use_date"], utc=True)

    logger.info(f"Loaded {len(df)} Claude repos")
    return df


def merge_with_predictions(first_use: pd.DataFrame, predictions_path: Path) -> pd.DataFrame:
    """
    Merge first use dates with predictions to get industry classification.

    Args:
        first_use: DataFrame with columns ['nwo', 'first_use_date']
        predictions_path: Path to predictions parquet file

    Returns:
        Merged DataFrame with industry classifications
    """
    logger.info(f"Loading predictions from {predictions_path}")
    predictions = pd.read_parquet(predictions_path)

    # Keep only necessary columns
    predictions = predictions[["nwo", "predicted_naics"]].copy()

    # Merge
    merged = first_use.merge(predictions, on="nwo", how="inner")
    logger.info(f"Merged {len(merged)} repos with industry classifications")

    return merged


def aggregate_by_month_industry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data by month and industry.

    Args:
        df: DataFrame with columns ['nwo', 'first_use_date', 'predicted_naics']

    Returns:
        Pivot table with months as index and industries as columns
    """
    # Extract year-month
    df = df.copy()
    df["year_month"] = df["first_use_date"].dt.to_period("M")

    # Count repos per industry per month
    monthly = df.groupby(["year_month", "predicted_naics"]).size().unstack(fill_value=0)

    # Sort by date
    monthly = monthly.sort_index()

    return monthly


def plot_stacked_area(
    data: pd.DataFrame,
    title: str,
    output_path: Path,
    top_n: int = 10
):
    """
    Create a stacked area chart of monthly adoption by industry.

    Args:
        data: Pivot table with months as index and industries as columns
        title: Chart title
        output_path: Path to save the chart
        top_n: Number of top industries to show (rest grouped as "Other")
    """
    # Get top N industries by total count
    industry_totals = data.sum().sort_values(ascending=False)
    top_industries = industry_totals.head(top_n).index.tolist()

    # Group remaining industries as "Other"
    plot_data = data[top_industries].copy()
    other_cols = [c for c in data.columns if c not in top_industries]
    if other_cols:
        plot_data["Other"] = data[other_cols].sum(axis=1)

    # Convert period index to datetime for plotting
    plot_data.index = plot_data.index.to_timestamp()

    # Create labels with descriptions
    columns_with_desc = []
    for col in plot_data.columns:
        if col == "Other":
            columns_with_desc.append("Other")
        else:
            desc = NAICS_DESCRIPTIONS.get(col, col)
            columns_with_desc.append(f"{col}: {desc}")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot stacked area
    plot_data.columns = columns_with_desc
    plot_data.plot.area(ax=ax, stacked=True, alpha=0.8)

    # Formatting
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("New Repos", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, title="Industry (NAICS)")
    ax.grid(axis="y", alpha=0.3)

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved chart to {output_path}")
    plt.close()


def process_agent(
    agent_name: str,
    first_use_df: pd.DataFrame,
    predictions_path: Path,
    output_prefix: str
):
    """
    Process a single agent: merge with predictions and create chart.

    Args:
        agent_name: Name of the agent (for logging)
        first_use_df: DataFrame with first use dates
        predictions_path: Path to predictions file
        output_prefix: Prefix for output files
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing {agent_name}")
    logger.info(f"{'='*50}")

    # Merge with predictions
    merged = merge_with_predictions(first_use_df, predictions_path)

    # Aggregate by month and industry
    monthly = aggregate_by_month_industry(merged)

    # Print summary
    logger.info(f"\nDate range: {monthly.index.min()} to {monthly.index.max()}")
    logger.info(f"Total repos: {monthly.sum().sum()}")
    logger.info(f"\nTop industries:")
    for naics, count in monthly.sum().sort_values(ascending=False).head(5).items():
        desc = NAICS_DESCRIPTIONS.get(naics, "Unknown")
        logger.info(f"  {naics} ({desc}): {count}")

    # Create visualization
    output_path = PROCESSED_DIR / f"industry_adoption_{output_prefix}.png"
    plot_stacked_area(
        monthly,
        f"Monthly Industry Adoption - {agent_name}",
        output_path
    )

    return merged


def main():
    """Main function to generate all adoption charts."""
    parser = argparse.ArgumentParser(
        description="Generate monthly industry adoption charts for AI agents"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default=DEFAULT_RAW_DATA_DIR,
        help="Directory containing PR JSONL files (copilot_prs.jsonl, codex_prs.jsonl)"
    )
    args = parser.parse_args()

    raw_data_dir = Path(args.raw_data_dir)

    logger.info("Starting industry adoption analysis")
    logger.info(f"Raw data directory: {raw_data_dir}")

    # Ensure output directory exists
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Process Claude Code
    claude_first_use = load_claude_first_use()
    process_agent(
        "Claude Code",
        claude_first_use,
        PROCESSED_DIR / "predictions_full.parquet",
        "claude"
    )

    # Process Copilot
    copilot_prs_path = raw_data_dir / "copilot_prs.jsonl"
    copilot_first_use_path = PROCESSED_DIR / "copilot_first_use.parquet"

    if copilot_prs_path.exists():
        copilot_first_use = extract_first_use_from_prs(copilot_prs_path, copilot_first_use_path)
        process_agent(
            "GitHub Copilot",
            copilot_first_use,
            PROCESSED_DIR / "copilot_predictions_full.parquet",
            "copilot"
        )
    else:
        logger.warning(f"Copilot PR data not found at {copilot_prs_path}")

    # Process Codex
    codex_prs_path = raw_data_dir / "codex_prs.jsonl"
    codex_first_use_path = PROCESSED_DIR / "codex_first_use.parquet"

    if codex_prs_path.exists():
        codex_first_use = extract_first_use_from_prs(codex_prs_path, codex_first_use_path)
        process_agent(
            "OpenAI Codex",
            codex_first_use,
            PROCESSED_DIR / "codex_predictions_full.parquet",
            "codex"
        )
    else:
        logger.warning(f"Codex PR data not found at {codex_prs_path}")

    logger.info("\nDone! Check data/processed/ for output files.")


if __name__ == "__main__":
    main()
