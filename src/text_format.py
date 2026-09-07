"""
Single source of truth for the text the model sees.

A classifier is only as good as the match between the text it was fine-tuned on
and the text it is asked about. Every entry point in this package -- training
(`data_loader.prepare_text_input`), single and file prediction
(`inference.format_repository_input`, `scripts/predict.py`) and batch inference
(`scripts/inference_batch.py`) -- builds that text through `format_model_input`
below, so the two sides cannot drift apart.

The format is:

    Repository: <name> | Description: <description> | Topics: <a; b> | README: <readme>

with `clean_readme_text` applied to the joined string, exactly as during
training. Fields that are missing are dropped along with their separator.

Historical note. Before version 1.1 the training path cleaned the text and
`format_repository_input` did not, so `scripts/predict.py` and the snippet in
the README fed the model raw markdown while the model had been fine-tuned on
cleaned text. Measured on 25,000 GitHub repositories the effect was small --
the two formats give the same sector for 99.4% of the repositories the model is
confident about -- but the mismatch was real and is now closed. Pass
`clean_text=False` to reproduce the older behaviour.
"""

import ast
import re
from typing import Optional

import numpy as np
import pandas as pd


def decode_description(value) -> str:
    """Turn a Python bytes repr back into text.

    Some GitHub exports serialise the description column as the repr of a bytes
    object -- ``b'Mestrado de engenharia inform\\xc3\\xa1tica'`` -- which reaches
    the tokenizer as the literal characters ``b``, ``'``, ``\\``, ``x``, ``c``,
    ``3`` instead of the accented letters they encode. This restores the text.
    Idempotent and forgiving: anything that is not such a repr is returned
    unchanged.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    if not (text.startswith("b'") or text.startswith('b"')):
        return text
    try:
        decoded = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text
    if isinstance(decoded, bytes):
        return decoded.decode("utf-8", errors="replace")
    return str(decoded)


def clean_topics(topic_str) -> str:
    """
    Clean and normalize the topics field.

    Handles various formats: lists, string representations of lists,
    comma-separated strings, etc.

    Args:
        topic_str: Topics field value (various formats)

    Returns:
        Cleaned topics string with semicolon separation
    """
    # Handle None/NaN values
    if topic_str is None or (isinstance(topic_str, float) and pd.isna(topic_str)):
        return ""

    # Handle arrays/lists
    if isinstance(topic_str, (list, tuple, np.ndarray)):
        if len(topic_str) == 0:
            return ""
        return "; ".join([str(item) for item in topic_str if item])

    # Handle empty strings
    if isinstance(topic_str, str) and (topic_str == "" or topic_str == "[]"):
        return ""

    try:
        # If it's a string representation of a list
        if isinstance(topic_str, str):
            if topic_str.startswith("[") and topic_str.endswith("]"):
                try:
                    topic_list = ast.literal_eval(topic_str)
                    if isinstance(topic_list, list):
                        return "; ".join([str(item) for item in topic_list if item])
                except (ValueError, SyntaxError):
                    pass

            # Clean string format
            return topic_str.replace("[", "").replace("]", "").replace(",", ";").strip()

        # Convert other types to string
        return str(topic_str)

    except Exception:
        return ""


def clean_readme_text(text: str) -> str:
    """
    Clean README text by removing markdown artifacts, code blocks, and noise.

    This is the cleaning applied during training, and -- since version 1.1 --
    at inference as well.

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


def _present(value) -> bool:
    """A field is present when it is not null and not the string 'nan'."""
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and text != "nan"


def format_model_input(
    repo_name=None,
    description=None,
    topics=None,
    readme=None,
    clean_text: bool = True,
    max_readme_chars: Optional[int] = None,
    decode_bytes_description: bool = True,
) -> str:
    """
    Build the text the model sees, identically for training and inference.

    Args:
        repo_name: Repository name
        description: Repository description
        topics: Topics, as a list, a repr of a list, or a separated string
        readme: README content
        clean_text: Apply `clean_readme_text` to the joined string. True is what
            training does; pass False only to reproduce the pre-1.1 behaviour of
            `format_repository_input`, or the published production datasets.
        max_readme_chars: Truncate the README to this many characters before
            joining. None (the default) matches training. Note that the
            tokenizer truncates to `max_length` tokens regardless, which for a
            512-token model binds long before any reasonable character cap.
        decode_bytes_description: Restore descriptions serialised as a Python
            bytes repr. A no-op on descriptions that are already text.

    Returns:
        Formatted text string for the model
    """
    components = []

    if _present(repo_name):
        components.append(f"Repository: {repo_name}")

    if _present(description):
        text = decode_description(description) if decode_bytes_description else str(description)
        if _present(text):
            components.append(f"Description: {text}")

    topics_str = clean_topics(topics)
    if _present(topics_str):
        components.append(f"Topics: {topics_str}")

    if _present(readme):
        readme_text = str(readme)
        if max_readme_chars is not None:
            readme_text = readme_text[:max_readme_chars]
        components.append(f"README: {readme_text}")

    combined = " | ".join(components)

    if clean_text:
        combined = clean_readme_text(combined)

    return combined
