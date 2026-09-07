"""The input the model sees must be built identically everywhere.

A classifier fine-tuned on cleaned text and asked about raw markdown is being
used out of distribution. Before version 1.1 that is what happened: training
cleaned the text, `format_repository_input` did not. These tests lock the
invariant so it cannot come back.

    python -m pytest tests/test_text_format.py -q
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data_loader import prepare_text_input  # noqa: E402
from src.inference import format_repository_input  # noqa: E402
from src.text_format import (  # noqa: E402
    clean_readme_text,
    clean_topics,
    decode_description,
    format_model_input,
)


def _batch_formatter():
    """scripts/inference_batch.py is a script, not a module: load it by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("ib", ROOT / "scripts" / "inference_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.format_input_text


CASES = [
    dict(
        repo_name="mediscan",
        description="AI diagnostic tool for radiology",
        topics=["healthcare", "medical-imaging"],
        readme="# MediScan\n\nUses **computer vision**.\n\n```python\nimport x\n```\nSee https://example.com/docs.",
    ),
    dict(repo_name="acme", description="b'Mestrado de engenharia inform\\xc3\\xa1tica'", topics="[]", readme=None),
    dict(repo_name="solo", description=None, topics=None, readme=None),
    dict(repo_name="topics-as-repr", description="d", topics="['a', 'b']", readme="plain readme"),
    dict(repo_name="nan-fields", description=float("nan"), topics=float("nan"), readme="x"),
]


@pytest.mark.parametrize("case", CASES)
def test_every_entry_point_builds_the_same_text(case):
    """Training, single prediction and batch inference must agree, character for character."""
    row = pd.Series({
        "name_repo": case["repo_name"], "description": case["description"],
        "topics": case["topics"], "readme_content": case["readme"],
    })
    from_training = prepare_text_input(row)
    from_predict = format_repository_input(
        case["repo_name"], case["description"], case["topics"], case["readme"]
    )
    from_batch = _batch_formatter()(
        case["repo_name"], case["description"], case["topics"], case["readme"]
    )
    assert from_training == from_predict == from_batch


def test_cleaning_is_on_by_default_at_inference():
    """The regression that motivated version 1.1."""
    readme = "# Title\n\n**bold** text with a badge ![b](http://x/y.png)"
    cleaned = format_repository_input("r", readme=readme)
    assert "**" not in cleaned and "![" not in cleaned
    assert "\n" not in cleaned


def test_the_old_uncleaned_behaviour_is_still_reachable():
    raw = format_repository_input("r", readme="# Title\n**bold**", clean_text=False)
    assert raw == "Repository: r | README: # Title\n**bold**"


def test_bytes_repr_descriptions_are_decoded():
    assert decode_description("b'caf\\xc3\\xa9'") == "café"
    assert decode_description("already text") == "already text"
    assert decode_description(None) == ""
    # idempotent: decoding twice changes nothing
    once = decode_description("b'caf\\xc3\\xa9'")
    assert decode_description(once) == once


def test_topics_normalise_to_semicolons():
    for value in (["a", "b"], "['a', 'b']", "a,b"):
        assert clean_topics(value) in ("a; b", "a;b")
    assert clean_topics(None) == ""
    assert clean_topics("[]") == ""


def test_missing_fields_are_dropped_with_their_separator():
    assert format_model_input(repo_name="r") == "Repository: r"
    assert format_model_input(repo_name="r", description="d") == "Repository: r | Description: d"
    assert format_model_input(description="d") == "Description: d"


def test_readme_truncation_is_off_by_default_and_works_when_asked():
    long_readme = "x" * 9000
    assert len(format_model_input(repo_name="r", readme=long_readme)) > 9000
    short = format_model_input(repo_name="r", readme=long_readme, max_readme_chars=100)
    assert len(short) < 200


def test_clean_readme_text_is_unchanged():
    """The function is part of the trained model's contract: do not edit it."""
    assert clean_readme_text("# H\n\n**b**") == "H b"
    assert clean_readme_text("see https://github.com/a/b now") == "see github.com now"
    assert clean_readme_text(None) == ""


# ---------------------------------------------------------------- frozen behaviour
# clean_readme_text is part of the trained model's contract. These tests pin the
# two rules that look like bugs and are not: see the docstring of the function.

def test_installation_rule_truncates_to_end_of_string():
    """Everything after the first install command is dropped, by design.

    The published training file was built this way, so the model expects it.
    Changing this means retraining.
    """
    text = "A farm management tool. pip install farmtool Then run it. It predicts crop yields."
    assert clean_readme_text(text) == "A farm management tool."


def test_code_fences_are_stripped_but_their_bodies_survive():
    """The 'keep language info' rule never fires; code bodies reach the model."""
    cleaned = clean_readme_text("intro\n```python\nimport torch\n```\nend")
    assert "code-python" not in cleaned
    assert "import torch" in cleaned


def test_pd_na_is_treated_as_missing():
    """pd.NA must not reach the model as the literal string '<NA>'."""
    import pandas as pd

    assert clean_readme_text(pd.NA) == ""
    assert clean_topics(pd.NA) == ""
    assert decode_description(pd.NA) == ""
    assert format_model_input(repo_name="r", description=pd.NA, topics=pd.NA, readme=pd.NA) == "Repository: r"


def test_clean_topics_is_idempotent():
    """format_model_input cleans topics, and the training path may have already
    done so; applying it twice must not change the result."""
    for value in (["a", "b"], "['a', 'b']", "a,b", ["deep, learning", "nlp"], None, "[]"):
        once = clean_topics(value)
        assert clean_topics(once) == once, f"not idempotent for {value!r}"
