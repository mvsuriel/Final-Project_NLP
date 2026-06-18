"""Pytest mirror of the in-notebook reproducibility guards (CI flavour). These re-run the
same fingerprint logic shipped in src/utils/guards.py and the prompt-fidelity check from
notebook 02, so a drifted split, a re-drawn train-32, or an edited frozen prompt fails here
too. The guards also run inside the notebooks; this suite is the standalone copy."""

from pathlib import Path

import pytest
import yaml

from src.data.data import PROCESSED
from src.utils.guards import run_split_guard, run_train32_guard
from src.models.llm import BASE_TASK, DECISION_RULES, make_candidate

_HAS_SPLIT = (PROCESSED / "split_manifest.json").exists()
_HAS_TRAIN32 = (PROCESSED / "train32_manifest.json").exists()

# The frozen 2c prompt hashes (must match notebook 02's fidelity self-check).
_EXPECTED_PROMPT_HASHES = {
    "v1_direct": "de97c1f4eb7a",
    "v1_cot": "c8adf0d41731",
    "v2_direct": "3a21817c83c1",
    "v2_cot": "cc45295b33bd",
}


@pytest.mark.skipif(not _HAS_SPLIT, reason="frozen split not present")
def test_split_guard_matches_manifest():
    ctx = run_split_guard(PROCESSED)  # raises on any drift
    assert ctx["manifest"]["test_ids_sha256"]
    assert len(ctx["classes"]) == 7


@pytest.mark.skipif(not _HAS_TRAIN32, reason="frozen train-32 not present")
def test_train32_guard_matches_manifest():
    train32 = run_train32_guard(PROCESSED)  # raises on any drift
    assert len(train32) == 32


def test_prompt_fidelity():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "config.yaml").read_text())
    fmax = {
        "direct": cfg["zero_shot_2c"]["max_new_tokens_direct"],
        "cot": cfg["zero_shot_2c"]["max_new_tokens_cot"],
    }
    candidates = [
        make_candidate("v1_direct", BASE_TASK, "direct", fmax),
        make_candidate("v1_cot", BASE_TASK, "cot", fmax),
        make_candidate("v2_direct", BASE_TASK + DECISION_RULES, "direct", fmax),
        make_candidate("v2_cot", BASE_TASK + DECISION_RULES, "cot", fmax),
    ]
    for c in candidates:
        assert c["hash"] == _EXPECTED_PROMPT_HASHES[c["name"]], f"prompt drift in {c['name']}"
