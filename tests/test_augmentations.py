"""Structural validation of the frozen augmented sets, the manual-review step from notebook
02 written as assertions: every suicidal seed must keep augmented copies, no augmented text
may be empty, and every label must stay within the seven classes. Skips when the augmented
sets have not been generated locally."""

import pytest

from src.data.data import load_labels, load_train32
from src.data.augment import AUGMENTED, load_augmented

pytestmark = pytest.mark.skipif(
    not (AUGMENTED / "augmented_manifest.json").exists(),
    reason="no frozen augmented sets present",
)

METHODS = ("backtranslation", "eda")


def test_every_suicidal_seed_has_augmented_copies():
    train32 = load_train32()
    sui_ids = [int(i) for i in train32.index if train32.loc[i, "status"] == "suicidal"]
    for method in METHODS:
        aug = load_augmented(method)
        for sid in sui_ids:
            assert (aug["source_id"] == sid).any(), f"{method}: no copies for suicidal seed {sid}"


def test_augmented_texts_nonempty_and_labels_valid():
    classes = set(load_labels()["classes"])
    for method in METHODS:
        aug = load_augmented(method)
        assert aug["text"].str.strip().str.len().gt(0).all(), f"{method}: empty augmented text"
        assert set(aug["status"]).issubset(classes), f"{method}: label outside the class set"
