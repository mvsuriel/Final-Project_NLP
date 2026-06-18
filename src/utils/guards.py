"""Reproducibility guards: the split guard refuses to score against a drifted test
set, the train-32 guard checks Part 2's frozen labeled sample. Each recomputes a
fingerprint from the parquet on disk and stops on mismatch. The splits and the
32-example draw are frozen; never regenerate them."""

import json
from pathlib import Path

import pandas as pd

from ..data.data import PROCESSED, fingerprint_ids, fingerprint_test_ids, load_labels, load_splits
from ..data.sampling import ALLOCATION


def run_split_guard(data_dir, expected_revision=None):
    """Check the persisted splits against split_manifest.json before any scoring.

    Recomputes the test-set fingerprint from the parquet on disk and compares it to the
    manifest in `data_dir`. On failure, artifacts and metrics are out of sync: stop and
    investigate, do NOT regenerate the split (Parts 2-4 depend on it). If
    `expected_revision` is given, also assert the manifest revision matches it. Returns
    {'labels', 'classes', 'splits', 'test_df', 'y_test', 'manifest'}.
    """
    data_dir = Path(data_dir)
    # Parquets load via PROCESSED, only the manifest comes from data_dir. A divergent
    # data_dir would check PROCESSED's parquets against another directory's manifest.
    assert data_dir.resolve() == PROCESSED.resolve(), (
        f"run_split_guard data_dir must be src.data.PROCESSED ({PROCESSED}) - got {data_dir}")
    labels = load_labels()
    classes = labels["classes"]
    splits = load_splits()
    test_df = splits["test"]

    manifest = json.loads((data_dir / "split_manifest.json").read_text())
    if expected_revision is not None:
        assert manifest["revision"] == expected_revision, (
            f"split_manifest.json revision {manifest['revision'][:12]}... differs from the "
            f"pinned revision {str(expected_revision)[:12]}... - the splits on disk are not the "
            "pinned dataset. Investigate before training or scoring; do NOT regenerate the split."
        )
    test_ids_sha256 = fingerprint_test_ids(test_df.index)
    assert test_ids_sha256 == manifest["test_ids_sha256"], (
        "test.parquet does NOT match the fingerprint in split_manifest.json - the metrics on "
        "disk may come from a different split. Investigate before evaluating anything."
    )
    for name, part in splits.items():
        assert len(part) == manifest["sizes"][name], (
            f"{name} split size changed: {len(part):,} vs manifest {manifest['sizes'][name]:,}")
    assert test_df["status"].value_counts().to_dict() == manifest["test_per_class"], \
        "Per-class test counts no longer match the manifest"

    y_test = test_df["status"].to_numpy()
    print(f"Split guard OK - test fingerprint {test_ids_sha256[:12]}... matches the manifest.")
    print(f"Frozen splits: {manifest['sizes']['train']:,} train / {manifest['sizes']['val']:,} val / "
          f"{manifest['sizes']['test']:,} test  -  dataset revision {manifest['revision'][:12]}...")
    return {"labels": labels, "classes": classes, "splits": splits,
            "test_df": test_df, "y_test": y_test, "manifest": manifest}


def run_train32_guard(data_dir):
    """Check that Part 2's fixed 32-example sample exists and matches its manifest.

    Recomputes the sampled ids' SHA-256 the way src/data/sampling.py wrote it and stops
    on mismatch. The draw happens once and is frozen, never redrawn. Returns the
    train-32 DataFrame.
    """
    data_dir = Path(data_dir)
    parquet_path = data_dir / "train_32.parquet"
    manifest_path = data_dir / "train32_manifest.json"
    missing = [p.name for p in (parquet_path, manifest_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{' and '.join(missing)} missing from {data_dir} - "
            "run python -m src.data.sampling ONCE from the project root (needs explicit "
            "approval - see docs/PART2_PLAN.md section 2, gap #1), then never again."
        )

    train32 = pd.read_parquet(parquet_path)
    manifest = json.loads(manifest_path.read_text())

    # Same recipe as the sampler: fingerprint_ids (sorted int ids, SHA-256).
    ids_sha256 = fingerprint_ids(train32.index)
    assert ids_sha256 == manifest["ids_sha256"], (
        "train_32.parquet does NOT match the ids_sha256 in train32_manifest.json - the 32 "
        "examples on disk are not the frozen draw. Investigate; never re-run src.data.sampling."
    )
    assert train32["status"].value_counts().to_dict() == ALLOCATION, \
        "train_32.parquet per-class counts no longer match sampling.ALLOCATION"

    print(f"Train-32 guard OK - ids fingerprint {ids_sha256[:12]}... matches the manifest "
          f"({len(train32)} examples).")
    return train32
