"""Draw and persist the fixed 32-example labeled sample for Part 2.

Stratified over the 7 classes (see docs/PART2_PLAN.md): the 4 largest train
classes get 5 examples, the 3 smallest get 4, for 32 total. Drawn from the TRAIN
split only at seed 42, and saved with a manifest (ids + sha256). The 32 are
frozen; Parts 2-4 all use this same sample.
"""

import json

import numpy as np
import pandas as pd

from .data import PROCESSED, fingerprint_ids

SEED = 42

# 4 largest train classes get 5, the 3 smallest get 4 (= 32)
ALLOCATION = {
    "anxiety": 5,
    "normal": 5,
    "depression": 5,
    "stress": 5,
    "personality disorder": 4,
    "bipolar": 4,
    "suicidal": 4,
}


def make_train32(seed=SEED):
    manifest_path = PROCESSED / "train32_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"{manifest_path} exists - the fixed 32-example sample is FROZEN (plan 2, gap #1). "
            "Never regenerate; load it via the train-32 guard (src.utils.guards.run_train32_guard). "
            "To intentionally re-draw, delete train_32.parquet + train32_manifest.json first."
        )
    train = pd.read_parquet(PROCESSED / "train.parquet")
    rng = np.random.default_rng(seed)

    parts = []
    for label, n in ALLOCATION.items():
        cls = train[train["status"] == label]
        pos = rng.choice(len(cls), size=n, replace=False)
        parts.append(cls.iloc[sorted(pos)])

    sample = pd.concat(parts).sort_index()
    assert len(sample) == 32, f"expected 32 rows, got {len(sample)}"
    assert sample.index.is_unique, "sampled ids are not unique"

    sample.to_parquet(PROCESSED / "train_32.parquet")

    ids = sorted(int(i) for i in sample.index)
    manifest = {
        "seed": seed,
        "allocation": ALLOCATION,
        "realized_counts": {k: int(v) for k, v in sample["status"].value_counts().items()},
        "selected_ids": ids,
        "ids_sha256": fingerprint_ids(ids),
        "source": {
            "file": "train.parquet",
            "split_seed": 42,
            "dataset_revision": "186902250e947738f5cb6668808431c56f75b018",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return sample, manifest


if __name__ == "__main__":
    sample, manifest = make_train32()
    print(json.dumps(manifest, indent=2))
