"""Data loading. Every notebook reads the frozen artifacts in data/processed/ so
all experiments use identical splits."""

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("NLP_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"


def load_labels():
    """Return {'classes': [...], 'label2id': {...}, 'id2label': {int: str}}."""
    labels = json.loads((PROCESSED / "labels.json").read_text())
    labels["id2label"] = {int(k): v for k, v in labels["id2label"].items()}
    return labels


def load_splits(splits=("train", "val", "test")):
    """Return {split_name: DataFrame} from the persisted parquet files."""
    return {s: pd.read_parquet(PROCESSED / f"{s}.parquet") for s in splits}


def load_train32():
    """Return the fixed 32-example labeled sample for Part 2."""
    path = PROCESSED / "train_32.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python -m src.data.sampling` from the project root first."
        )
    return pd.read_parquet(path)


def fingerprint_ids(ids):
    """SHA-256 of the comma-joined sorted integer ids. Single source of the fingerprint
    recipe for every frozen-id check (split guard, train-32 guard and sampler, augmented
    stamp, LLM subsample hash). Ids may arrive in any order; we cast to int and sort.
    """
    return hashlib.sha256(",".join(map(str, sorted(int(i) for i in ids))).encode()).hexdigest()


# Alias: the split fingerprint is an id fingerprint over the test ids.
fingerprint_test_ids = fingerprint_ids


def build_splits(df_clean, *, rows_raw, seed, dataset_name, dataset_revision, data_dir):
    """Build the train/val/test split and persist the parquet files, labels.json, and
    split_manifest.json under data_dir. Returns (train_df, val_df, test_df, classes).

    Run once by notebook 01; the split is frozen and Parts 2-4 depend on it. If a manifest
    exists, the recomputed test fingerprint must match or we refuse to overwrite.
    """
    from sklearn.model_selection import train_test_split

    data_dir = Path(data_dir)
    idx = df_clean.index.to_numpy()
    y_all = df_clean['status'].to_numpy()

    idx_temp, idx_test, y_temp, y_test = train_test_split(
        idx, y_all, test_size=0.2, random_state=seed, stratify=y_all
    )
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_temp, y_temp, test_size=0.2, random_state=seed, stratify=y_temp
    )

    train_df, val_df, test_df = (df_clean.loc[i] for i in (idx_train, idx_val, idx_test))

    for name, part in (('Train', train_df), ('Validation', val_df), ('Test', test_df)):
        print(f"{name + ':':<12}{len(part):,} samples ({len(part)/len(df_clean)*100:.1f}%)")

    test_ids_sha256 = fingerprint_test_ids(idx_test)
    manifest_path = data_dir / "split_manifest.json"
    if manifest_path.exists():
        persisted = json.loads(manifest_path.read_text())["test_ids_sha256"]
        assert persisted == test_ids_sha256, (
            "Recomputed test split does NOT match the persisted split_manifest.json - "
            "refusing to overwrite data/processed/. Investigate before deleting the manifest."
        )
        print("Split guard: recomputed test ids match the persisted manifest. OK")

    cols = ['text', 'status']
    for name, part in (('train', train_df), ('val', val_df), ('test', test_df)):
        part[cols].to_parquet(data_dir / f"{name}.parquet")

    classes = sorted(df_clean['status'].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    (data_dir / "labels.json").write_text(json.dumps(
        {"classes": classes, "label2id": label2id,
         "id2label": {str(i): c for c, i in label2id.items()}}, indent=2))

    manifest = {
        "dataset": dataset_name,
        "revision": dataset_revision,
        "seed": seed,
        "cleaning": {"rows_raw": int(rows_raw), "rows_clean": int(len(df_clean))},
        "sizes": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
        "test_per_class": {k: int(v) for k, v in test_df['status'].value_counts().items()},
        "test_ids_sha256": test_ids_sha256,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nSaved train/val/test parquet + labels.json + split_manifest.json to {data_dir}/")
    return train_df, val_df, test_df, classes
