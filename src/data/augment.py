"""Build and load the frozen Part 2b augmented sets (docs/PART2_PLAN.md 4b).

Two seeded generators over the fixed 32-example sample (never val/test):
back-translation round-trips en -> pivot -> en through MarianMT (Helsinki-NLP
opus-mt), one paraphrase per pivot (de/fr/es/ru); long posts are sentence-split,
translated piece by piece and rejoined so nothing is truncated at Marian's
~512-token limit. EDA uses the vendored eda_vendored at alpha 0.05, n_aug 4
(size-matched to the 4 pivots), with PROTECTED_WORDS active.

make_augmented_sets() runs once on Colab and persists
data/augmented/{backtrans,eda}.parquet + augmented_manifest.json. It refuses to
run again once the manifest exists. The augmented sets are frozen; never
regenerate. load_augmented() re-fingerprints the parquet against the manifest on
every read.
"""

import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import PROCESSED, fingerprint_ids

AUGMENTED = PROCESSED.parent / "augmented"  # never data/processed/

SEED = 42
PIVOTS = ("de", "fr", "es", "ru")
EDA_ALPHA = 0.05  # all four EDA operation rates
EDA_NAUG = 4      # size-matched to the 4 pivots
BT_BATCH = 16

# Sentence splitter on punctuation boundaries; avoids an nltk punkt download.
# The word-chunk cap handles run-on posts with no punctuation.
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_MAX_CHUNK_WORDS = 200

_PARQUET_BY_METHOD = {"backtranslation": "backtrans.parquet", "eda": "eda.parquet"}


def _sentences(text):
    """Split one post into translatable chunks, keeping all content."""
    parts = [p.strip() for p in _SENT_RE.split(text.strip()) if p.strip()]
    out = []
    for part in parts or [text.strip()]:
        words = part.split()
        for i in range(0, len(words), _MAX_CHUNK_WORDS):
            out.append(" ".join(words[i : i + _MAX_CHUNK_WORDS]))
    return out


def _translate(texts, tokenizer, model, batch_size, device):
    """Batched translation using the model's default generation (deterministic)."""
    import torch

    out = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i : i + batch_size], return_tensors="pt",
                        padding=True, truncation=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc)
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out


def _round_trip(texts, pivot, batch_size=BT_BATCH):
    """en -> pivot -> en paraphrases for a list of texts.

    transformers v5 dropped the 'translation' pipeline task, so we drive the
    Marian models directly: tokenizer -> generate -> batch_decode.
    """
    import torch  # only the generation path needs the GPU stack
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    legs = []
    for name in (f"Helsinki-NLP/opus-mt-en-{pivot}", f"Helsinki-NLP/opus-mt-{pivot}-en"):
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device).eval()
        legs.append((tokenizer, model))

    per_text = [_sentences(t) for t in texts]
    flat = [s for sents in per_text for s in sents]
    pivoted = _translate(flat, *legs[0], batch_size, device)
    restored = _translate(pivoted, *legs[1], batch_size, device)

    out, pos = [], 0
    for sents in per_text:
        out.append(" ".join(restored[pos : pos + len(sents)]).strip())
        pos += len(sents)
    del legs
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _normalize(text):
    return " ".join(text.lower().split())


def _dedup(df, train32):
    """Drop paraphrases identical to their source or to an earlier paraphrase of
    the same source; counts reported are post-dedup."""
    source_norm = train32["text"].map(_normalize)
    keep, seen = [], set()
    for row in df.itertuples():
        norm = _normalize(row.text)
        key = (row.source_id, norm)
        identical_to_source = norm == source_norm.loc[row.source_id]
        keep.append(not identical_to_source and key not in seen)
        seen.add(key)
    return df[np.array(keep)].reset_index(drop=True)


def _fingerprint(df):
    """Content hash over sorted (source_id, method, variant, text) rows."""
    rows = sorted(df[["source_id", "method", "variant", "text"]].itertuples(index=False))
    blob = "\x1f".join("\x1e".join(map(str, r)) for r in rows)
    return hashlib.sha256(blob.encode()).hexdigest()


def _counts(generated, df, train32):
    per_class = df["status"].value_counts().to_dict()
    return {
        "generated": int(generated),
        "dropped_identical": int(generated - len(df)),
        "effective": int(len(df)),
        "effective_per_class": {k: int(v) for k, v in per_class.items()},
        "n_train_with_originals": int(len(df) + len(train32)),
    }


def make_augmented_sets(train32, out_dir=AUGMENTED):
    """Generate both augmented sets + manifest once. Frozen afterwards."""
    out_dir = Path(out_dir)
    manifest_path = out_dir / "augmented_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"{manifest_path} exists - data/augmented/ is FROZEN (plan 4b). "
            "Never regenerate; load with src.augment.load_augmented()."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)

    # back-translation: one paraphrase per pivot per seed example
    bt_rows = []
    texts = train32["text"].tolist()
    for pivot in PIVOTS:
        for (sid, row), para in zip(train32.iterrows(), _round_trip(texts, pivot)):
            bt_rows.append(
                {"aug_id": f"bt_{sid}_{pivot}", "text": para, "status": row["status"],
                 "source_id": int(sid), "method": "backtranslation", "variant": pivot}
            )
    bt = _dedup(pd.DataFrame(bt_rows), train32)

    # EDA: importing eda_vendored seeds global random to 1, so re-seed after
    from . import eda_vendored

    random.seed(SEED)
    eda_rows = []
    for sid, row in train32.iterrows():
        # eda() appends the cleaned original last; drop it, originals enter
        # training from train32 itself.
        paras = eda_vendored.eda(
            row["text"], alpha_sr=EDA_ALPHA, alpha_ri=EDA_ALPHA,
            alpha_rs=EDA_ALPHA, p_rd=EDA_ALPHA, num_aug=EDA_NAUG,
        )[:-1]
        for k, para in enumerate(paras):
            eda_rows.append(
                {"aug_id": f"eda_{sid}_v{k}", "text": para, "status": row["status"],
                 "source_id": int(sid), "method": "eda", "variant": f"v{k}"}
            )
    eda_df = _dedup(pd.DataFrame(eda_rows), train32)

    bt.to_parquet(out_dir / _PARQUET_BY_METHOD["backtranslation"])
    eda_df.to_parquet(out_dir / _PARQUET_BY_METHOD["eda"])

    manifest = {
        "seed": SEED,
        "pivots": list(PIVOTS),
        "eda_params": {"alpha": EDA_ALPHA, "num_aug": EDA_NAUG,
                       "protected_words": sorted(eda_vendored.PROTECTED_WORDS)},
        "counts": {
            "backtranslation": _counts(len(bt_rows), bt, train32),
            "eda": _counts(len(eda_rows), eda_df, train32),
        },
        "fingerprints": {"backtranslation": _fingerprint(bt), "eda": _fingerprint(eda_df)},
        "source": {
            "file": "train_32.parquet",
            "ids_sha256": fingerprint_ids(train32.index),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return bt, eda_df, manifest


def load_augmented(method, out_dir=AUGMENTED):
    """Read one frozen augmented set; fail on any content drift."""
    out_dir = Path(out_dir)
    parquet_path = out_dir / _PARQUET_BY_METHOD[method]
    manifest_path = out_dir / "augmented_manifest.json"
    missing = [p.name for p in (parquet_path, manifest_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{' and '.join(missing)} missing from {out_dir} - run the one-time "
            "generation cell in notebooks/02_limited_data.ipynb ON COLAB first (plan 4b)."
        )
    df = pd.read_parquet(parquet_path)
    manifest = json.loads(manifest_path.read_text())
    assert _fingerprint(df) == manifest["fingerprints"][method], (
        f"{parquet_path.name} does NOT match the fingerprint in augmented_manifest.json - "
        "the augmented set on disk is not the frozen one. Investigate; never regenerate."
    )
    return df
