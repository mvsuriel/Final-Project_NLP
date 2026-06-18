"""Shared evaluation helpers so every experiment reports identical metrics."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)

from ..data.data import PROCESSED, RESULTS


def logits_to_preds(logits):
    """Argmax label ids from logits, unwrapping the tuple some HF models return."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return np.argmax(logits, axis=-1)


def evaluate_predictions(y_true, y_pred, labels):
    """Full metric dict for string-label predictions on string-label ground truth."""
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "per_class": {
            label: {
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for label, p, r, f, s in zip(labels, prec, rec, f1, support)
        },
    }


def make_trainer_compute_metrics(id2label):
    """compute_metrics fn for the HF Trainer; returns accuracy and macro F1."""

    def compute_metrics(eval_pred):
        preds = logits_to_preds(eval_pred.predictions)
        label_ids = eval_pred.label_ids
        return {
            "accuracy": float(accuracy_score(label_ids, preds)),
            "macro_f1": float(f1_score(label_ids, preds, average="macro", zero_division=0)),
        }

    return compute_metrics


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=42):
    """95% bootstrap CIs for accuracy and macro F1 (resampling with replacement)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    accs, f1s = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        accs[b] = accuracy_score(y_true[i], y_pred[i])
        f1s[b] = f1_score(y_true[i], y_pred[i], average="macro", zero_division=0)
    pct = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"accuracy_ci95": pct(accs), "macro_f1_ci95": pct(f1s)}


def mcnemar_test(y_true, pred_a, pred_b):
    """Exact-binomial McNemar test for two classifiers on the same test set; returns the two disagreement counts and the p-value."""
    y_true, pred_a, pred_b = (np.asarray(v) for v in (y_true, pred_a, pred_b))
    a_correct, b_correct = pred_a == y_true, pred_b == y_true
    a_only = int((a_correct & ~b_correct).sum())
    b_only = int((~a_correct & b_correct).sum())
    n = a_only + b_only
    p_value = binomtest(min(a_only, b_only), n, 0.5).pvalue if n else 1.0
    return {"a_only_correct": a_only, "b_only_correct": b_only, "p_value": float(p_value)}


def save_predictions(y_true, y_pred, name, index=None):
    """Write per-example predictions to results/predictions/{name}.csv and return the path; `index` must be test_df.index so files from different experiments align row by row for mcnemar_test()."""
    if index is None:
        raise ValueError(
            "save_predictions requires the frozen test index (pass index=test_df.index) - "
            "a predictions file without it cannot be aligned for paired tests."
        )
    out_dir = RESULTS / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)}, index=index
    )
    frame.index.name = "test_id"
    path = out_dir / f"{name}.csv"
    frame.to_csv(path)
    return path


def load_predictions(name):
    """Read back a predictions file written by save_predictions()."""
    return pd.read_csv(RESULTS / "predictions" / f"{name}.csv", index_col="test_id")


def _provenance():
    """Stamp recording when the result was computed and on which split."""
    prov = {"computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    manifest_path = PROCESSED / "split_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        prov.update(
            dataset_revision=manifest["revision"],
            split_seed=manifest["seed"],
            test_ids_sha256=manifest["test_ids_sha256"],
        )
    return prov


def save_metrics(result, name):
    """Write a metrics dict to results/metrics/{name}.json and return the path."""
    out_dir = RESULTS / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps({"name": name, **result, "provenance": _provenance()}, indent=2))
    return path
