"""Shared evaluation contract: bind the frozen test set into evaluate_model() so every
experiment is scored through identical code. Reproducibility guards live in
src/utils/guards.py."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay

from .metrics import bootstrap_ci, evaluate_predictions, save_metrics, save_predictions


def make_evaluate_model(test_df, y_test, classes, fig_dir, n_boot, seed):
    """Bind the frozen test set into evaluate_model(). Call once after run_split_guard();
    every part scores through the returned closure."""
    fig_dir = Path(fig_dir)

    def evaluate_model(name, y_pred, extras=None, make_figure=True, title=None):
        """Score one model's test predictions and write the JSON, predictions, and figure.

        name    : 'part{N}_{slug}', used for the output filenames
        y_pred  : string labels from labels.json, in frozen test-set order
        extras  : optional dict merged into the JSON
        title   : confusion-matrix title; defaults to one derived from name
        Returns the metrics dict.
        """
        y_pred = np.asarray(y_pred)
        assert len(y_pred) == len(y_test), f"{len(y_pred):,} predictions for {len(y_test):,} test rows"
        unknown = sorted(set(y_pred) - set(classes))
        assert not unknown, f"Predictions contain labels outside labels.json: {unknown}"

        results = evaluate_predictions(y_test, y_pred, classes)
        results |= bootstrap_ci(y_test, y_pred, n_boot=n_boot, seed=seed)
        if extras:
            results |= extras
        save_metrics(results, name)
        save_predictions(y_test, y_pred, name, index=test_df.index)

        print(name)
        print(f"Accuracy: {results['accuracy']:.4f}  "
              f"(95% CI {results['accuracy_ci95'][0]:.4f}-{results['accuracy_ci95'][1]:.4f})")
        print(f"Macro F1: {results['macro_f1']:.4f}  "
              f"(95% CI {results['macro_f1_ci95'][0]:.4f}-{results['macro_f1_ci95'][1]:.4f})")
        print(f"Cohen's kappa: {results['kappa']:.4f}")
        sui = results["per_class"]["suicidal"]
        print(f"Suicidal recall: {sui['recall']:.4f} - misses {1 - sui['recall']:.0%} "
              f"of {sui['support']:,} suicidal test texts")
        missed = (y_test == "suicidal") & (y_pred != "suicidal")
        if missed.any():
            print("Missed suicidal texts are classified as:")
            print(pd.Series(y_pred[missed]).value_counts().to_string())

        if make_figure:
            fig, ax = plt.subplots(figsize=(10, 8))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, ax=ax, xticks_rotation=45, colorbar=False,
                normalize="true", values_format=".0%",
            )
            plt.title(title or f"{name.replace('_', ' ')} - Confusion Matrix (row-normalised)")
            plt.tight_layout()
            # PNG for slides, PDF for the report.
            plt.savefig(fig_dir / f"{name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
            plt.savefig(fig_dir / f"{name}_confusion_matrix.pdf", bbox_inches="tight")
            plt.show()
        return results

    return evaluate_model


def load_all_metrics(metrics_dir, test_ids_sha256):
    """Load every results/metrics/*.json scored on the frozen test set. Skip and warn on
    any file whose test-set fingerprint is missing or does not match test_ids_sha256, so a
    drifted run never enters the ladder."""
    metrics = []
    for p in sorted(metrics_dir.glob('*.json')):
        m = json.loads(p.read_text())
        stamp = m.get('provenance', {}).get('test_ids_sha256')
        if stamp is None:
            print(f"WARNING: {p.name} carries no test-set fingerprint - excluded "
                  f"(re-score it through evaluate_model()).")
        elif stamp != test_ids_sha256:
            print(f"WARNING: {p.name} was scored on a DIFFERENT test set "
                  f"({stamp[:12]}... vs manifest {test_ids_sha256[:12]}...) - excluded.")
        else:
            metrics.append(m)
    return metrics
