"""Setup shared by notebooks 00-03.

No torch/transformers imports here, so local scoring stays light; the GPU block lives in the
notebook behind `if RUN_TRAINING`. The project-root search must stay in the notebook since it
is what puts src/ on the path. Everything after that runs here.
"""

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


def bootstrap(project_root):
    """Load config.yaml, seed random and numpy, derive the results paths.

    Returns a namespace with cfg (parsed YAML), SEED, IN_COLAB, RUN_TRAINING, the six paths
    (DATA_DIR, RESULTS_DIR, METRICS_DIR, PRED_DIR, FIG_DIR, RAW_PRED_DIR) and save_fig. Per-part
    hyperparameters stay in the notebook, read from cfg.
    """
    project_root = Path(project_root)
    cfg = yaml.safe_load((project_root / 'config' / 'config.yaml').read_text())

    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)

    in_colab = 'google.colab' in sys.modules

    data_dir = project_root / 'data' / 'processed'
    results_dir = project_root / 'results'
    metrics_dir = results_dir / 'metrics'
    pred_dir = results_dir / 'predictions'
    fig_dir = results_dir / 'figures'
    raw_pred_dir = results_dir / 'colab_runs'
    fig_dir.mkdir(parents=True, exist_ok=True)
    raw_pred_dir.mkdir(parents=True, exist_ok=True)

    def save_fig(name):
        """Save the current figure to results/figures/ as PNG and PDF, then show."""
        import matplotlib.pyplot as plt
        plt.savefig(fig_dir / f'{name}.png', dpi=150, bbox_inches='tight')
        plt.savefig(fig_dir / f'{name}.pdf', bbox_inches='tight')
        plt.show()

    return SimpleNamespace(
        cfg=cfg, SEED=seed, IN_COLAB=in_colab, RUN_TRAINING=in_colab,
        PROJECT_ROOT=project_root, DATA_DIR=data_dir, RESULTS_DIR=results_dir,
        METRICS_DIR=metrics_dir, PRED_DIR=pred_dir, FIG_DIR=fig_dir, RAW_PRED_DIR=raw_pred_dir,
        save_fig=save_fig,
    )
