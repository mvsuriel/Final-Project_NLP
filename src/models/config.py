"""Training config types. No torch/transformers imports so local runs stay light."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """Fine-tuning hyperparameters and run paths from config/config.yaml."""

    model_name: str
    max_length: int
    train_batch: int
    eval_batch: int
    lr: float
    epochs: int
    warmup_steps: int
    weight_decay: float
    logging_steps: int
    max_final_train_loss: float
    canonical_seed: int
    ckpt_dir: Path
    raw_pred_dir: Path
