"""BERT fine-tuning for Part 2: the train-and-predict routine shared by every retraining arm
(2a baseline, 2b augmentation, 2e distillation) plus the encoder-parameterized 2e variant.

Training runs on the Colab GPU; scoring is local through the evaluation contract, so these
routines only train and stage raw predictions to results/colab_runs/. Hyperparameters and
run paths come from a TrainConfig (config/config.yaml); data and tokenizers are passed in."""

import json
from typing import Any, Callable, Optional

import accelerate
import pandas as pd
import torch
import transformers
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


from .config import TrainConfig
from ..utils.metrics import logits_to_preds


def _train_model(model_name, train_ds, tokenizer, run_name, seed, *, cfg, classes, id2label, label2id):
    """Seed, build the model and the TrainingArguments/Trainer every arm shares, train, and
    return the trainer and loss history. Prediction and file IO stay in the callers."""
    set_seed(seed)
    torch.manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(classes), id2label=id2label, label2id=label2id)
    args = TrainingArguments(
        output_dir=str(cfg.ckpt_dir / run_name),
        seed=seed,
        data_seed=seed,
        learning_rate=cfg.lr,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch,
        per_device_eval_batch_size=cfg.eval_batch,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        fp16=torch.cuda.is_available(),
        eval_strategy='no',
        save_strategy='no',
        logging_steps=cfg.logging_steps,
        report_to='none',
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    trainer.train()
    losses = [h['loss'] for h in trainer.state.log_history if 'loss' in h]
    final_loss = losses[-1] if losses else float('nan')
    return model, trainer, args, losses, final_loss


def train_and_predict(
    train_df: pd.DataFrame,
    run_name: str,
    seed: int,
    *,
    cfg: TrainConfig,
    classes: list[str],
    id2label: dict[int, str],
    label2id: dict[str, int],
    tokenizer: Any,
    tokenize_frame: Callable[..., Any],
    test_ds: Any,
    val_sanity_ds: Any,
    test_df: pd.DataFrame,
    save_canonical: bool = True,
) -> dict[str, Any]:
    """Fine-tune cfg.model_name, predict the test set, write CSV and JSON summary."""
    model, trainer, args, train_losses, final_loss = _train_model(
        cfg.model_name, tokenize_frame(train_df, with_labels=True), tokenizer, run_name, seed,
        cfg=cfg, classes=classes, id2label=id2label, label2id=label2id)

    val_spread = len(set(logits_to_preds(trainer.predict(val_sanity_ds).predictions)))

    y_pred = [id2label[int(i)] for i in logits_to_preds(trainer.predict(test_ds).predictions)]
    csv_path = cfg.raw_pred_dir / f'{run_name}.csv'
    pd.DataFrame({'y_pred': y_pred}, index=test_df.index).rename_axis('test_id').to_csv(csv_path)

    if seed == cfg.canonical_seed and save_canonical:
        model_dir = cfg.raw_pred_dir / f'{run_name}_model'
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        (model_dir / 'training_args.json').write_text(json.dumps(args.to_dict(), indent=2, default=str))

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        'run_name': run_name,
        'seed': seed,
        'final_loss': float(final_loss),
        'loss_history': train_losses,
        'val_class_spread': int(val_spread),
        'torch': torch.__version__,
        'transformers': transformers.__version__,
        'accelerate': accelerate.__version__,
        'csv_path': str(csv_path)
    }
    (cfg.raw_pred_dir / f'{run_name}.json').write_text(json.dumps(summary, indent=2))
    return summary


def train_2e(
    train_df: pd.DataFrame,
    run_name: str,
    eval_df: pd.DataFrame,
    eval_tag: str,
    *,
    cfg: TrainConfig,
    classes: list[str],
    id2label: dict[int, str],
    label2id: dict[str, int],
    run_training: bool,
    model_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Fine-tune model_name on train_df (2a recipe, canonical seed), predict eval_df, and stage
    cfg.raw_pred_dir/{run_name}_{eval_tag}.csv (eval_df.index-aligned, one y_pred column). Scoring
    is local. Returns a summary dict, or None when not on the GPU path. model_name defaults to
    cfg.model_name (the 2a encoder)."""
    if not run_training:
        print(f'{run_name}: RUN_TRAINING is False; train on Colab. Skipping.')
        return None
    model_name = model_name if model_name is not None else cfg.model_name
    tok = AutoTokenizer.from_pretrained(model_name)

    def tok_frame(frame, with_labels):
        ds = Dataset.from_pandas(frame.reset_index(drop=True))
        if with_labels:
            ds = ds.map(lambda b: {'labels': [label2id[s] for s in b['status']]}, batched=True)
        return ds.map(lambda b: tok(b['text'], truncation=True, max_length=cfg.max_length), batched=True,
                      remove_columns=[c for c in ('text', 'status') if c in ds.column_names])

    model, trainer, args, losses, final_loss = _train_model(
        model_name, tok_frame(train_df, with_labels=True), tok, run_name, cfg.canonical_seed,
        cfg=cfg, classes=classes, id2label=id2label, label2id=label2id)
    if not final_loss < cfg.max_final_train_loss:
        print(f'WARNING: {run_name} final train loss {final_loss:.4f} >= {cfg.max_final_train_loss}; '
              'investigate before trusting it.')

    y_pred = [id2label[int(i)] for i in logits_to_preds(trainer.predict(tok_frame(eval_df, with_labels=False)).predictions)]
    csv_path = cfg.raw_pred_dir / f'{run_name}_{eval_tag}.csv'
    pd.DataFrame({'y_pred': y_pred}, index=eval_df.index).rename_axis('row_id').to_csv(csv_path)
    spread = len(set(y_pred))
    print(f'{run_name} [{model_name}]: loss {final_loss:.4f}, {spread}/{len(classes)} classes '
          f'on {eval_tag} ({len(eval_df):,} rows) -> {csv_path.name}')

    del model, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {'run_name': run_name, 'encoder': model_name, 'eval_tag': eval_tag,
            'final_loss': float(final_loss), 'class_spread': int(spread), 'csv_path': str(csv_path)}
