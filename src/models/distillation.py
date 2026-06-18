"""Knowledge distillation for Part 4: train a small student (DistilBERT) against the frozen
Part-3 teacher (part3_bert_pct100), so the student matches teacher accuracy at a fraction of
the parameters.

Same division of labour as the Part-3 driver: training runs on the Colab GPU and stages raw
test predictions to results/colab_runs/; scoring is local through the evaluation contract
(notebook 00 / evaluate_model). The student trains on the full 64k set, so it reuses the
Part-3 BERT regime (warmup is a fraction of total steps); the distillation knobs (teacher,
student encoder, temperature, alpha) are passed in.

The teacher (bert-base) and student (distilbert-base-uncased) share the bert-base-uncased
vocabulary, so one tokenizer feeds both. The Trainer strips token_type_ids for the student
(DistilBERT does not accept them); the teacher is called with input_ids/attention_mask only.
"""

import json
import math

import accelerate
import pandas as pd
import torch
import torch.nn.functional as F
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

from ..utils.metrics import logits_to_preds


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """Combined loss = alpha * KD + (1 - alpha) * cross-entropy. The KD term is the KL
    divergence between temperature-softened student and teacher distributions, scaled by
    temperature**2 so its gradient magnitude is comparable to the supervised term."""
    kd_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    ce_loss = F.cross_entropy(student_logits, labels)
    total = alpha * kd_loss + (1 - alpha) * ce_loss
    return total, kd_loss, ce_loss


class DistillationTrainer(Trainer):
    """HF Trainer whose loss is distillation_loss against a frozen teacher. The teacher stays
    in eval mode and contributes no gradients."""

    def __init__(self, teacher_model=None, temperature=2.0, alpha=0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        student_outputs = model(**inputs)
        with torch.no_grad():
            teacher_logits = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            ).logits
        loss, _, _ = distillation_loss(
            student_outputs.logits, teacher_logits, labels, self.temperature, self.alpha
        )
        return (loss, student_outputs) if return_outputs else loss


def train_distilled(
    train_df: pd.DataFrame,
    run_name: str,
    teacher_dir: str,
    seed: int,
    *,
    classes: list[str],
    id2label: dict[int, str],
    label2id: dict[str, int],
    test_df: pd.DataFrame,
    val_sanity_df: pd.DataFrame,
    raw_pred_dir,
    ckpt_dir,
    student_model_name: str,
    max_length: int,
    train_batch: int,
    eval_batch: int,
    lr: float,
    epochs: int,
    warmup_frac: float,
    weight_decay: float,
    logging_steps: int,
    temperature: float,
    alpha: float,
    save_checkpoint_as: str | None = None,
) -> dict:
    """Distill a student from the frozen teacher at teacher_dir on the full train set, predict
    the test set, and stage raw_pred_dir/{run_name}.csv (test_df.index-aligned, one y_pred
    column) plus a {run_name}.json sidecar. Scoring is local through evaluate_model. Returns the
    summary dict (same shape as the Part-3 driver, with the distillation fields added).

    teacher_dir : path to the saved teacher model (e.g. results/colab_runs/part3_bert_pct100_model).
    """
    set_seed(seed)
    torch.manual_seed(seed)

    steps_per_epoch = math.ceil(len(train_df) / train_batch)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_frac))
    print(f"{run_name}: {len(train_df):,} rows -> {total_steps:,} optimizer steps ({warmup_steps} warmup)")

    tokenizer = AutoTokenizer.from_pretrained(teacher_dir)

    def tokenize_frame(frame, with_labels):
        ds = Dataset.from_pandas(frame.reset_index(drop=True))
        if with_labels:
            ds = ds.map(lambda b: {"labels": [label2id[s] for s in b["status"]]}, batched=True)
        return ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=max_length),
            batched=True,
            remove_columns=[c for c in ("text", "status") if c in ds.column_names],
        )

    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_dir)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = AutoModelForSequenceClassification.from_pretrained(
        student_model_name, num_labels=len(classes), id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=str(ckpt_dir / run_name),
        seed=seed,
        data_seed=seed,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch,
        per_device_eval_batch_size=eval_batch,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        fp16=torch.cuda.is_available(),
        eval_strategy="no",
        save_strategy="no",
        logging_steps=logging_steps,
        report_to="none",
    )
    trainer = DistillationTrainer(
        teacher_model=teacher,
        temperature=temperature,
        alpha=alpha,
        model=student,
        args=args,
        train_dataset=tokenize_frame(train_df, with_labels=True),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    if torch.cuda.is_available():
        teacher.to(trainer.args.device)
    trainer.train()

    train_losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    final_loss = train_losses[-1] if train_losses else float("nan")
    print(f"{run_name}: final train loss {final_loss:.4f}")

    val_spread = len(set(logits_to_preds(trainer.predict(tokenize_frame(val_sanity_df, with_labels=False)).predictions)))
    print(f"{run_name}: {val_spread}/{len(classes)} classes predicted on the val sanity sample")

    y_pred = [id2label[int(i)] for i in logits_to_preds(trainer.predict(tokenize_frame(test_df, with_labels=False)).predictions)]
    csv_path = raw_pred_dir / f"{run_name}.csv"
    pd.DataFrame({"y_pred": y_pred}, index=test_df.index).rename_axis("test_id").to_csv(csv_path)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    if save_checkpoint_as:
        model_dir = raw_pred_dir / save_checkpoint_as
        trainer.save_model(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        (model_dir / "training_args.json").write_text(json.dumps(args.to_dict(), indent=2, default=str))
        print(f"{run_name}: final model + training config saved to {model_dir}")

    del student, teacher, trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = {
        "run_name": run_name,
        "seed": seed,
        "n_train_rows": int(len(train_df)),
        "total_steps": int(total_steps),
        "warmup_steps": int(warmup_steps),
        "final_loss": float(final_loss),
        "loss_history": train_losses,
        "val_class_spread": int(val_spread),
        "teacher_dir": str(teacher_dir),
        "student_model": student_model_name,
        "temperature": temperature,
        "alpha": alpha,
        "teacher_parameters": int(teacher_params),
        "student_parameters": int(student_params),
        "parameter_reduction_pct": float((1 - student_params / teacher_params) * 100),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "csv_path": str(csv_path),
    }
    (raw_pred_dir / f"{run_name}.json").write_text(json.dumps(summary, indent=2))
    return summary
