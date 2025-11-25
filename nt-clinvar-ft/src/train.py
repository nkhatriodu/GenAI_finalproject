import os
import json
from typing import Dict, Any, Tuple

import torch
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .config_utils import load_config, ensure_dirs, set_seed, get_device_and_precision
from .data import (
    build_or_load_paired_dataframe,
    subsample_df,
    prepare_splits,
    ClinVarPairedDataset,
    DataCollatorSiamese,
)
from .model import build_tokenizer, SiameseNTClassifier, apply_lora_if_enabled


# ---------- Metric computation ----------


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = probs.argmax(axis=1)
    pos_probs = probs[:, 1]

    out = {}
    try:
        out["auroc"] = roc_auc_score(labels, pos_probs)
    except Exception:
        out["auroc"] = 0.0
    out["auprc"] = average_precision_score(labels, pos_probs)
    out["accuracy"] = accuracy_score(labels, preds)
    out["f1"] = f1_score(labels, preds, zero_division=0)
    out["precision"] = precision_score(labels, preds, zero_division=0)
    out["recall"] = recall_score(labels, preds, zero_division=0)
    return out


# ---------- CSV logging callback ----------


class CSVLoggerCallback(TrainerCallback):
    """Logs train & eval metrics to CSV (step, epoch, split, metric, value)."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w") as f:
                f.write("step,epoch,split,metric,value\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        epoch = state.epoch if state.epoch is not None else -1

        with open(self.csv_path, "a") as f:
            for k, v in logs.items():
                if not isinstance(v, (int, float)):
                    continue
                if not torch.isfinite(torch.tensor(v)):
                    continue

                if k.startswith("eval_"):
                    split = "eval"
                    metric = k.replace("eval_", "")
                elif k.startswith("train_"):
                    split = "train"
                    metric = k.replace("train_", "")
                else:
                    split = "train"
                    metric = k

                f.write(f"{step},{epoch},{split},{metric},{v}\n")


# ---------- Trainer builder ----------


def build_trainer(cfg: Dict[str, Any]) -> Tuple[Trainer, Any, str]:
    """Build Trainer + test dataset + run directory.

    Returns
    -------
    trainer : transformers.Trainer
    test_ds : Dataset
    run_dir : str
    """
    ensure_dirs(cfg)
    set_seed(cfg["training"]["seed"])
    device, use_fp16, use_bf16 = get_device_and_precision()

    # Run-specific output dir
    base_out = cfg["paths"]["output_dir"]
    run_name = cfg["model"].get("run_name") or f"embed_{cfg['model']['embed_method']}"
    run_dir = os.path.join(base_out, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # 1) DataFrame with all variants
    df_all = build_or_load_paired_dataframe(cfg)
    df_all = subsample_df(df_all, cfg["data"]["subset_per_class"], cfg["training"]["seed"])

    # 2) Splits + class weights
    train_df, val_df, test_df, class_weights = prepare_splits(cfg, df_all)

    # 3) Tokenizer + datasets
    tokenizer = build_tokenizer(cfg)
    max_tokens = cfg["data"]["max_tokens"]

    train_ds = ClinVarPairedDataset(train_df, tokenizer, max_tokens)
    val_ds = ClinVarPairedDataset(val_df, tokenizer, max_tokens)
    test_ds = ClinVarPairedDataset(test_df, tokenizer, max_tokens)

    # 4) Model
    model = SiameseNTClassifier(
        base_model_name=cfg["model"]["name"],
        num_labels=2,
        embed_method=cfg["model"]["embed_method"],
        class_weights=class_weights,
    )
    model = apply_lora_if_enabled(cfg, model)
    model.to(device)

    # 5) TrainingArguments
    log_dir = os.path.join(run_dir, "runs")
    csv_log_path = os.path.join(run_dir, "logs", "train_log.csv")

    
    from inspect import signature

    # Safely coerce basic types from YAML (in case they are strings)
    lr = float(cfg["training"]["lr"])
    batch_size = int(cfg["training"]["batch_size"])
    num_epochs = int(cfg["training"]["num_epochs"])

    ta_kwargs = dict(
        output_dir=run_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model=cfg["training"]["metric_for_best"],
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=cfg["training"]["logging_steps"],
        logging_first_step=True,
        dataloader_num_workers=cfg["training"]["num_workers"],
        dataloader_pin_memory=True,
        fp16=use_fp16,
        bf16=use_bf16,
        report_to=["tensorboard"],
        logging_dir=log_dir,
    )

    ta_sig = signature(TrainingArguments.__init__)
    allowed = set(ta_sig.parameters.keys())
    # Filter out any kwargs that this transformers version does not support
    ta_filtered = {k: v for k, v in ta_kwargs.items() if k in allowed}
    args = TrainingArguments(**ta_filtered)




    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorSiamese(),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=1e-4,
            ),
            CSVLoggerCallback(csv_log_path),
        ],
    )

    return trainer, test_ds, run_dir


# ---------- End-to-end train + evaluate ----------


def train_and_evaluate(config_path: str) -> Dict[str, Any]:
    """Full pipeline: train, evaluate, and save metrics + best model.

    Steps:
      - load config
      - build trainer
      - train with early stopping
      - evaluate on val + test
      - save best model/tokenizer and metrics.json
    """
    cfg = load_config(config_path)
    trainer, test_ds, run_dir = build_trainer(cfg)

    print(f"Run directory: {run_dir}")
    train_result = trainer.train()
    print("Train result:", train_result)

    val_metrics = trainer.evaluate()
    print("Validation metrics:", val_metrics)

    test_metrics = trainer.evaluate(test_ds)
    print("Test metrics:", test_metrics)

    # Best checkpoint directory
    best_dir = trainer.state.best_model_checkpoint or run_dir
    print(f"Best checkpoint: {best_dir}")

    # Save model + tokenizer from best checkpoint
    trainer.save_model(best_dir)
    if trainer.tokenizer is not None:
        trainer.tokenizer.save_pretrained(best_dir)

    # Save metrics
    metrics_path = os.path.join(run_dir, "metrics.json")
    all_metrics = {
        "val": val_metrics,
        "test": test_metrics,
        "best_checkpoint": best_dir,
    }
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    return all_metrics
