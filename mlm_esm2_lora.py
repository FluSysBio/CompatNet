#!/usr/bin/env python3
"""
mlm_esm2_lora.py — ESM-2 LoRA Masked Language Model Fine-Tuning
================================================================
Fine-tunes ESM-2 on ACE2 and RBD protein sequences using LoRA adapters
for masked language modelling (MLM).

Usage:
    python mlm_esm2_lora.py --ace2 ace2.fasta --rbd rbd.fasta --out ./output

    # To build a merged RBD FASTA from DMS and label sources:
    python mlm_esm2_lora.py --build-rbd-fasta \
        --dms-fasta dms.fasta --label-fasta labels.fasta --out-fasta merged_rbd.fasta
"""

import os

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import csv
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from torch.utils.data import Dataset
import transformers as _tf
import peft as _pf
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    EsmForMaskedLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model



# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CFG: Dict[str, Any] = {
    "ACE2_FASTA": "",
    "RBD_FASTA": "",
    "OUT_DIR": "",

    # Model
    "BASE_ID": "facebook/esm2_t12_35M_UR50D",

    # Training
    "EPOCHS": 15,
    "BATCH": 32,
    "ACCUM": 1,
    "LR": 1e-4,
    "WEIGHT_DECAY": 0.01,
    "WARMUP_RATIO": 0.06,
    "MAX_LEN": 1024,
    "MAX_GRAD_NORM": 1.0,
    "FP16": True,
    "BF16": False,
    "SEED": 42,
    "GRAD_CKPT": True,

    # Masking
    "MLM_PROB": 0.15,
    "SPAN_MIN": 1,
    "SPAN_MAX": 5,

    # Data split
    "VAL_FRAC": 0.10,
    "MIN_VAL": 5,

    # Checkpoint (0 = once per epoch)
    "EVAL_STEPS": 0,
    "SAVE_STEPS": 0,
    "SAVE_TOTAL": 3,

    # LoRA
    "LORA_R": 16,
    "LORA_ALPHA": 32,
    "LORA_DROP": 0.05,
    "LORA_TARGETS": [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    ],
}


# =============================================================================
# Logging
# =============================================================================

def setup_logging(out_dir: str) -> logging.Logger:
    """Configure file and console logging under ``out_dir/logs/``."""
    logs_dir = os.path.join(out_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"mlm_lora_{timestamp}.log")

    logger = logging.getLogger("mlm_lora")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    logger.info(f"Log file: {log_path}")
    return logger


# =============================================================================
# Reproducibility
# =============================================================================

def set_all_seeds(seed: int) -> None:
    """Seed all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


# =============================================================================
# FASTA Utilities
# =============================================================================

def read_fasta(path: str, min_len: int = 10) -> List[Tuple[str, str]]:
    """Parse a FASTA file and return ``(id, sequence)`` pairs.

    Sequences shorter than ``min_len`` residues are discarded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"FASTA file not found: {path}")

    records = []
    for rec in SeqIO.parse(path, "fasta"):
        seq = str(rec.seq).upper().replace(" ", "").replace("-", "")
        if len(seq) >= min_len:
            records.append((rec.id, seq))

    if not records:
        raise ValueError(
            f"No sequences with at least {min_len} residues found in: {path}"
        )
    return records


def build_merged_rbd_fasta(dms_fasta: str, label_fasta: str, out_fasta: str) -> int:
    """Merge two RBD FASTA sources, deduplicating by sequence content.

    Returns the number of unique sequences written.
    """
    dms_records = list(SeqIO.parse(dms_fasta, "fasta"))
    label_records = list(SeqIO.parse(label_fasta, "fasta"))

    seen, merged = set(), []
    for rec in dms_records + label_records:
        seq = str(rec.seq).upper().replace("-", "").replace(" ", "")
        if seq and seq not in seen:
            seen.add(seq)
            merged.append(SeqRecord(Seq(seq), id=rec.id, description=""))

    SeqIO.write(merged, out_fasta, "fasta")
    print(f"Merged RBD FASTA: {len(merged)} unique sequences written to {out_fasta}")
    return len(merged)


# =============================================================================
# Data Splitting
# =============================================================================

def stratified_split(
    sources: Dict[str, List[Tuple[str, str]]],
    val_frac: float,
    min_val: int,
    seed: int,
    logger: logging.Logger,
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Split each source independently, then combine into train and val sets.

    Each record is tagged with its source label as ``(source, id, sequence)``.
    """
    rng = random.Random(seed)
    train_all, val_all = [], []

    for src, records in sources.items():
        indices = list(range(len(records)))
        rng.shuffle(indices)

        n_val = max(min_val, int(len(records) * val_frac))
        n_val = min(n_val, len(records) - 1)
        val_indices = set(indices[:n_val])

        for i, (sid, seq) in enumerate(records):
            entry = (src, sid, seq)
            (val_all if i in val_indices else train_all).append(entry)

        logger.info(
            f"  {src:6s}: {len(records) - n_val:>6d} train  /  {n_val:>4d} val"
        )

    rng.shuffle(train_all)
    rng.shuffle(val_all)
    return train_all, val_all


# =============================================================================
# Dataset
# =============================================================================

class ProteinMLMDataset(Dataset):
    """Tokenises protein sequences for masked language modelling."""

    def __init__(
        self,
        records: List[Tuple[str, str, str]],
        tokenizer,
        max_length: int = 1024,
    ) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        _src, _sid, seq = self.records[idx]
        enc = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            add_special_tokens=True,
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        }


# =============================================================================
# Data Collator — Span Masking
# =============================================================================

@dataclass
class ProteinSpanMaskCollator:
    """Pads a batch and applies span-based masking.

    Masking follows the standard 80/10/10 split:
      - 80 % of selected tokens are replaced with ``[MASK]``
      - 10 % are replaced with a random token
      - 10 % are left unchanged (but still contribute to the loss)
    """

    tokenizer: Any
    mlm_prob: float = 0.15
    span_min: int = 1
    span_max: int = 5
    pad_to_multiple_of: int = 8

    def __call__(
        self, features: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            features,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        input_ids = batch["input_ids"].clone()
        labels = input_ids.clone()

        # Identify special tokens (CLS, EOS, PAD) — never masked
        special = torch.zeros_like(input_ids, dtype=torch.bool)
        for b in range(input_ids.size(0)):
            mask_flags = self.tokenizer.get_special_tokens_mask(
                input_ids[b].tolist(), already_has_special_tokens=True
            )
            special[b] = torch.tensor(mask_flags, dtype=torch.bool)

        masked = self._span_mask(input_ids, special)
        labels[~masked] = -100  # only compute loss on masked positions

        rand = torch.rand_like(input_ids, dtype=torch.float)
        to_mask = masked & (rand < 0.80)
        to_random = masked & (rand >= 0.80) & (rand < 0.90)

        input_ids[to_mask] = self.tokenizer.mask_token_id
        random_tokens = torch.randint(
            len(self.tokenizer), input_ids.shape, dtype=torch.long
        )
        input_ids[to_random] = random_tokens[to_random]

        return {
            "input_ids": input_ids,
            "attention_mask": batch["attention_mask"],
            "labels": labels,
        }

    def _span_mask(
        self, input_ids: torch.Tensor, special: torch.Tensor
    ) -> torch.Tensor:
        B, L = input_ids.shape
        masked = torch.zeros(B, L, dtype=torch.bool)
        prob = torch.full((B, L), self.mlm_prob)
        prob[special] = 0.0
        seeds = torch.bernoulli(prob).bool()

        for b in range(B):
            for p in seeds[b].nonzero(as_tuple=False).flatten().tolist():
                span_len = random.randint(self.span_min, self.span_max)
                for pos in range(p, min(p + span_len, L)):
                    if not special[b, pos]:
                        masked[b, pos] = True

            # Guarantee at least one masked token per sequence
            if not masked[b].any():
                candidates = (~special[b]).nonzero(as_tuple=False).flatten()
                if len(candidates):
                    masked[b, candidates[random.randrange(len(candidates))]] = True

        return masked


# =============================================================================
# LoRA Setup
# =============================================================================

def build_lora_model(base_mlm, cfg: Dict[str, Any], logger: logging.Logger):
    """Attach LoRA adapters to ``base_mlm`` using verified target modules.

    Target module names that are not present in the model are skipped with a
    warning rather than raising an error, so the script remains compatible with
    different ESM-2 checkpoints.
    """
    all_module_names = [name for name, _ in base_mlm.named_modules()]
    verified = [t for t in cfg["LORA_TARGETS"] if any(t in n for n in all_module_names)]
    missing = [t for t in cfg["LORA_TARGETS"] if t not in verified]

    logger.info(
        f"LoRA target modules: {len(verified)} found, {len(missing)} not present"
    )
    for t in verified:
        example = next((n for n in all_module_names if t in n), "—")
        logger.info(f"  [found]   {t}  (e.g. {example})")
    for t in missing:
        logger.warning(f"  [missing] {t}  — skipped")

    if not verified:
        raise RuntimeError(
            "None of the specified LoRA target modules were found in the model. "
            "Check LORA_TARGETS against the model architecture."
        )

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=cfg["LORA_R"],
        lora_alpha=cfg["LORA_ALPHA"],
        lora_dropout=cfg["LORA_DROP"],
        bias="none",
        target_modules=verified,
    )
    model = get_peft_model(base_mlm, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.4f} %)"
    )
    return model


# =============================================================================
# Training Arguments
# =============================================================================

def build_training_args(
    out_dir: str, cfg: Dict[str, Any], eval_steps: int, save_steps: int
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg["EPOCHS"],
        per_device_train_batch_size=cfg["BATCH"],
        per_device_eval_batch_size=max(1, cfg["BATCH"] // 2),
        gradient_accumulation_steps=cfg["ACCUM"],
        learning_rate=cfg["LR"],
        weight_decay=cfg["WEIGHT_DECAY"],
        warmup_ratio=cfg["WARMUP_RATIO"],
        lr_scheduler_type="cosine",
        fp16=cfg["FP16"] and torch.cuda.is_available(),
        bf16=cfg["BF16"],
        max_grad_norm=cfg["MAX_GRAD_NORM"],
        logging_steps=max(1, eval_steps // 4),
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=cfg["SAVE_TOTAL"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=cfg["GRAD_CKPT"],
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to=["none"],
    )


# =============================================================================
# Metrics
# =============================================================================

def save_metrics(
    trainer, out_dir: str, logger: logging.Logger
) -> Dict[str, Any]:
    """Persist training history and a metrics summary to ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    log_history = list(getattr(trainer.state, "log_history", []))

    # Full history — JSON
    json_path = os.path.join(out_dir, "log_history.json")
    with open(json_path, "w") as f:
        json.dump(log_history, f, indent=2)
    logger.info(f"Training log history: {json_path}")

    # Full history — CSV
    all_keys = list(dict.fromkeys(k for row in log_history for k in row))
    csv_path = os.path.join(out_dir, "log_history.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in log_history:
            writer.writerow({k: row.get(k, "") for k in all_keys})
    logger.info(f"Training log CSV:     {csv_path}")

    # Summary
    best_eval = min(
        (float(r["eval_loss"]) for r in log_history if "eval_loss" in r),
        default=None,
    )
    last_train = next(
        (float(r["train_loss"]) for r in reversed(log_history) if "train_loss" in r),
        None,
    )
    summary = {
        "best_eval_loss": best_eval,
        "best_perplexity": math.exp(best_eval) if best_eval is not None else None,
        "final_train_loss": last_train,
    }
    summary_path = os.path.join(out_dir, "metrics_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Metrics summary:      {summary_path}")

    if best_eval is not None:
        logger.info(f"Best eval loss:   {best_eval:.4f}")
        logger.info(f"Best perplexity:  {math.exp(best_eval):.2f}")
    if last_train is not None:
        logger.info(f"Final train loss: {last_train:.4f}")

    return summary


# =============================================================================
# Main
# =============================================================================

def main(cfg: Dict[str, Any]) -> None:
    out_dir = cfg["OUT_DIR"]
    os.makedirs(out_dir, exist_ok=True)

    logger = setup_logging(out_dir)
    set_all_seeds(cfg["SEED"])

    device_label = (
        f"cuda ({torch.cuda.get_device_name(0)})"
        if torch.cuda.is_available()
        else "cpu"
    )

    logger.info("=" * 70)
    logger.info("  ESM-2 LoRA MLM Fine-Tuning")
    logger.info("=" * 70)
    logger.info(f"  PyTorch      : {torch.__version__}")
    logger.info(f"  Transformers : {_tf.__version__}")
    logger.info(f"  PEFT         : {_pf.__version__}")
    logger.info(f"  Device       : {device_label}")
    logger.info(f"  Base model   : {cfg['BASE_ID']}")
    logger.info(f"  Output dir   : {out_dir}")
    logger.info(f"  Seed         : {cfg['SEED']}")
    logger.info(f"  Epochs       : {cfg['EPOCHS']}")
    logger.info(f"  Batch size   : {cfg['BATCH']}  (accumulation steps: {cfg['ACCUM']})")
    logger.info(f"  Learning rate: {cfg['LR']}")
    logger.info(f"  LoRA rank    : {cfg['LORA_R']}  alpha: {cfg['LORA_ALPHA']}")
    logger.info(f"  MLM prob     : {cfg['MLM_PROB']}  spans: [{cfg['SPAN_MIN']}–{cfg['SPAN_MAX']}]")

    # Tokenizer and base model 
    logger.info("Loading tokenizer and base ESM-2 model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["BASE_ID"], do_lower_case=False)
    base_mlm = EsmForMaskedLM.from_pretrained(cfg["BASE_ID"])

    # LoRA adapters
    logger.info("Attaching LoRA adapters...")
    model = build_lora_model(base_mlm, cfg, logger)
    if cfg["GRAD_CKPT"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Data 
    logger.info("Reading FASTA files...")
    ace2_records = read_fasta(cfg["ACE2_FASTA"])
    rbd_records = read_fasta(cfg["RBD_FASTA"])
    logger.info(f"  ACE2 : {len(ace2_records):>6d} sequences  ←  {Path(cfg['ACE2_FASTA']).name}")
    logger.info(f"  RBD  : {len(rbd_records):>6d} sequences  ←  {Path(cfg['RBD_FASTA']).name}")

    logger.info("Splitting into train and validation sets...")
    train_records, val_records = stratified_split(
        sources={"ACE2": ace2_records, "RBD": rbd_records},
        val_frac=cfg["VAL_FRAC"],
        min_val=cfg["MIN_VAL"],
        seed=cfg["SEED"],
        logger=logger,
    )
    logger.info(f"  Total train : {len(train_records):>6d}")
    logger.info(f"  Total val   : {len(val_records):>6d}")

    train_ds = ProteinMLMDataset(train_records, tokenizer, cfg["MAX_LEN"])
    val_ds = ProteinMLMDataset(val_records, tokenizer, cfg["MAX_LEN"])
    collator = ProteinSpanMaskCollator(
        tokenizer=tokenizer,
        mlm_prob=cfg["MLM_PROB"],
        span_min=cfg["SPAN_MIN"],
        span_max=cfg["SPAN_MAX"],
    )

    # Evaluation  
    steps_per_epoch = math.ceil(
        len(train_ds) / (cfg["BATCH"] * max(1, cfg["ACCUM"]))
    )
    eval_steps = cfg["EVAL_STEPS"] if cfg["EVAL_STEPS"] > 0 else steps_per_epoch
    save_steps = cfg["SAVE_STEPS"] if cfg["SAVE_STEPS"] > 0 else eval_steps
    logger.info(f"Steps per epoch : {steps_per_epoch}")
    logger.info(f"Eval / save     : every {eval_steps} steps")
    logger.info(f"Total steps     : {steps_per_epoch * cfg['EPOCHS']}")

    # Training 
    training_args = build_training_args(out_dir, cfg, eval_steps, save_steps)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    logger.info("=" * 70)
    logger.info("Starting training...")
    logger.info("=" * 70)
    train_result = trainer.train()
    logger.info("Training complete — best checkpoint loaded.")
    logger.info(
        f"Total runtime    : {train_result.metrics.get('train_runtime', 0):.1f}s"
    )
    logger.info(
        f"Throughput       : {train_result.metrics.get('train_samples_per_second', 0):.2f} samples/s"
    )

    # Metrics
    metrics = save_metrics(trainer, out_dir, logger)

    # Save LoRA adapter 
    adapter_dir = os.path.join(out_dir, "lora_adapter")
    tokenizer.save_pretrained(adapter_dir)
    model.save_pretrained(adapter_dir)
    logger.info(f"LoRA adapter saved: {adapter_dir}")

    #Merge LoRA weights into base model
    merged_dir = os.path.join(out_dir, "merged_backbone")
    try:
        logger.info("Merging LoRA weights into base model...")
        merged = model.merge_and_unload()
        merged.esm.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        logger.info(f"Merged backbone saved: {merged_dir}")
        for f in sorted(Path(merged_dir).iterdir()):
            logger.info(f"  {f.name:<35s}  {f.stat().st_size / 1024:>10.1f} KB")
    except Exception as exc:
        logger.error(
            f"Merge failed: {exc!r}. "
            "Use the LoRA adapter directory for inference instead."
        )
        merged_dir = None

    #Training card 
    card = {
        "timestamp": datetime.now().isoformat(),
        "base_model": cfg["BASE_ID"],
        "ace2_sequences": len(ace2_records),
        "rbd_sequences": len(rbd_records),
        "train_records": len(train_records),
        "val_records": len(val_records),
        "masking": {
            "mlm_prob": cfg["MLM_PROB"],
            "span_min": cfg["SPAN_MIN"],
            "span_max": cfg["SPAN_MAX"],
        },
        "lora": {
            "r": cfg["LORA_R"],
            "alpha": cfg["LORA_ALPHA"],
            "dropout": cfg["LORA_DROP"],
            "targets": cfg["LORA_TARGETS"],
        },
        "training": {
            "epochs": cfg["EPOCHS"],
            "batch_size": cfg["BATCH"],
            "grad_accumulation": cfg["ACCUM"],
            "lr": cfg["LR"],
            "lr_scheduler": "cosine",
            "warmup_ratio": cfg["WARMUP_RATIO"],
        },
        "results": {
            "best_eval_loss": metrics.get("best_eval_loss"),
            "best_perplexity": metrics.get("best_perplexity"),
            "final_train_loss": metrics.get("final_train_loss"),
            "train_runtime_s": train_result.metrics.get("train_runtime"),
        },
        "artifacts": {
            "adapter_dir": adapter_dir,
            "merged_dir": merged_dir,
        },
    }
    card_path = os.path.join(out_dir, "training_card.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)
    logger.info(f"Training card saved: {card_path}")
    for handler in logger.handlers:
        handler.flush()
        handler.close()


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ESM-2 + LoRA masked language model fine-tuning on ACE2/RBD sequences.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ace2", type=str, required=True, help="Path to ACE2 FASTA file")
    parser.add_argument("--rbd", type=str, required=True, help="Path to RBD FASTA file")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--base-id", type=str, help="HuggingFace model identifier")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch", type=int, help="Per-device batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--lora-r", type=int, help="LoRA rank")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 training")

    # Utility: build merged RBD FASTA
    parser.add_argument(
        "--build-rbd-fasta",
        action="store_true",
        help="Merge two RBD FASTA sources and exit",
    )
    parser.add_argument("--dms-fasta", type=str, help="DMS source FASTA (for --build-rbd-fasta)")
    parser.add_argument("--label-fasta", type=str, help="Label source FASTA (for --build-rbd-fasta)")
    parser.add_argument("--out-fasta", type=str, help="Output path for merged FASTA")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.build_rbd_fasta:
        if not (args.dms_fasta and args.label_fasta and args.out_fasta):
            raise SystemExit(
                "missing argument of rbd fasta file"
            )
        build_merged_rbd_fasta(args.dms_fasta, args.label_fasta, args.out_fasta)
        raise SystemExit(0)

    cfg = dict(DEFAULT_CFG)
    cfg["ACE2_FASTA"] = args.ace2
    cfg["RBD_FASTA"] = args.rbd
    cfg["OUT_DIR"] = args.out

    if args.base_id:  cfg["BASE_ID"] = args.base_id
    if args.epochs:   cfg["EPOCHS"]  = args.epochs
    if args.batch:    cfg["BATCH"]   = args.batch
    if args.lr:       cfg["LR"]      = args.lr
    if args.lora_r:   cfg["LORA_R"]  = args.lora_r
    if args.seed:     cfg["SEED"]    = args.seed
    if args.no_fp16:  cfg["FP16"]    = False

    main(cfg)