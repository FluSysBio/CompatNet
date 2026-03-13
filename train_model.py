#!/usr/bin/env python3
"""
Main Training Script for ACE2-RBD Binding Prediction
=====================================================
Full model training pipeline.
Split: 80% train / 10% val / 10% test
"""
import argparse
import csv
import time
import json
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch

from config import Config
from model import ACE2RBDBindingModel
from data_utils import DataSplitter, create_data_loaders, load_preprocessed_data
from training import CombinedLoss, SAM, get_cosine_schedule_with_warmup, train_epoch, evaluate
from utils import set_seed, setup_logger, format_time, count_parameters, get_device, save_checkpoint, EarlyStopping


# Helpers

def _log_metrics(logger, label: str, m: Dict) -> None:
    """Log regression + classification metrics in two compact lines."""
    logger.info(f"{label}")
    logger.info(f"  R²: {m['r2']:.4f} | RMSE: {m['rmse']:.4f} | MAE: {m['mae']:.4f}")
    logger.info(
        f"  Acc: {m['accuracy']:.4f} | "
        f"F1 (W/M): {m['f1_weighted']:.4f}/{m['f1_macro']:.4f} | "
        f"Prec (W/M): {m['precision_weighted']:.4f}/{m['precision_macro']:.4f} | "
        f"Recall (W/M): {m['recall_weighted']:.4f}/{m['recall_macro']:.4f}"
    )


def _history_row(stage: int, epoch: int, train_loss: float, val: Dict) -> Dict:
    """Build a compact training-history row."""
    return {
        'stage':                  stage,
        'epoch':                  epoch,
        'train_loss':             round(float(train_loss),              6),
        'val_r2':                 round(float(val['r2']),               4),
        'val_rmse':               round(float(val['rmse']),             4),
        'val_mae':                round(float(val['mae']),              4),
        'val_accuracy':           round(float(val['accuracy']),         4),
        'val_f1_weighted':        round(float(val['f1_weighted']),      4),
        'val_f1_macro':           round(float(val['f1_macro']),         4),
        'val_precision_weighted': round(float(val['precision_weighted']), 4),
        'val_precision_macro':    round(float(val['precision_macro']),  4),
        'val_recall_weighted':    round(float(val['recall_weighted']),  4),
        'val_recall_macro':       round(float(val['recall_macro']),     4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core training function
# ─────────────────────────────────────────────────────────────────────────────

def train_single_model(
    model: torch.nn.Module,
    loaders: Dict,
    config: Config,
    cls_counts: List[int],
    device: torch.device,
    logger,
    use_two_stage: bool = True,
) -> Dict:
    """
    Two-stage training loop.
    Returns best state_dict, metrics for train/val/test, per-epoch history,
    and model config.
    """
    cls_loss_fn = CombinedLoss(
        cls_counts=cls_counts,
        use_focal=config.training.use_focal_loss,
        use_ldam=config.training.use_ldam,
        class_weights=config.training.class_weights,
        focal_gamma=config.training.focal_gamma,
        ldam_max_m=config.training.ldam_max_m,
        ldam_s=config.training.ldam_s,
    )

    optimizer = SAM(
        model.parameters(),
        torch.optim.AdamW,
        rho=config.training.sam_rho,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    class_probs = 1.0 / np.array(cls_counts)
    class_probs /= class_probs.sum()

    best_val_loss = float('inf')
    best_state    = None
    best_metrics  = None
    train_history = []

    def _run_epoch(loader, scheduler):
        return train_epoch(
            model, loader, optimizer, scheduler, cls_loss_fn, device,
            grad_accum_steps=config.training.grad_accumulation,
            reg_weight=config.training.regression_weight,
            cls_weight=config.training.classification_weight,
            use_mixup=config.training.use_mixup,
            mixup_alpha=config.training.mixup_alpha,
            class_probs=class_probs if config.training.class_balanced_mixup else None,
            max_grad_norm=config.training.max_grad_norm,
        )

    # Stage 1: Balanced training 
    if use_two_stage and 'balanced' in loaders:
        logger.info("Stage 1: Balanced Training")
        loader       = loaders['balanced']
        total_steps  = config.training.stage1_epochs * len(loader) // config.training.grad_accumulation
        warmup_steps = config.training.warmup_epochs * len(loader) // config.training.grad_accumulation
        scheduler    = get_cosine_schedule_with_warmup(
            optimizer.base_optimizer, warmup_steps, total_steps, config.training.min_lr_ratio
        )
        early_stop = EarlyStopping(patience=config.training.patience // 2, min_delta=1e-6)

        for epoch in range(1, config.training.stage1_epochs + 1):
            train_m = _run_epoch(loader, scheduler)

            if epoch % 10 == 0 or epoch == 1:
                val_m    = evaluate(model, loaders['val'], device)
                val_loss = 0.6 * (1 - val_m['r2']) + 0.4 * (1 - val_m['accuracy'])
                logger.info(
                    f"[S1] Epoch {epoch:3d} | Loss: {train_m['loss']:.4f} | "
                    f"Val R²: {val_m['r2']:.4f} | Acc: {val_m['accuracy']:.4f} | "
                    f"F1(W): {val_m['f1_weighted']:.4f}"
                )
                train_history.append(_history_row(1, epoch, train_m['loss'], val_m))

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                if early_stop(val_loss):
                    logger.info(f"  Early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        logger.info("Stage 1 complete")

    # Stage 2: Full data training 
    logger.info("Stage 2: Full Data Training")
    stage2_epochs = config.training.stage2_epochs if use_two_stage else config.training.epochs
    loader        = loaders['train']
    total_steps   = stage2_epochs * len(loader) // config.training.grad_accumulation
    warmup_steps  = config.training.warmup_epochs * len(loader) // config.training.grad_accumulation
    scheduler     = get_cosine_schedule_with_warmup(
        optimizer.base_optimizer, warmup_steps, total_steps, config.training.min_lr_ratio
    )
    early_stop    = EarlyStopping(patience=config.training.patience, min_delta=1e-6)
    best_val_loss = float('inf')

    for epoch in range(1, stage2_epochs + 1):
        train_m = _run_epoch(loader, scheduler)

        if epoch % 15 == 0 or epoch == 1:
            val_m    = evaluate(model, loaders['val'], device)
            val_loss = 0.6 * (1 - val_m['r2']) + 0.4 * (1 - val_m['accuracy'])
            logger.info(
                f"[S2] Epoch {epoch:3d} | Loss: {train_m['loss']:.4f} | "
                f"Val R²: {val_m['r2']:.4f} | Acc: {val_m['accuracy']:.4f} | "
                f"F1(W): {val_m['f1_weighted']:.4f} | MAE: {val_m['mae']:.4f}"
            )
            train_history.append(_history_row(2, epoch, train_m['loss'], val_m))

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_metrics  = val_m.copy()

            if early_stop(val_loss):
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluations 
    train_final = evaluate(model, loaders['train'], device)
    test_final  = evaluate(model, loaders['test'],  device)

    logger.info("=" * 60)
    _log_metrics(logger, "Train (best checkpoint):", train_final)
    _log_metrics(logger, "Val   (best checkpoint):", best_metrics)
    _log_metrics(logger, "Test:",                    test_final)
    logger.info("=" * 60)

    return {
        'state_dict':    best_state,
        'train_metrics': train_final,
        'val_metrics':   best_metrics,
        'test_metrics':  test_final,
        'train_history': train_history,
        'model_config':  model.get_config(),
    }


# Main
def main(args):
    # Config 
    config = Config.from_yaml(args.config) if args.config else Config()
    if args.data:
        config.paths.input_json = args.data
    if args.output_dir:
        config.paths.output_dir  = args.output_dir
        config.paths.models_dir  = str(Path(args.output_dir) / "models")
        config.paths.results_dir = str(Path(args.output_dir) / "results")

    Path(config.paths.models_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.results_dir).mkdir(parents=True, exist_ok=True)

    device = get_device(config.device)
    logger = setup_logger("training", log_file=str(Path(config.paths.output_dir) / "training.log"))

    logger.info("=" * 80)
    logger.info("ACE2-RBD BINDING PREDICTION - TRAINING")
    logger.info("=" * 80)
    logger.info(f"Device: {device} | Folds: {config.training.num_folds} | Seeds: {config.training.num_seeds}")

    # Load data
    logger.info("\nLoading preprocessed data...")
    data = load_preprocessed_data(args.data)
    logger.info(f"  {len(data['sample_ids'])} samples | RBD: {data['rbd_embeddings'].shape} | ACE2: {data['ace2_embeddings'].shape}")

    # Normalization stats 
    if 'norm_rbd_mean' not in data:
        raise KeyError(
            "\n  norm_rbd_mean not found in preprocessed data!\n"
        )
    normalization_stats = {
        'rbd_mean':  data['norm_rbd_mean'].tolist(),
        'rbd_std':   data['norm_rbd_std'].tolist(),
        'ace2_mean': data['norm_ace2_mean'].tolist(),
        'ace2_std':  data['norm_ace2_std'].tolist(),
    }
    logger.info(f"  Norm — RBD mean/std: {data['norm_rbd_mean'].mean():.6f}/{data['norm_rbd_std'].mean():.6f} | "
                f"ACE2 mean/std: {data['norm_ace2_mean'].mean():.6f}/{data['norm_ace2_std'].mean():.6f}")

    # Training loop 
    all_results    = []
    all_histories  = []
    best_val_r2    = -float('inf')
    best_model_info = None

    n_samples  = len(data['sample_ids'])
    indices    = np.arange(n_samples)
    run_counter = 0
    total_runs  = config.training.num_folds * config.training.num_seeds
    overall_start = time.time()

    for fold in range(config.training.num_folds):
        for seed_idx in range(config.training.num_seeds):
            run_counter  += 1
            current_seed  = config.seed + seed_idx
            set_seed(current_seed)

            logger.info(f"\n{'='*60}")
            logger.info(f"RUN {run_counter}/{total_runs} | FOLD {fold+1} | SEED {current_seed}")
            logger.info(f"{'='*60}")

            # Splits: 80 / 10 / 10 
            if config.training.num_folds > 1:
                fold_size   = n_samples // config.training.num_folds
                test_start  = fold * fold_size
                test_end    = (fold + 1) * fold_size if fold < config.training.num_folds - 1 else n_samples
                test_idx    = indices[test_start:test_end]
                train_val_idx = np.concatenate([indices[:test_start], indices[test_end:]])
            else:
                train_val_idx, test_idx = DataSplitter.split_train_val(
                    indices, data['class_ids'], val_ratio=0.10, seed=current_seed
                )

            train_idx, val_idx = DataSplitter.split_train_val(
                train_val_idx, data['class_ids'],
                val_ratio=0.111,   # 10% of total (= 10/90)
                seed=current_seed,
            )
            logger.info(f"  Split — Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

            # Balanced subset for stage 1 
            balanced_idx = None
            if config.training.use_two_stage:
                balanced_idx = DataSplitter.create_balanced_subset(
                    train_idx, data['class_ids'], 'hybrid', current_seed
                )
                logger.info(f"  Balanced (stage 1): {len(balanced_idx)}")

            cls_counts = [int(np.sum(data['class_ids'][train_idx] == i))
                          for i in range(config.model.num_classes)]

            loaders = create_data_loaders(
                data, train_idx, val_idx, test_idx,
                batch_size=config.training.batch_size,
                eval_batch_size=config.training.eval_batch_size,
                balanced_idx=balanced_idx,
            )

            # Model 
            model = ACE2RBDBindingModel(
                embedding_dim=data['rbd_embeddings'].shape[-1],
                num_classes=config.model.num_classes,
                dropout=config.model.dropout,
                drop_path=config.model.drop_path,
                num_heads=config.model.num_attention_heads,
            ).to(device)

            if run_counter == 1:
                p = count_parameters(model)
                logger.info(f"  Params: {p['total']:,} total | {p['trainable']:,} trainable")

            # Train 
            t0     = time.time()
            result = train_single_model(
                model, loaders, config, cls_counts, device, logger,
                use_two_stage=config.training.use_two_stage,
            )
            result.update({
                'fold': fold, 'seed': current_seed, 'run': run_counter,
                'train_time_seconds': time.time() - t0,
                'normalization_stats': normalization_stats,
            })
            logger.info(f"  Done in {format_time(result['train_time_seconds'])} | "
                        f"Val R²: {result['val_metrics']['r2']:.4f} | "
                        f"Test R²: {result['test_metrics']['r2']:.4f}")

            # Save checkpoints 
            model_filename = f"model_fold{fold+1}_seed{current_seed}.pt"
            save_checkpoint(result, str(Path(config.paths.models_dir) / model_filename))

            val_r2 = result['val_metrics']['r2']
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_model_info = {
                    'fold': fold, 'seed': current_seed, 'run': run_counter,
                    'val_r2': val_r2, 'model_path': model_filename,
                }
                save_checkpoint(result, str(Path(config.paths.models_dir) / "best_model.pt"))
                logger.info(f"  ✓ New best model — Val R²: {val_r2:.4f}")

            all_results.append(result)
            for row in result['train_history']:
                row.update({'fold': fold, 'seed': current_seed, 'run': run_counter})
            all_histories.extend(result['train_history'])

    total_time = time.time() - overall_start

    # Aggregate results 
    logger.info(f"\n{'='*80}")
    logger.info(f"TRAINING COMPLETE — {format_time(total_time)}")
    logger.info(f"Best: Fold {best_model_info['fold']+1} | Seed {best_model_info['seed']} | Val R²: {best_model_info['val_r2']:.4f}")
    logger.info(f"{'='*80}")

    def _cv_stat(split: str, key: str) -> Dict:
        vals = [r[f"{split}_metrics"][key] for r in all_results]
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4)}

    metric_keys = [
        ("r2", "R²"), ("mae", "MAE"), ("rmse", "RMSE"),
        ("accuracy", "Accuracy"), ("f1_weighted", "F1(W)"), ("f1_macro", "F1(M)"),
    ]
    cv_stats = {
        f"{split}_{k}": _cv_stat(split, k)
        for split in ("val", "test")
        for k in [
            "r2", "mae", "rmse", "accuracy",
            "f1_weighted", "f1_macro",
            "precision_weighted", "precision_macro",
            "recall_weighted", "recall_macro",
        ]
    }

    logger.info(f"\nCV Results (n={len(all_results)}):")
    for k, label in metric_keys:
        v, t = cv_stats[f"val_{k}"], cv_stats[f"test_{k}"]
        logger.info(f"  {label:<12} Val: {v['mean']:.4f} ± {v['std']:.4f} | Test: {t['mean']:.4f} ± {t['std']:.4f}")

    # Save history CSV 
    if all_histories:
        history_csv = Path(config.paths.results_dir) / "train_history.csv"
        with open(history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_histories[0].keys()))
            writer.writeheader()
            writer.writerows(all_histories)
        logger.info(f"\nTraining history saved: {history_csv}")

    # Save summary JSON 
    import datetime
    summary = {
        "created_utc":        datetime.datetime.utcnow().isoformat() + "Z",
        "num_samples":        int(len(data.get("pkd_values", []))),
        "num_folds":          config.training.num_folds,
        "num_seeds":          config.training.num_seeds,
        "total_runs":         len(all_results),
        "total_time_seconds": float(total_time),
        "split":              "80/10/10 (train/val/test)",
        "best_model":         best_model_info,
        "cv_statistics":      cv_stats,
        "has_normalization_stats": True,
        "individual_results": [
            {
                "fold": r["fold"], "seed": r["seed"], "run": r["run"],
                "train": {k: round(float(v), 4) for k, v in r["train_metrics"].items()},
                "val":   {k: round(float(v), 4) for k, v in r["val_metrics"].items()},
                "test":  {k: round(float(v), 4) for k, v in r["test_metrics"].items()},
                "model_file": f"model_fold{r['fold']+1}_seed{r['seed']}.pt",
            }
            for r in all_results
        ],
    }
    summary_path = Path(config.paths.results_dir) / "model_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ACE2-RBD binding prediction model")
    parser.add_argument("--config",     type=str, help="Path to config YAML")
    parser.add_argument("--data",       type=str, required=True, help="Path to preprocessed .npz file")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    main(parser.parse_args())