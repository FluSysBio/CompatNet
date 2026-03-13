#!/usr/bin/env python3
"""
ACE2-RBD Binding Prediction - Inference
========================================
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from model import ACE2RBDBindingModel
from preprocessing import SequenceEmbedder, DataLoader as DataPreprocessor
from data_utils import ACE2RBDDataset, collate_batch
from torch.utils.data import DataLoader
from utils import set_seed, setup_logger, pkd_to_kd_nm
_COMBINED_KEYS = ["best_eval_model", "best_train_model"]


class InferenceEngine:
    """Loads a trained model from a checkpoint and runs predictions.
    Args:
        model_path: Path to the .pt checkpoint file.
        device:     'cuda' or 'cpu'.
        seed:       Random seed for reproducibility.
        sub_model:  Sub-model key to extract from a combined checkpoint.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        seed: int = 42,
        sub_model: Optional[str] = None,
    ):
        set_seed(seed)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if sub_model is not None:
            if sub_model not in checkpoint:
                raise ValueError(
                    f"\n  Sub-model '{sub_model}' not found in checkpoint.\n"
                    f"  Available keys: {list(checkpoint.keys())}"
                )
            print(f"  Using explicitly requested sub-model: '{sub_model}'")
            checkpoint = checkpoint[sub_model]

        elif any(k in checkpoint for k in _COMBINED_KEYS):
            chosen = next(k for k in _COMBINED_KEYS if k in checkpoint)
            print(f"  Combined checkpoint detected — auto-selecting '{chosen}'")
            print(f"  (use --sub-model to override; available: "
                  f"{[k for k in _COMBINED_KEYS if k in checkpoint]})")
            checkpoint = checkpoint[chosen]
        model_config = checkpoint.get("model_config", {"embedding_dim": 480, "num_classes": 4})

        self.model = ACE2RBDBindingModel(**model_config).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        self.test_metrics = checkpoint.get("test_metrics", {})
        norm = checkpoint.get("normalization_stats")
        if norm is None:
            raise ValueError(
                "\n  normalization_stats not found in checkpoint!\n"
            )

        self.rbd_mean  = np.array(norm["rbd_mean"],  dtype=np.float32)
        self.rbd_std   = np.array(norm["rbd_std"],   dtype=np.float32)
        self.ace2_mean = np.array(norm["ace2_mean"], dtype=np.float32)
        self.ace2_std  = np.array(norm["ace2_std"],  dtype=np.float32)

    @torch.no_grad()
    def predict(
        self,
        rbd_embeddings: np.ndarray,
        ace2_embeddings: np.ndarray,
        rbd_masks: np.ndarray,
        ace2_masks: np.ndarray,
        batch_size: int = 32,
    ) -> Dict[str, np.ndarray]:

        data = {
            "rbd_embeddings":  rbd_embeddings,
            "ace2_embeddings": ace2_embeddings,
            "rbd_masks":       rbd_masks,
            "ace2_masks":      ace2_masks,
            "pkd_values":      np.zeros(len(rbd_embeddings), dtype=np.float32),
            "class_ids":       np.zeros(len(rbd_embeddings), dtype=np.int64),
        }

        dataset = ACE2RBDDataset(data, np.arange(len(rbd_embeddings)))
        loader  = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_batch, num_workers=0,
        )

        all_pkd = []

        for rbd, ace2, rbd_mask, ace2_mask, _, _ in loader:
            rbd       = rbd.to(self.device)
            ace2      = ace2.to(self.device)
            rbd_mask  = rbd_mask.to(self.device)
            ace2_mask = ace2_mask.to(self.device)

            pkd_pred, _ = self.model(rbd, ace2, rbd_mask, ace2_mask)
            all_pkd.append(pkd_pred.cpu().numpy())

        return {"pkd": np.concatenate(all_pkd)}


def run_inference(
    model_path: str,
    input_json: str,
    output_csv: str,
    esm_model: str = "esm2_t12_35M_UR50D",
    max_rbd_len: int = 350,
    max_ace2_len: int = 900,
    batch_size: int = 32,
    device: str = "cuda",
    seed: int = 42,
    sub_model: Optional[str] = None,
) -> None:

    log_dir = Path("./outputs/inference_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"inference_{timestamp}.log"

    logger = setup_logger("inference", log_file=str(log_file))

    logger.info("=" * 80)
    logger.info("ACE2-RBD BINDING PREDICTION - INFERENCE")
    logger.info("=" * 80)
    logger.info(f"Log file:   {log_file}")
    logger.info(f"Model:      {model_path}")
    logger.info(f"Sub-model:  {sub_model if sub_model else 'auto-detect'}")
    logger.info(f"Input:      {input_json}")
    logger.info(f"Output:     {output_csv}")
    logger.info(f"ESM model:  {esm_model}")
    logger.info(f"Device:     {device}")

    # Load model
    logger.info("\nLoading model checkpoint...")
    engine = InferenceEngine(model_path, device, seed, sub_model=sub_model)

    if engine.test_metrics:
        logger.info("  Checkpoint test metrics:")
        for key, val in engine.test_metrics.items():
            logger.info(f"    {key}: {val:.4f}")

    # Load test sequences
    logger.info("\nLoading test data...")
    samples     = DataPreprocessor.load_json(input_json)
    data_parsed = DataPreprocessor.parse_samples(samples)
    n_test      = len(data_parsed["sample_ids"])
    logger.info(f"  Loaded {n_test} test samples")

    # Generate ESM-2 embeddings
    logger.info(f"\nGenerating ESM-2 embeddings...")
    embedder = SequenceEmbedder(esm_model, device)

    rbd_records  = list(zip(data_parsed["sample_ids"], data_parsed["rbd_sequences"]))
    rbd_list     = embedder.embed_sequences(rbd_records, max_rbd_len, 2)

    ace2_records = list(zip(data_parsed["sample_ids"], data_parsed["ace2_sequences"]))
    ace2_list    = embedder.embed_sequences(ace2_records, max_ace2_len, 2)

    rbd_embeddings,  rbd_masks  = SequenceEmbedder.pad_only(rbd_list)
    ace2_embeddings, ace2_masks = SequenceEmbedder.pad_only(ace2_list)

    rbd_embeddings  = (rbd_embeddings  - engine.rbd_mean)  / engine.rbd_std
    ace2_embeddings = (ace2_embeddings - engine.ace2_mean) / engine.ace2_std

    # Run predictions
    logger.info(f"\nRunning predictions on {n_test} samples...")
    predictions = engine.predict(
        rbd_embeddings, ace2_embeddings, rbd_masks, ace2_masks, batch_size
    )

    kd_values = pkd_to_kd_nm(predictions["pkd"])

    lineages, species = [], []
    for i, sample_id in enumerate(data_parsed["sample_ids"]):
        if data_parsed["metadata"] and i < len(data_parsed["metadata"]):
            lineages.append(data_parsed["metadata"][i].get("lineage", sample_id))
            species.append(data_parsed["metadata"][i].get("species", "Unknown"))
        else:
            lineages.append(sample_id)
            species.append("Unknown")

    results_df = pd.DataFrame({
        "Sample_ID":       [f"{l}_{s}" for l, s in zip(lineages, species)],
        "Lineage":         lineages,
        "ACE2_Species":    species,
        "Predicted_Kd_nM": kd_values,
    })

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    logger.info("\n" + "=" * 80)
    logger.info("INFERENCE COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total predictions : {len(results_df)}")
    logger.info(f"Kd range          : {kd_values.min():.2f} – {kd_values.max():.2f} nM  "
                f"(mean {kd_values.mean():.2f} nM)")
    logger.info(f"\nResults saved : {output_csv}")
    logger.info(f"Log saved     : {log_file}")
    logger.info("=" * 80)


def main(args) -> None:
    run_inference(
        model_path   = args.model,
        input_json   = args.input,
        output_csv   = args.output,
        esm_model    = args.esm_model,
        max_rbd_len  = args.max_rbd_len,
        max_ace2_len = args.max_ace2_len,
        batch_size   = args.batch_size,
        device       = args.device,
        seed         = args.seed,
        sub_model    = args.sub_model,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ACE2-RBD binding prediction inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",        type=str, required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--input",        type=str, required=True,
                        help="Path to input JSON file")
    parser.add_argument("--output",       type=str, default="predictions.csv",
                        help="Path to output CSV file")
    parser.add_argument("--sub-model",    type=str, default=None,
                        choices=["best_eval_model", "best_train_model"],
                        help="Sub-model key to extract from a combined checkpoint. "
                             "If omitted, auto-detection is used (prefers best_eval_model).")
    parser.add_argument("--max-rbd-len",  type=int, default=350)
    parser.add_argument("--max-ace2-len", type=int, default=900)
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--seed",         type=int, default=42)

    args = parser.parse_args()
    main(args)