#!/usr/bin/env python3
"""
Preprocessing Module for SARS-CoV2:ACE2-RBD Binding Prediction Project
=====================================================
Data loading, embedding generation, and preprocessing.
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    print("Warning: ESM not available")
_ESM_NAMES = {
    "esm2_t12_35M_UR50D",
}


class SequenceEmbedder:
    """
    Generate ESM2 embeddings for protein sequences.

    Supports both esm2 model (auto-detected from model_name):
      1. fair-esm [480 dimension base esm2 model]       — "esm2_t12_35M_UR50D"
      2. HuggingFace(Method using stage1: Lora adapted Fine tuned ESM2 model)    — "runs/stage1/merged_backbone"
    """

    def __init__(self,
                 model_name: str = "esm2_t12_35M_UR50D",
                 device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._use_hf = (model_name not in _ESM_NAMES)

        if self._use_hf:
            self._load_hf(model_name)
        else:
            self._load_esm(model_name)

        print(f" Model loaded on {self.device}")
        print(f"  Embedding dimension: {self.embed_dim}")

    def _load_hf(self, model_path: str):
        print(f"Loading fine-tuned ESM-2 backbone: {model_path}")
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
        self.hf_model     = AutoModel.from_pretrained(model_path).to(self.device)
        self.hf_model.eval()
        self.embed_dim = self.hf_model.config.hidden_size
        print(f"  Layers : {self.hf_model.config.num_hidden_layers}")

    def _load_esm(self, model_name: str):
        if not ESM_AVAILABLE:
            raise ImportError("ESM not available")
        print(f"Loading ESM2 model: {model_name}")
        self.model, self.alphabet = getattr(esm.pretrained, model_name)()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.repr_layer      = self.model.num_layers
        self.pad_idx         = self.alphabet.padding_idx
        self.embed_dim       = self.model.embed_dim
        print(f"  Number of layers: {self.repr_layer}")

    @torch.no_grad()
    def embed_sequences(self,
                        sequences: List[Tuple[str, str]],
                        max_length: int = 1024,
                        batch_size: int = 2) -> List[np.ndarray]:
        if self._use_hf:
            return self._embed_hf(sequences, max_length, batch_size)
        else:
            return self._embed_esm(sequences, max_length, batch_size)

    @torch.no_grad()
    def _embed_hf(self, sequences, max_length, batch_size):
        embeddings = []
        total = len(sequences)
        print(f"Generating embeddings for {total} sequences (fine-tuned HF backend)")

        for start in tqdm(range(0, total, batch_size), desc="Embedding"):
            batch = sequences[start:start + batch_size]
            seqs  = [seq[:max_length] for _, seq in batch]

            enc = self.hf_tokenizer(
                seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length + 2,
                add_special_tokens=True,
            )
            enc  = {k: v.to(self.device) for k, v in enc.items()}
            out  = self.hf_model(**enc)
            hidden = out.last_hidden_state
            attn   = enc["attention_mask"]

            for b in range(len(batch)):
                valid = attn[b].bool().nonzero(as_tuple=False).flatten()
                if len(valid) > 2:
                    emb = hidden[b, valid[1]:valid[-1], :]
                    embeddings.append(emb.cpu().float().numpy().astype(np.float32))
                else:
                    embeddings.append(np.zeros((0, self.embed_dim), dtype=np.float32))

        return embeddings

    @torch.no_grad()
    def _embed_esm(self, sequences, max_length, batch_size):
        embeddings = []
        total = len(sequences)
        print(f"Generating embeddings for {total} sequences...")

        for start in tqdm(range(0, total, batch_size), desc="Embedding"):
            batch  = sequences[start:start + batch_size]
            batch  = [(seq_id, seq[:max_length]) for seq_id, seq in batch]
            _, _, tokens = self.batch_converter(batch)
            tokens = tokens.to(self.device)
            output          = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)
            representations = output["representations"][self.repr_layer]
            tokens_cpu = tokens.cpu().numpy()

            for b in range(len(batch)):
                valid_mask = tokens_cpu[b] != self.pad_idx
                positions  = np.where(valid_mask)[0]
                if len(positions) > 1:
                    emb = representations[b, positions[0]+1:positions[-1], :]
                    embeddings.append(emb.cpu().float().numpy().astype(np.float32))
                else:
                    embeddings.append(np.zeros((0, representations.shape[-1]), dtype=np.float32))

        return embeddings

    @staticmethod
    def pad_only(embeddings: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(embeddings)
        embed_dim = embeddings[0].shape[1] if n_samples > 0 else 0
        max_len   = max(emb.shape[0] for emb in embeddings)
        padded = np.zeros((n_samples, max_len, embed_dim), dtype=np.float32)
        masks  = np.zeros((n_samples, max_len), dtype=bool)
        for i, emb in enumerate(embeddings):
            seq_len = emb.shape[0]
            if seq_len > 0:
                padded[i, :seq_len] = emb
                masks[i,  :seq_len] = True
        return padded, masks

    @staticmethod
    def pad_and_standardize(embeddings: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        n_samples = len(embeddings)
        embed_dim = embeddings[0].shape[1] if n_samples > 0 else 0
        max_len   = max(emb.shape[0] for emb in embeddings)
        padded = np.zeros((n_samples, max_len, embed_dim), dtype=np.float32)
        masks  = np.zeros((n_samples, max_len), dtype=bool)
        for i, emb in enumerate(embeddings):
            seq_len = emb.shape[0]
            if seq_len > 0:
                padded[i, :seq_len] = emb
                masks[i, :seq_len]  = True
        mask_float = masks.astype(np.float32)[..., None]
        mean = (padded * mask_float).sum(axis=(0, 1)) / mask_float.sum()
        var  = ((padded - mean) ** 2 * mask_float).sum(axis=(0, 1)) / mask_float.sum()
        std  = np.sqrt(var + 1e-6)
        padded = (padded - mean) / std
        return padded, masks


class DataLoader:
    @staticmethod
    def load_json(json_path: str) -> List[Dict[str, Any]]:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            samples = data
        elif isinstance(data, dict):
            samples = data.get('data', data.get('samples', data.get('items', [])))
        else:
            raise ValueError(f"Unsupported JSON structure")
        return samples

    @staticmethod
    def parse_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        data = {
            'sample_ids':    [],
            'rbd_sequences': [],
            'ace2_sequences':[],
            'pkd_values':    [],
            'class_ids':     [],
            'metadata':      []
        }
        for sample in samples:
            data['sample_ids'].append(sample.get('Sample_ID', ''))
            data['rbd_sequences'].append(sample.get('RBD_Sequence', ''))
            data['ace2_sequences'].append(sample.get('ACE2_sequence', ''))
            pkd = sample.get('pKd', sample.get('pkd', 0.0))
            data['pkd_values'].append(float(pkd))
            class_id = sample.get('Class_ID', sample.get('class_id', 0))
            data['class_ids'].append(int(class_id))
            metadata = {
                'species':       sample.get('species',      sample.get('Species',      'Unknown')),
                'lineage':       sample.get('lineage',       sample.get('variant',      'Unknown')),
                'binding_class': sample.get('Binding_Class', sample.get('binding_class','Unknown'))
            }
            data['metadata'].append(metadata)
        data['pkd_values'] = np.array(data['pkd_values'], dtype=np.float32)
        data['class_ids']  = np.array(data['class_ids'],  dtype=np.int64)
        return data

    @staticmethod
    def print_data_summary(data: Dict[str, Any], class_names: List[str]):
        n_samples = len(data['sample_ids'])
        print(f"\nData Summary:")
        print(f"  Total samples: {n_samples}")
        print(f"  Unique RBD sequences:  {len(set(data['rbd_sequences']))}")
        print(f"  Unique ACE2 sequences: {len(set(data['ace2_sequences']))}")
        print(f"\npKd statistics:")
        print(f"  Mean: {data['pkd_values'].mean():.3f}")
        print(f"  Std:  {data['pkd_values'].std():.3f}")
        print(f"  Min:  {data['pkd_values'].min():.3f}")
        print(f"  Max:  {data['pkd_values'].max():.3f}")
        print(f"\nClass distribution:")
        for i, name in enumerate(class_names):
            count = np.sum(data['class_ids'] == i)
            pct   = 100 * count / n_samples
            print(f"  {name:12s}: {count:4d} ({pct:5.1f}%)")
        species_list = [m['species'] for m in data['metadata']]
        lineage_list = [m['lineage'] for m in data['metadata']]
        print(f"\nDiversity:")
        print(f"  Unique species:  {len(set(species_list))}")
        print(f"  Unique lineages: {len(set(lineage_list))}")


def preprocess_and_save(json_path: str,
                        output_path: str,
                        model_name: str = "esm2_t12_35M_UR50D",
                        max_rbd_len: int = 350,
                        max_ace2_len: int = 900,
                        batch_size: int = 2,
                        device: str = "cuda",
                        class_names: List[str] = None):
    if class_names is None:
        class_names = ["strong", "medium", "weak", "no_bind"]

    print("="*80)
    print(" ACE2-RBD BINDING PREDICTION - PREPROCESSING")
    print("="*80)

    print("\nStep 1: Loading data...")
    samples = DataLoader.load_json(json_path)
    data    = DataLoader.parse_samples(samples)
    DataLoader.print_data_summary(data, class_names)

    print(f"\nStep 2: Generating ESM2 embeddings...")
    embedder = SequenceEmbedder(model_name, device)

    print("\n  Generating RBD embeddings...")
    rbd_records    = list(zip(data['sample_ids'], data['rbd_sequences']))
    rbd_embeddings = embedder.embed_sequences(rbd_records, max_rbd_len, batch_size)

    print("\n  Generating ACE2 embeddings...")
    ace2_records    = list(zip(data['sample_ids'], data['ace2_sequences']))
    ace2_embeddings = embedder.embed_sequences(ace2_records, max_ace2_len, batch_size)

    print("\nStep 3: Padding and standardizing embeddings...")
    rbd_padded,  rbd_masks  = SequenceEmbedder.pad_only(rbd_embeddings)
    ace2_padded, ace2_masks = SequenceEmbedder.pad_only(ace2_embeddings)

    rbd_mf    = rbd_masks.astype(np.float32)[..., None]
    rbd_mean  = (rbd_padded  * rbd_mf).sum(axis=(0, 1)) / rbd_mf.sum()
    rbd_var   = ((rbd_padded  - rbd_mean) ** 2 * rbd_mf).sum(axis=(0, 1)) / rbd_mf.sum()
    rbd_std   = np.sqrt(rbd_var + 1e-6)

    ace2_mf   = ace2_masks.astype(np.float32)[..., None]
    ace2_mean = (ace2_padded * ace2_mf).sum(axis=(0, 1)) / ace2_mf.sum()
    ace2_var  = ((ace2_padded - ace2_mean) ** 2 * ace2_mf).sum(axis=(0, 1)) / ace2_mf.sum()
    ace2_std  = np.sqrt(ace2_var + 1e-6)

    print(f"  Normalization stats computed from RAW embeddings:")
    print(f"    RBD  mean: {rbd_mean.mean():.6f}, std: {rbd_std.mean():.6f}")
    print(f"    ACE2 mean: {ace2_mean.mean():.6f}, std: {ace2_std.mean():.6f}")

    rbd_padded  = (rbd_padded  - rbd_mean) / rbd_std
    ace2_padded = (ace2_padded - ace2_mean) / ace2_std

    print(f"  RBD  embeddings shape: {rbd_padded.shape}")
    print(f"  ACE2 embeddings shape: {ace2_padded.shape}")

    print(f"\nStep 4: Saving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        sample_ids     = np.array(data['sample_ids'],    dtype=object),
        rbd_embeddings = rbd_padded,
        ace2_embeddings= ace2_padded,
        rbd_masks      = rbd_masks,
        ace2_masks     = ace2_masks,
        pkd_values     = data['pkd_values'],
        class_ids      = data['class_ids'],
        rbd_sequences  = np.array(data['rbd_sequences'],  dtype=object),
        ace2_sequences = np.array(data['ace2_sequences'], dtype=object),
        metadata       = np.array(data['metadata'],       dtype=object),
        norm_rbd_mean  = rbd_mean.astype(np.float32),
        norm_rbd_std   = rbd_std.astype(np.float32),
        norm_ace2_mean = ace2_mean.astype(np.float32),
        norm_ace2_std  = ace2_std.astype(np.float32),
    )

    print(f" Preprocessing complete!")
    print(f"  Output saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def load_preprocessed_data(npz_path: str) -> Dict[str, np.ndarray]:
    data = dict(np.load(npz_path, allow_pickle=True))
    data['rbd_masks']  = data['rbd_masks'].astype(bool)
    data['ace2_masks'] = data['ace2_masks'].astype(bool)
    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess ACE2-RBD binding data")
    parser.add_argument("--input",        type=str, required=True,
                        help="Input JSON file")
    parser.add_argument("--output",       type=str, required=True,
                        help="Output .npz file")
    parser.add_argument("--model",        type=str, default="esm2_t12_35M_UR50D",
                        help="ESM2 model name OR path to merged_backbone directory")
    parser.add_argument("--max-rbd-len",  type=int, default=350,
                        help="Maximum RBD length")
    parser.add_argument("--max-ace2-len", type=int, default=900,
                        help="Maximum ACE2 length")
    parser.add_argument("--batch-size",   type=int, default=2,
                        help="Batch size for embedding generation")
    parser.add_argument("--device",       type=str, default="cuda",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()

    preprocess_and_save(
        json_path    = args.input,
        output_path  = args.output,
        model_name   = args.model,
        max_rbd_len  = args.max_rbd_len,
        max_ace2_len = args.max_ace2_len,
        batch_size   = args.batch_size,
        device       = args.device,
    )
