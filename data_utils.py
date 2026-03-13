#!/usr/bin/env python3
"""
Data Utilities for SARS-CoV2:ACE2-RBD Binding Prediction Project
===============================================
Dataset classes, data loaders, and splitting strategies.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold


def load_preprocessed_data(data_path: str) -> Dict[str, np.ndarray]:
    """
    Load preprocessed data from .npz file.
    
    Args:
        data_path: Path to .npz file
        
    Returns:
        Dictionary containing embeddings, masks, labels, and metadata
    """
    data = np.load(data_path, allow_pickle=True)
    data_dict = {}
    
    # Required fields for training
    required_fields = [
        'rbd_embeddings',
        'ace2_embeddings', 
        'rbd_masks',
        'ace2_masks',
        'pkd_values',
        'class_ids'
    ]
    optional_fields = [
        'sample_ids', 
        'rbd_sequences', 
        'ace2_sequences',
        'norm_rbd_mean',
        'norm_rbd_std',
        'norm_ace2_mean',
        'norm_ace2_std'
    ]
    for field in required_fields:
        if field in data:
            data_dict[field] = data[field]
        else:
            raise ValueError(f"Required field '{field}' not found in {data_path}")
    for field in optional_fields:
        if field in data:
            data_dict[field] = data[field]
    print(f"  Available fields: {list(data_dict.keys())}")
    
    return data_dict


class ACE2RBDDataset(Dataset):
    """
    PyTorch Dataset for ACE2-RBD binding data.
    """
    
    def __init__(self, 
                 data: Dict[str, np.ndarray], 
                 indices: np.ndarray):
        """
        Args:
            data: Dictionary containing embeddings, masks, and labels
            indices: Indices of samples to include in this dataset
        """
        self.data = data
        self.indices = np.array(indices)
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.
        
        Returns:
            (rbd_emb, ace2_emb, rbd_mask, ace2_mask, pkd, class_id)
        """
        sample_idx = self.indices[idx]
        
        return (
            torch.from_numpy(self.data['rbd_embeddings'][sample_idx]),
            torch.from_numpy(self.data['ace2_embeddings'][sample_idx]),
            torch.from_numpy(self.data['rbd_masks'][sample_idx]),
            torch.from_numpy(self.data['ace2_masks'][sample_idx]),
            torch.tensor(self.data['pkd_values'][sample_idx], dtype=torch.float32),
            torch.tensor(self.data['class_ids'][sample_idx], dtype=torch.long),
        )


def collate_batch(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched tensors
    """
    rbd_emb, ace2_emb, rbd_mask, ace2_mask, pkd, class_id = zip(*batch)
    
    return (
        torch.stack(rbd_emb),
        torch.stack(ace2_emb),
        torch.stack(rbd_mask),
        torch.stack(ace2_mask),
        torch.stack(pkd),
        torch.stack(class_id)
    )


class DataSplitter:
    """
    Create train/val/test splits with stratification.
    """
    
    @staticmethod
    def create_folds(class_ids: np.ndarray,
                    pkd_values: np.ndarray,
                    n_folds: int = 3,
                    seed: int = 42) -> List[np.ndarray]:
        """
        Create stratified k-fold splits based on class distribution.
        
        Args:
            class_ids: Array of class IDs
            pkd_values: Array of pKd values (for reporting)
            n_folds: Number of folds
            seed: Random seed
            
        Returns:
            List of test indices for each fold
        """
        n_samples = len(class_ids)
        indices = np.arange(n_samples)
        
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        folds = []
        
        print(f"\nCreating {n_folds}-fold stratified splits...")

        
        for fold_idx, (_, test_idx) in enumerate(skf.split(indices, class_ids)):
            folds.append(test_idx)
            fold_class_ids = class_ids[test_idx]
            fold_pkd = pkd_values[test_idx]
            
            print(f"Fold {fold_idx + 1}:")
            print(f"  pKd: {fold_pkd.mean():.3f} ± {fold_pkd.std():.3f}")
            
            # Class distribution
            unique_classes = np.unique(class_ids)
        
        # Check balance
        fold_means = [pkd_values[f].mean() for f in folds]
        mean_range = max(fold_means) - min(fold_means)
        
        print(f"  pKd mean range: {min(fold_means):.3f} - {max(fold_means):.3f}")

        
        return folds
    
    @staticmethod
    def split_train_val(train_indices: np.ndarray,
                       class_ids: np.ndarray,
                       val_ratio: float = 0.2,
                       seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split training indices into train and validation sets.
        
        Args:
            train_indices: Indices for training pool
            class_ids: Array of class IDs
            val_ratio: Ratio of validation data
            seed: Random seed
            
        Returns:
            (train_idx, val_idx)
        """
        n_val = max(1, int(val_ratio * len(train_indices)))
        
        rng = np.random.default_rng(seed)
        shuffled = train_indices.copy()
        rng.shuffle(shuffled)
        
        val_idx = shuffled[:n_val]
        train_idx = shuffled[n_val:]
        
        return train_idx, val_idx
    
    @staticmethod
    def create_balanced_subset(train_indices: np.ndarray,
                              class_ids: np.ndarray,
                              strategy: str = 'hybrid',
                              seed: int = 42) -> np.ndarray:
        """
        Create balanced subset for two-stage training.
        
        Strategies:
            - 'undersample': Sample to minority class size
            - 'oversample': Sample to majority class size
            - 'hybrid': Sample to median class size
        
        Args:
            train_indices: Training indices
            class_ids: Array of class IDs
            strategy: Balancing strategy
            seed: Random seed
            
        Returns:
            Balanced indices
        """
        rng = np.random.default_rng(seed)
        
        # Get indices for each class
        unique_classes = np.unique(class_ids[train_indices])
        class_indices = {
            cls_id: train_indices[class_ids[train_indices] == cls_id]
            for cls_id in unique_classes
        }
        
        class_counts = {cls_id: len(idx) for cls_id, idx in class_indices.items()}
        
        print(f"\nCreating balanced subset (strategy: {strategy}):")
        print(f"  Original counts: {class_counts}")
        
        # Determine target size
        if strategy == 'undersample':
            target_size = min(class_counts.values())
        elif strategy == 'oversample':
            target_size = max(class_counts.values())
        elif strategy == 'hybrid':
            target_size = int(np.median(list(class_counts.values())))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Sample from each class
        balanced_indices = []
        for cls_id, idx in class_indices.items():
            if len(idx) > target_size:
                # Undersample
                sampled = rng.choice(idx, size=target_size, replace=False)
            else:
                # Oversample
                sampled = rng.choice(idx, size=target_size, replace=True)
            
            balanced_indices.extend(sampled)
        
        # Shuffle
        balanced_indices = np.array(balanced_indices)
        rng.shuffle(balanced_indices)
        
        # Report
        balanced_counts = {
            cls_id: np.sum(class_ids[balanced_indices] == cls_id)
            for cls_id in unique_classes
        }
        
        print(f"  Balanced counts: {balanced_counts}")
        print(f"  Total: {len(train_indices)} → {len(balanced_indices)}")
        
        return balanced_indices


def create_data_loaders(data: Dict[str, np.ndarray],
                       train_idx: np.ndarray,
                       val_idx: np.ndarray,
                       test_idx: np.ndarray,
                       batch_size: int = 8,
                       eval_batch_size: int = 32,
                       num_workers: int = 0,
                       balanced_idx: Optional[np.ndarray] = None) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data: Preprocessed data dictionary
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices
        batch_size: Training batch size
        eval_batch_size: Evaluation batch size
        num_workers: Number of workers for data loading
        balanced_idx: balanced indices for stage 1 training
        
    Returns:
        Dictionary of data loaders
    """
    loaders = {}
    
    # Training loader
    train_dataset = ACE2RBDDataset(data, train_idx)
    loaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Balanced training loader (for two-stage training)
    if balanced_idx is not None:
        balanced_dataset = ACE2RBDDataset(data, balanced_idx)
        loaders['balanced'] = DataLoader(
            balanced_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    # Validation loader
    val_dataset = ACE2RBDDataset(data, val_idx)
    loaders['val'] = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Test loader
    test_dataset = ACE2RBDDataset(data, test_idx)
    loaders['test'] = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return loaders