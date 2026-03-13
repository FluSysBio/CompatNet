#!/usr/bin/env python3
"""
Utility Functions for SARS-CoV2:ACE2-RBD Binding Prediction
"""

import os
import random
import logging
from typing import Optional, Dict, Any
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Random seeds are set for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
    torch.use_deterministic_algorithms(True, warn_only=True)
    import warnings
    warnings.filterwarnings('ignore', message='.*cumsum_cuda_kernel.*')


def setup_logger(name: str = "ace2_rbd", 
                 log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level 
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
    
    return logger


def print_section(title: str, char: str = "=", width: int = 80):
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        char: Character to use for border
        width: Width of the border
    """
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}\n")


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count trainable and total parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    
    return {
        'trainable': trainable,
        'total': total,
        'non_trainable': total - trainable
    }


def get_device(device_str: str = "cuda") -> torch.device:
    """
    Get PyTorch device with fallback.
    
    Args:
        device_str: Requested device ("cuda" or "cpu")
        
    Returns:
        torch.device object
    """
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device


def save_checkpoint(state: Dict[str, Any], 
                   filepath: str,
                   is_best: bool = False):
    """
    Save model checkpoint.
    
    Args:
        state: Dictionary containing model state and metadata
        filepath: Path to save checkpoint
        is_best: If True, also save as best model
    """
    torch.save(state, filepath)
    
    if is_best:
        best_path = filepath.replace('.pt', '_best.pt')
        torch.save(state, best_path)


def load_checkpoint(filepath: str, 
                   device: torch.device) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        device: Device to load checkpoint to
        
    Returns:
        Checkpoint dictionary
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=device)
    return checkpoint


def pkd_to_kd_nm(pkd: np.ndarray) -> np.ndarray:
    """
    Convert pKd to Kd in nanomolar (nM).
    
    Args:
        pkd: pKd values (array or scalar)
        
    Returns:
        Kd values in nM
    """
    return np.power(10, 9 - pkd)


def kd_nm_to_pkd(kd_nm: np.ndarray) -> np.ndarray:
    """
    Convert Kd in nanomolar (nM) to pKd.
    
    Formula: pKd = 9 - log10(Kd[nM])
    
    Args:
        kd_nm: Kd values in nM (array or scalar)
    Returns:
        pKd values
    """
    kd_nm = np.maximum(kd_nm, 1e-10)
    return 9.0 - np.log10(kd_nm)


class AverageMeter:
    """
    Computes and stores the average and current value will be 
    useful for tracking metrics during training.
    """
    
    def __init__(self, name: str = "", fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class EarlyStopping:
    """
    Early stopping handler to stop training when validation loss stops improving.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for minimizing metric, 'max' for maximizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False
    
    def reset(self):
        """Reset early stopping state"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0

