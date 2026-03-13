#!/usr/bin/env python3
"""
Training Module for SARS-CoV2:ACE2-RBD Binding Prediction
================================================
Loss functions, optimizers, schedulers, and training loops.
"""

import math
from typing import Tuple, Dict, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
)


# =========================
# LOSS FUNCTIONS
# =========================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, 
                 alpha: Optional[torch.Tensor] = None, 
                 gamma: float = 2.0):
        """
        Args:
            alpha: Class weights (tensor of shape [num_classes])
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            targets: (batch,) class indices
        """
        alpha = self.alpha.to(logits.device) if self.alpha is not None else None
        
        ce_loss = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss for long-tailed recognition.
    """
    
    def __init__(self, 
                 cls_num_list: List[int], 
                 max_m: float = 0.5, 
                 s: float = 30):
        """
        Args:
            cls_num_list: Number of samples per class
            max_m: Maximum margin
            s: Scale parameter
        """
        super().__init__()
        
        # Compute per-class margins
        m_list = 1.0 / np.sqrt(np.sqrt(np.array(cls_num_list)))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.FloatTensor(m_list)
        self.s = s
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            labels: (batch,) class indices
        """
        self.m_list = self.m_list.to(logits.device)
        
        # Create index mask
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, labels.unsqueeze(1), 1)
        
        # Apply class-specific margins
        batch_m = self.m_list[labels].unsqueeze(1)
        logits_m = logits - batch_m * index.float()
        
        return F.cross_entropy(self.s * logits_m, labels)


class CombinedLoss(nn.Module):
    """
    Combined loss function for classification.
    """
    
    def __init__(self, 
                 cls_counts: List[int],
                 use_focal: bool = True,
                 use_ldam: bool = True,
                 class_weights: Optional[List[float]] = None,
                 focal_gamma: float = 2.0,
                 ldam_max_m: float = 0.5,
                 ldam_s: float = 30):
        """
        Args:
            cls_counts: Number of samples per class
            use_focal: Use focal loss
            use_ldam: Use LDAM loss
            class_weights: Optional custom class weights
            focal_gamma: Focal loss gamma parameter
            ldam_max_m: LDAM maximum margin
            ldam_s: LDAM scale parameter
        """
        super().__init__()
        self.use_focal = use_focal
        self.use_ldam = use_ldam
        
        # Compute class weights
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
        else:
            total = sum(cls_counts)
            self.class_weights = torch.FloatTensor([
                total / (len(cls_counts) * c) for c in cls_counts
            ])
        
        # Initialize loss functions
        if use_focal:
            self.focal = FocalLoss(alpha=self.class_weights, gamma=focal_gamma)
        
        if use_ldam:
            self.ldam = LDAMLoss(cls_counts, max_m=ldam_max_m, s=ldam_s)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            labels: (batch,) class indices
        """
        if self.use_focal and self.use_ldam:
            return 0.5 * self.focal(logits, labels) + 0.5 * self.ldam(logits, labels)
        elif self.use_focal:
            return self.focal(logits, labels)
        elif self.use_ldam:
            return self.ldam(logits, labels)
        else:
            return F.cross_entropy(logits, labels, 
                                 weight=self.class_weights.to(logits.device))


def focal_mse_loss(pred: torch.Tensor, 
                   target: torch.Tensor, 
                   gamma: float = 2.0) -> torch.Tensor:
    """
    Focal MSE loss for regression - focuses on hard examples.
    
    Args:
        pred: Predicted values
        target: Target values
        gamma: Focusing parameter
    """
    error = (pred - target).abs()
    weights = (error + 1e-8).pow(gamma)
    weights = weights / weights.mean().clamp_min(1e-8)
    
    return (weights * (pred - target).pow(2)).mean()


# =========================
# AUGMENTATION
# =========================

def mixup_data(rbd: torch.Tensor,
              ace2: torch.Tensor,
              rbd_mask: torch.Tensor,
              ace2_mask: torch.Tensor,
              y_reg: torch.Tensor,
              y_cls: torch.Tensor,
              alpha: float = 0.4,
              class_probs: Optional[np.ndarray] = None) -> Tuple:
    """
    Mixup augmentation with optional class-balanced sampling.
    
    Args:
        rbd, ace2: Embeddings
        rbd_mask, ace2_mask: Masks
        y_reg: Regression targets
        y_cls: Classification targets
        alpha: Mixup alpha parameter
        class_probs: Optional class probabilities for sampling
        
    Returns:
        Mixed data and targets
    """
    if alpha <= 0:
        return rbd, ace2, rbd_mask, ace2_mask, y_reg, y_cls, None, None, 1.0
    
    batch_size = rbd.size(0)
    lam = np.random.beta(alpha, alpha)
    
    # Class-balanced sampling
    if class_probs is not None:
        weights = torch.tensor([class_probs[c.item()] for c in y_cls], 
                              device=rbd.device)
        weights = weights / weights.sum()
        indices = torch.multinomial(weights, batch_size, replacement=True)
    else:
        indices = torch.randperm(batch_size, device=rbd.device)
    
    rbd_mixed = lam * rbd + (1 - lam) * rbd[indices]
    ace2_mixed = lam * ace2 + (1 - lam) * ace2[indices]
    rbd_mask_mixed = rbd_mask | rbd_mask[indices]
    ace2_mask_mixed = ace2_mask | ace2_mask[indices]
    y_reg_mixed = lam * y_reg + (1 - lam) * y_reg[indices]
    y_cls_a = y_cls
    y_cls_b = y_cls[indices]
    
    return (rbd_mixed, ace2_mixed, rbd_mask_mixed, ace2_mask_mixed, 
            y_reg_mixed, y_cls_a, y_cls_b, lam)


# =========================
# SAM OPTIMIZER
# =========================

class SAM(Optimizer):
    """
    Sharpness Aware Minimization (SAM) optimizer wrapper.
    """
    
    def __init__(self, 
                 params, 
                 base_optimizer: type, 
                 rho: float = 0.05, 
                 **kwargs):
        """
        Args:
            params: Model parameters
            base_optimizer: Base optimizer class (e.g., torch.optim.AdamW)
            rho: SAM neighborhood size
            **kwargs: Arguments for base optimizer
        """
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        First step of SAM: compute and apply adversarial perturbation.
        """
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Save current parameters
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        Second step of SAM: revert perturbation and update parameters.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                # Revert to original parameters
                p.data = self.state[p]["old_p"]
        
        # Update with base optimizer
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    def step(self, closure=None):
        """
        Standard step method (calls first_step and second_step).
        """
        raise NotImplementedError(
            "SAM doesn't support step()"
        )
    
    def _grad_norm(self) -> torch.Tensor:
        """
        Compute gradient norm across all parameters.
        """
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2)
                for group in self.param_groups
                for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm


# =========================
# SCHEDULER
# =========================

def get_cosine_schedule_with_warmup(optimizer: Optimizer,
                                   num_warmup_steps: int,
                                   num_training_steps: int,
                                   min_lr_ratio: float = 0.01) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Cosine learning rate schedule with warmup.
    
    Args:
        optimizer: PyTorch optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr_ratio: Minimum learning rate ratio
        
    Returns:
        LR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =========================
# TRAINING FUNCTIONS
# =========================

@torch.no_grad()
def evaluate(model: nn.Module,
            loader: torch.utils.data.DataLoader,
            device: torch.device, 
            return_outputs: bool = False):
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        loader: Data loader
        device: Device to run on
        return_outputs: If True, return (metrics, outputs) tuple with predictions
        
    Returns:
        If return_outputs=False: Dictionary of metrics
        If return_outputs=True: (metrics_dict, outputs_dict) tuple
    """
    model.eval()
    
    all_pkd_true, all_pkd_pred = [], []
    all_cls_true, all_cls_pred = [], []
    
    for rbd, ace2, rbd_mask, ace2_mask, pkd, cls_id in loader:
        rbd, ace2 = rbd.to(device), ace2.to(device)
        rbd_mask, ace2_mask = rbd_mask.to(device), ace2_mask.to(device)
        
        # Forward pass
        pkd_pred, cls_logits = model(rbd, ace2, rbd_mask, ace2_mask)
        cls_pred = cls_logits.argmax(dim=1)
        
        # Collect predictions
        all_pkd_true.append(pkd.numpy())
        all_pkd_pred.append(pkd_pred.cpu().numpy())
        all_cls_true.append(cls_id.numpy())
        all_cls_pred.append(cls_pred.cpu().numpy())
    
    # Concatenate
    pkd_true = np.concatenate(all_pkd_true)
    pkd_pred = np.concatenate(all_pkd_pred)
    cls_true = np.concatenate(all_cls_true)
    cls_pred = np.concatenate(all_cls_pred)
    
    # Regression metrics
    mae    = np.mean(np.abs(pkd_true - pkd_pred))
    rmse   = np.sqrt(np.mean((pkd_true - pkd_pred) ** 2))
    ss_res = np.sum((pkd_true - pkd_pred) ** 2)
    ss_tot = np.sum((pkd_true - pkd_true.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Classification metrics
    accuracy = float((cls_true == cls_pred).mean())

    shared = dict(y_true=cls_true, y_pred=cls_pred, zero_division=0)
    f1_weighted        = float(f1_score(        average="weighted", **shared))
    f1_macro           = float(f1_score(        average="macro",    **shared))
    precision_weighted = float(precision_score( average="weighted", **shared))
    precision_macro    = float(precision_score( average="macro",    **shared))
    recall_weighted    = float(recall_score(    average="weighted", **shared))
    recall_macro       = float(recall_score(    average="macro",    **shared))

    metrics = {
        # Regression
        "r2":                 float(r2),
        "mae":                float(mae),
        "rmse":               float(rmse),
        # Classification
        "accuracy":           accuracy,
        "f1_weighted":        f1_weighted,
        "f1_macro":           f1_macro,
        "precision_weighted": precision_weighted,
        "precision_macro":    precision_macro,
        "recall_weighted":    recall_weighted,
        "recall_macro":       recall_macro,
    }

    if return_outputs:
        outputs = {
            "y_true_reg": pkd_true,
            "y_pred_reg": pkd_pred,
            "y_true_cls": cls_true,
            "y_pred_cls": cls_pred,
        }
        return metrics, outputs

    return metrics


def train_epoch(model: nn.Module,
               loader: torch.utils.data.DataLoader,
               optimizer: Optimizer,
               scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
               cls_loss_fn: nn.Module,
               device: torch.device,
               grad_accum_steps: int = 2,
               reg_weight: float = 1.5,
               cls_weight: float = 1.0,
               use_mixup: bool = False,
               mixup_alpha: float = 0.4,
               class_probs: Optional[np.ndarray] = None,
               max_grad_norm: float = 1.0) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        loader: Data loader
        optimizer: Optimizer (SAM or regular)
        scheduler: Optional LR scheduler
        cls_loss_fn: Classification loss function
        device: Device to run on
        grad_accum_steps: Gradient accumulation steps
        reg_weight: Regression loss weight
        cls_weight: Classification loss weight
        use_mixup: Use mixup augmentation
        mixup_alpha: Mixup alpha parameter
        class_probs: Class probabilities for balanced mixup
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    total_loss = 0.0
    total_reg_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0
    
    is_sam = isinstance(optimizer, SAM)
    
    for batch_idx, (rbd, ace2, rbd_mask, ace2_mask, pkd, cls_id) in enumerate(loader):
        rbd, ace2 = rbd.to(device), ace2.to(device)
        rbd_mask, ace2_mask = rbd_mask.to(device), ace2_mask.to(device)
        pkd, cls_id = pkd.to(device), cls_id.to(device)
        
        # Apply mixup
        if use_mixup:
            (rbd, ace2, rbd_mask, ace2_mask, pkd, 
             cls_a, cls_b, lam) = mixup_data(
                rbd, ace2, rbd_mask, ace2_mask, pkd, cls_id,
                mixup_alpha, class_probs
            )
        else:
            cls_a, cls_b, lam = cls_id, None, 1.0
        
        # Forward pass
        pkd_pred, cls_logits = model(rbd, ace2, rbd_mask, ace2_mask)
        
        # Compute losses
        loss_reg = focal_mse_loss(pkd_pred, pkd)
        
        if cls_b is not None:
            loss_cls = (lam * cls_loss_fn(cls_logits, cls_a) + 
                       (1 - lam) * cls_loss_fn(cls_logits, cls_b))
        else:
            loss_cls = cls_loss_fn(cls_logits, cls_a)
        
        loss = reg_weight * loss_reg + cls_weight * loss_cls
        loss = loss / grad_accum_steps
        
        # Backward pass
        loss.backward()
        
        # Accumulate gradients
        if (batch_idx + 1) % grad_accum_steps == 0:
            if is_sam:
                # SAM first step
                optimizer.first_step(zero_grad=True)
                
                # Second forward-backward pass
                pkd_pred, cls_logits = model(rbd, ace2, rbd_mask, ace2_mask)
                loss_reg = focal_mse_loss(pkd_pred, pkd)
                
                if cls_b is not None:
                    loss_cls = (lam * cls_loss_fn(cls_logits, cls_a) + 
                               (1 - lam) * cls_loss_fn(cls_logits, cls_b))
                else:
                    loss_cls = cls_loss_fn(cls_logits, cls_a)
                
                loss = reg_weight * loss_reg + cls_weight * loss_cls
                loss = loss / grad_accum_steps
                loss.backward()
                
                # Gradient clipping and SAM second step
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.second_step(zero_grad=True)
            else:
                # Regular optimizer
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
        
        # Track losses
        total_loss += loss.item() * grad_accum_steps
        total_reg_loss += loss_reg.item()
        total_cls_loss += loss_cls.item()
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'reg_loss': total_reg_loss / num_batches,
        'cls_loss': total_cls_loss / num_batches,
    }