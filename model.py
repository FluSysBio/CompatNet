#!/usr/bin/env python3
"""
Model Architecture for SARS-CoV2:ACE2-RBD Binding Prediction
===================================================
Multi-task model for predicting binding affinity (regression) and binding class (classification).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute mean pooling with masking for variable-length sequences
    
    Args:
        x: Input tensor of shape (batch, seq_len, dim)
        mask: Boolean mask of shape (batch, seq_len)
        
    Returns:
        Pooled tensor of shape (batch, dim)
    """
    w = mask.unsqueeze(-1).float()
    return (x * w).sum(1) / w.sum(1).clamp_min(1.0)


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample for residual blocks
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        
        return x.div(keep_prob) * random_tensor


class LayerScale(nn.Module):
    """
    Layer scaling for training stability
    """
    
    def __init__(self, dim: int, init_value: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class LocalCNN(nn.Module):
    """
    Local context modeling using CNNs with residual connections
    """
    
    def __init__(self, 
                 dim: int, 
                 kernel_size: int = 3, 
                 dropout: float = 0.15, 
                 drop_path: float = 0.0):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim * 2, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(8, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * 2, dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(1, dim),
        )
        
        self.drop_path = DropPath(drop_path)
        self.scale = LayerScale(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            (batch, seq_len, dim)
        """
        # Conv expects (batch, dim, seq_len)
        identity = x
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        return identity + self.drop_path(self.scale(x))


class GlobalAttn(nn.Module):
    """
    Global context modeling using multi-head attention with depth-wise separable convolutions
    """
    
    def __init__(self, 
                 dim: int, 
                 kernel_size: int = 9,
                 num_heads: int = 12, 
                 dropout: float = 0.15, 
                 drop_path: float = 0.0):
        super().__init__()
        
        # Depth-wise separable convolution for position encoding
        self.dw_conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.pw_conv = nn.Conv1d(dim, dim, 1)
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)
        self.scale = LayerScale(dim)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len) boolean mask
        Returns:
            (batch, seq_len, dim)
        """
        identity = x
        
        # Apply depth-wise separable conv for position information
        z = x.transpose(1, 2)
        z = self.dw_conv(z)
        z = self.pw_conv(z)
        z = z.transpose(1, 2)
        
        # Multi-head attention with masking
        attn_out, _ = self.attn(z, z, z, key_padding_mask=~mask)
        
        # Residual connection with drop path and layer scale
        out = identity + self.drop_path(self.scale(self.dropout(attn_out)))
        return self.norm(out)


class RefinementModule(nn.Module):
    """
    Refinement module combining local and global context.
    """
    
    def __init__(self, dim: int, dropout: float = 0.15, drop_path: float = 0.0):
        super().__init__()
        
        self.local = LocalCNN(dim, dropout=dropout, drop_path=drop_path)
        self.global_attn = GlobalAttn(dim, dropout=dropout, drop_path=drop_path)
        
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len) boolean mask
        Returns:
            (batch, seq_len, dim)
        """
        local_out = self.local(x)
        global_out = self.global_attn(x, mask)
        
        # Fuse local and global features
        fused = self.fusion(torch.cat([local_out, global_out], dim=-1))
        
        return fused


class CrossAttention(nn.Module):
    """
    Cross-attention module for modeling interactions between RBD and ACE2.
    """
    
    def __init__(self, 
                 dim: int, 
                 num_heads: int = 12, 
                 dropout: float = 0.15, 
                 drop_path: float = 0.0):
        super().__init__()
        
        self.attn1 = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn2 = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # Gating mechanism
        self.gate1 = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.gate2 = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)
        
        self.scale1 = LayerScale(dim)
        self.scale2 = LayerScale(dim)
    
    def forward(self, 
                x1: torch.Tensor, 
                x2: torch.Tensor, 
                mask1: torch.Tensor, 
                mask2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x1: (batch, seq_len1, dim) - RBD features
            x2: (batch, seq_len2, dim) - ACE2 features
            mask1: (batch, seq_len1) - RBD mask
            mask2: (batch, seq_len2) - ACE2 mask
        Returns:
            Updated (x1, x2)
        """
        # Cross-attention: x1 attends to x2
        attn1, _ = self.attn1(x1, x2, x2, key_padding_mask=~mask2)
        gate1 = self.gate1(torch.cat([x1, attn1], dim=-1))
        x1 = self.norm1(x1 + self.drop_path(self.scale1(self.dropout(attn1 * gate1))))
        
        # Cross-attention: x2 attends to x1
        attn2, _ = self.attn2(x2, x1, x1, key_padding_mask=~mask1)
        gate2 = self.gate2(torch.cat([x2, attn2], dim=-1))
        x2 = self.norm2(x2 + self.drop_path(self.scale2(self.dropout(attn2 * gate2))))
        
        return x1, x2


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction head for regression (pKd) and classification (binding class).
    """
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.15):
        super().__init__()
        
        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.25),
            
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        # Regression head (pKd prediction)
        self.regression_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        
        # Classification head (binding class prediction)
        self.classification_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            (regression_output, classification_logits)
        """
        shared_repr = self.shared(x)
        
        # Regression output (pKd)
        reg_out = self.regression_head(shared_repr).squeeze(-1)
        
        # Classification logits
        cls_out = self.classification_head(shared_repr)
        
        return reg_out, cls_out


class ACE2RBDBindingModel(nn.Module):
    """
    Complete multi-task model for ACE2-RBD binding prediction.
    
    Architecture:
        1. Refinement modules for RBD and ACE2 sequences
        2. Multi-level cross-attention between RBD and ACE2
        3. Multi-task prediction head for pKd and binding class
    """
    
    def __init__(self, 
                 embedding_dim: int = 480, 
                 num_classes: int = 4,
                 dropout: float = 0.15,
                 drop_path: float = 0.15,
                 num_heads: int = 12):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Refinement modules
        self.rbd_refine = RefinementModule(embedding_dim, dropout=0.2, drop_path=drop_path)
        self.ace2_refine = RefinementModule(embedding_dim, dropout=0.2, drop_path=drop_path)
        
        # Multi-level cross-attention
        self.cross_attn1 = CrossAttention(embedding_dim, num_heads, 0.15, drop_path * 0.5)
        self.cross_attn2 = CrossAttention(embedding_dim, num_heads, 0.15, drop_path)
        self.cross_attn3 = CrossAttention(embedding_dim, num_heads, 0.15, drop_path * 1.5)
        
        # Multi-task head
        self.head = MultiTaskHead(embedding_dim * 2, num_classes, dropout)
    
    def forward(self, 
                rbd_emb: torch.Tensor, 
                ace2_emb: torch.Tensor, 
                rbd_mask: torch.Tensor, 
                ace2_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            rbd_emb: (batch, rbd_len, dim) - RBD embeddings
            ace2_emb: (batch, ace2_len, dim) - ACE2 embeddings
            rbd_mask: (batch, rbd_len) - RBD mask
            ace2_mask: (batch, ace2_len) - ACE2 mask
            
        Returns:
            (pkd_pred, class_logits)
                - pkd_pred: (batch,) predicted pKd values
                - class_logits: (batch, num_classes) classification logits
        """
        # Refine individual sequences
        rbd = self.rbd_refine(rbd_emb, rbd_mask)
        ace2 = self.ace2_refine(ace2_emb, ace2_mask)
        
        # Multi-level cross-attention
        rbd, ace2 = self.cross_attn1(rbd, ace2, rbd_mask, ace2_mask)
        rbd, ace2 = self.cross_attn2(rbd, ace2, rbd_mask, ace2_mask)
        rbd, ace2 = self.cross_attn3(rbd, ace2, rbd_mask, ace2_mask)
        
        # Pool sequences
        rbd_pooled = masked_mean(rbd, rbd_mask)
        ace2_pooled = masked_mean(ace2, ace2_mask)
        
        # Concatenate features
        combined = torch.cat([rbd_pooled, ace2_pooled], dim=1)
        
        # Multi-task predictions
        pkd_pred, class_logits = self.head(combined)
        
        return pkd_pred, class_logits
    
    def get_config(self) -> dict:
        """Get model configuration for saving"""
        return {
            'embedding_dim': self.embedding_dim,
            'num_classes': self.num_classes,
        }
    
    @classmethod
    def from_config(cls, config: dict) -> 'ACE2RBDBindingModel':
        """Create model from configuration"""
        return cls(**config)