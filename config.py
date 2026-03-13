#!/usr/bin/env python3
"""
Configuration Management for SARS-CoV2:ACE2-RBD Binding Prediction Project
=========================================================
Centralized configuration for all hyperparameters and paths.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import json
import yaml


@dataclass
class PathConfig:
    """Paths configuration"""
    # Input data
    input_json: str = ""
    
    # Output directories
    output_dir: str = "./outputs"
    embeddings_dir: str = "./outputs/embeddings"
    models_dir: str = "./outputs/models"
    results_dir: str = "./outputs/results"
    plots_dir: str = "./outputs/plots"
    
    def __post_init__(self):
        """Create output directories"""
        for dir_path in [self.output_dir, self.embeddings_dir, self.models_dir, 
                         self.results_dir, self.plots_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    esm2_model: str = "esm2_t12_35M_UR50D"
    max_rbd_len: int = 350
    max_ace2_len: int = 900
    
    # Architecture hyperparameters
    embedding_dim: int = 480  # ESM2-35M output dimension
    num_classes: int = 4
    dropout: float = 0.15
    drop_path: float = 0.15
    num_attention_heads: int = 12
    
    # Class names
    class_names: List[str] = field(default_factory=lambda: 
                                   ["strong", "medium", "weak", "no_bind"])
    class_name_map: dict = field(default_factory=lambda: 
                                 {0: "strong", 1: "medium", 2: "weak", 3: "no_bind"})


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    train_split: float = 0.8  # 80% train
    val_split: float = 0.10   # 10% validation
    test_split: float = 0.10  # 10% test
    
    num_folds: int = 1 
    num_seeds: int = 1
    
    epochs: int = 400
    patience: int = 35
    
    # Optimizer parameters
    learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    batch_size: int = 8
    eval_batch_size: int = 32
    grad_accumulation: int = 2
    
    # Scheduler parameters
    warmup_epochs: int = 15
    min_lr_ratio: float = 0.01
    
    # SAM parameters
    sam_rho: float = 0.05
    
    # Two-stage training
    use_two_stage: bool = True
    stage1_epochs: int = 100
    stage2_epochs: int = 300
    
    # Loss function parameters
    use_class_weights: bool = True
    use_focal_loss: bool = True
    use_ldam: bool = True
    focal_gamma: float = 2.0
    ldam_max_m: float = 0.5
    ldam_s: int = 30
    
    # Custom class weights (optional)
    class_weights: Optional[List[float]] = field(default_factory=lambda: 
                                                  [3.0, 2.5, 4.0, 0.5])
    
    # Augmentation parameters
    use_mixup: bool = True
    mixup_alpha: float = 0.4
    class_balanced_mixup: bool = True
    
    # Loss weights
    regression_weight: float = 1.5
    classification_weight: float = 1.0
    
    # Gradient clipping
    max_grad_norm: float = 1.0


@dataclass
class InferenceConfig:
    """Inference configuration"""
    model_path: str = ""
    test_data_path: str = ""
    output_csv: str = "./outputs/predictions.csv"
    batch_size: int = 32
    auto_select_best: bool = True  # ADDED: For backward compatibility


@dataclass
class Config:
    """Master configuration"""
    # Reproducibility
    seed: int = 42
    device: str = "cuda"  # "cuda" or "cpu"
    
    # Sub-configurations
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'cuda'),
            paths=PathConfig(**config_dict.get('paths', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            inference=InferenceConfig(**config_dict.get('inference', {}))
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> 'Config':
        """Load configuration from JSON file"""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'cuda'),
            paths=PathConfig(**config_dict.get('paths', {})),
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            inference=InferenceConfig(**config_dict.get('inference', {}))
        )
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'seed': self.seed,
            'device': self.device,
            'paths': self.paths.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        config_dict = {
            'seed': self.seed,
            'device': self.device,
            'paths': self.paths.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'inference': self.inference.__dict__
        }
        
        with open(json_path, 'w') as f:
            json.dump(config_dict, f, indent=2)


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


if __name__ == "__main__":
    # Example: Create and save default config
    config = get_default_config()
    config.to_yaml("default_config.yaml")
    config.to_json("default_config.json")
    print("Default configuration saved!")
