from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    theta: float = 10000.0

@dataclass
class OptimConfig:
    # AdamW
    lr: float
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01
    # Cosine schedule
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 2000
    cosine_cycle_iters: int = 100000
    # Clip
    grad_clip: float = 1.0

@dataclass
class TrainConfig:
    batch_size: int
    total_iters: int
    seed: int
    device: str = "auto"
    dtype: str = "float32"
    
    log_interval: int = 10
    eval_interval: int = 500
    eval_iters: int = 100
    ckpt_interval: int = 1000

@dataclass
class IOConfig:
    train_data_path: str
    val_data_path: str
    ckpt_dir: str
    resume_from: Optional[str] = None
    keep_last_k: int = 3

@dataclass
class Config:
    """顶层 config，CLI 解析后填进来。"""
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    io: IOConfig