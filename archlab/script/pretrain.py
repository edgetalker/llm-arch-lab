import argparse
import torch
import numpy as np
import random
from dataclasses import asdict

from archlab.model.baseline import TransformerLM
from archlab.trainer.train_model import AdamW, get_batch, cross_entropy, gradient_clipping
from archlab.trainer.config_loader import parse_args


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
    else: 
        return torch.device(device)

def build_model(cfg, device):
    return TransformerLM(**asdict(cfg)).to(device)

def build_optimizer(model, cfg):
    return AdamW(
        model.parameters(),
        lr=cfg.max_lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )


def evaluate(model, val_data, cfg, device) -> float:
    pass  # TODO

def main():
    # 1. 解析配置 + 准备输出目录
    cfg = parse_args()
    # TODO: mkdir, dump_config

    # 2. 全局初始化
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    # 3. 数据
    train_data = np.memmap(cfg.io.train_data_path, dtype=np.uint16, mode="r")
    
    # 4. 模型 + 优化器
    model = build_model(cfg.model, device)
    optimizer = build_optimizer(model, cfg.optim)


    x, y = get_batch(train_data, cfg.train.batch_size, cfg.model.context_length, str(device))
    print("x.shape:", x.shape, "y.shape:", y.shape, "x.dtype:", x.dtype)
    
    logits = model(x)
    print("logits.shape:", logits.shape)
    
    loss = cross_entropy(logits, y)
    print(f"initial loss: {loss.item():.4f}")
    
    loss.backward()
    grad_norm = gradient_clipping(model.parameters(), cfg.optim.grad_clip)
    print(f"grad_norm: {grad_norm:.4f}")
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    print("step 1 done")

    
    # 5. resume（可选）
    start_step = 0
    # TODO

    # 6. wandb（可选）
    # TODO

    # 7. 主循环
    for step in range(start_step, cfg.train.total_iters):
        pass  # TODO: train_step + log + eval + ckpt

    # 8. 收尾存档
    # TODO


if __name__ == "__main__":
    main()