import argparse
import torch
import time
import numpy as np
import random
from dataclasses import asdict

from archlab.model.baseline import TransformerLM
from archlab.trainer.train_model import *
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
    cfg = parse_args()
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    print(f"device: {device}")

    # 数据
    train_data = np.memmap(cfg.io.train_data_path, dtype=np.uint16, mode="r")
    val_data   = np.memmap(cfg.io.val_data_path,   dtype=np.uint16, mode="r")
    print(f"train tokens: {len(train_data):,}")
    print(f"val tokens:   {len(val_data):,}")

    # 模型 + 优化器
    model = build_model(cfg.model, device)
    optimizer = build_optimizer(model, cfg.optim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")

    # 训练循环
    model.train()
    t_log = time.time()

    for step in range(cfg.train.total_iters):
        # 1. LR schedule
        lr = get_lr_cosine_schedule(
            step,
            cfg.optim.max_lr,
            cfg.optim.min_lr,
            cfg.optim.warmup_iters,
            cfg.optim.cosine_cycle_iters,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 2. 采 batch
        x, y = get_batch(
            train_data,
            cfg.train.batch_size,
            cfg.model.context_length,
            str(device),
        )

        # 3. forward + backward
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        grad_norm = gradient_clipping(model.parameters(), cfg.optim.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # 4. log
        if step % cfg.train.log_interval == 0:
            now = time.time()
            dt = now - t_log
            tok_per_step = cfg.train.batch_size * cfg.model.context_length
            tok_per_sec = tok_per_step * cfg.train.log_interval / max(dt, 1e-6)
            print(
                f"step {step:5d}  "
                f"loss {loss.item():.4f}  "
                f"lr {lr:.2e}  "
                f"|g| {grad_norm:.3f}  "
                f"tok/s {tok_per_sec:.0f}"
            )
            t_log = now

if __name__ == "__main__":
    main()