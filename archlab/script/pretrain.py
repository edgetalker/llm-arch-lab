import math
import torch
import time
import numpy as np
import random
import wandb
from pathlib import Path

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

@torch.no_grad()
def evaluate(model, val_data, cfg, device) -> float:
    model.eval()
    losses = torch.zeros(cfg.train.eval_iters, device=device)
    for k in range(cfg.train.eval_iters):
        x, y = get_batch(
            val_data,
            cfg.train.batch_size,
            cfg.model.context_length,
            str(device)
        )
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses[k] = loss
    model.train()
    return losses.mean().item()   

def save_with_retention(model, optimizer, step, cfg):
    ckpt_dir = Path(cfg.io.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    path = ckpt_dir / f"step_{step:07d}.pt"
    save_checkpoint(model, optimizer, step, path)
    
    # 滚动删除
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-cfg.io.keep_last_k]:
        old.unlink()
    
    # 更新 latest
    latest = ckpt_dir / "latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(path.name)


def maybe_resume(model, optimizer, cfg):
    if not cfg.io.resume_from:
        return 0
    return load_checkpoint(cfg.io.resume_from, model, optimizer)

def init_wandb(cfg):
    if cfg.io.no_wandb:
        return None
    return wandb.init(
        project=cfg.io.wandb_project,
        name=cfg.io.wandb_run_name,
        config=asdict(cfg),
        resume="allow" if cfg.io.resume_from else None,
    )

def main():
    cfg = parse_args()
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    print(f"device: {device}")
    wandb = init_wandb(cfg) 

    # 数据
    train_data = np.memmap(cfg.io.train_data_path, dtype=np.uint16, mode="r")
    val_data   = np.memmap(cfg.io.val_data_path,   dtype=np.uint16, mode="r")
    print(f"train tokens: {len(train_data):,}")
    print(f"val tokens:   {len(val_data):,}")

    # 模型 + 优化器
    model = build_model(cfg.model, device)
    optimizer = build_optimizer(model, cfg.optim)
    start_step = maybe_resume(model, optimizer, cfg)
    if start_step > 0:
        print(f"resumed from step {start_step}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")

    # 训练循环
    model.train()
    t_log = time.time()

    for step in range(start_step, cfg.train.total_iters):
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
            metrics = {
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/grad_norm": grad_norm,
                "train/tok_per_sec": tok_per_sec,
            }
            
            if wandb is not None:
                wandb.log(metrics, step=step)
            
            print(
                f"step {step:5d}  loss {loss.item():.4f}  lr {lr:.2e}  "
                f"|g| {grad_norm:.3f}  tok/s {tok_per_sec:.0f}"
            )
            t_log = now

        # 5. eval
        if step > 0 and step % cfg.train.eval_interval == 0:
            val_loss = evaluate(model, val_data, cfg, device)
            val_ppl = math.exp(val_loss)
            
            if wandb is not None:
                wandb.log({"val/loss": val_loss, "val/ppl": val_ppl}, step=step)
            
            print(f"step {step:5d}  ├─ val_loss {val_loss:.4f}  val_ppl {val_ppl:.2f}")
        
        # 6. ckp
        if step > 0 and step % cfg.train.ckpt_interval == 0:
            save_with_retention(model, optimizer, step, cfg)

    save_with_retention(model, optimizer, cfg.train.total_iters, cfg)

    if wandb is not None:
        wandb.finish()
        
if __name__ == "__main__":
    main()