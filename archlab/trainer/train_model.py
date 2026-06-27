import os
import torch
import numpy as np
import math
import typing
from torch import Tensor
from jaxtyping import Float, Int
from typing import Optional, Union
from collections.abc import Callable, Iterable

def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."]
) -> Tensor:
    logit_max = logits.amax(dim=-1, keepdim=True)
    z = logits - logit_max
    lse = logit_max.squeeze(-1) + torch.log(torch.exp(z).sum(-1))

    tgt_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    log_prob = lse - tgt_logit

    return log_prob.mean()

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t+1
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        defaults = {"lr":lr, "weight_decay":weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
            
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                grad = p.grad
                if len(state) == 0:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                m, v = state["m"], state["v"]
                t = state["t"]
                state["t"] += 1

                # learning rate 
                step_size = lr * math.sqrt(1 - beta2**t) / (1- beta1**t)
                # weight decay
                p.mul_(1 - lr * wd)
                # update the monument estimate
                m.mul_(beta1).add_(grad, alpha = 1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)

                # update params
                denom = v.sqrt().add(eps)
                p.addcdiv_(m, denom, value=-step_size)
        
        return loss
    
def get_lr_cosine_schedule(t: int, lr_max: float, lr_min: float, warmup: int, cosine: int):
    if t < warmup:
        return t / warmup * lr_max
    elif t >= warmup and t <= cosine:
        return lr_min + 0.5 * (1 + math.cos((t - warmup)/(cosine - warmup)*math.pi)) * (lr_max - lr_min)
    else:
        return lr_min
    
def gradient_clipping(
    parameters: Union[torch.Tensor, Iterable[torch.Tensor]],
    max_norm: float,
    eps: float = 1e-6
) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    grads = [p.grad for p in parameters if p.grad is not None]

    device = grads[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), p=2).to(device) for g in grads]),
        p = 2,
    )

    clip_coef = max_norm / (total_norm + eps)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    for g in grads:
        g.mul_(clip_coef_clamped)

    return total_norm

def get_batch(
    dataset: Union[list[int], np.ndarray, torch.Tensor],
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(dataset, torch.Tensor):
        data = dataset.cpu().numpy()
    else: 
        data = np.asarray(dataset)
    
    n = len(data)
    assert n >= context_length + 1

    starts = np.random.randint(0, n - context_length, size = batch_size)

    # slice 
    x = np.stack([data[s: s + context_length] for s in starts]).astype(np.int64)
    y = np.stack([data[s+1: s + context_length + 1] for s in starts]).astype(np.int64)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    if torch.device(device).type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    
    return x, y

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    obj = torch.load(src, weights_only=False, map_location="cpu")
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]