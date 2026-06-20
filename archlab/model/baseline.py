import math
import torch
from torch import Tensor
import torch.nn as nn

from einops import einsum
from jaxtyping import Int, Float

# Linear
class Linear(nn.Module):
    def __init__(
        self, in_features: int,
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, std=std, mean=0.0, a=-3*std, b=3*std)

    def forward(
        self, x: Float[Tensor, "... in_dim"]
    ) -> Float[Tensor, "... out_dim"]:
        return einsum(x, self.weight, "... in_dim, out_dim in_dim -> ... out_dim")
    
# Embedding
class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))

        torch.nn.init.trunc_normal_(self.weight, std=1, mean=0.0, a=-3, b=3)

    def forward(
        self, token_ids: Int[Tensor, "..."]
    ) -> Float[Tensor, "... d_model"]:
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        eps: float = 1e-5,
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(
        self, x: Float[Tensor, "... d_model"]
    ) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms * self.weight).to(in_dtype)

# FFN(x) = w2(SiLU(w1x) * w3x)
class PositionwiseFFN(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_ff: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    
    def forward(
        self, x: Float[Tensor, "batch_size seq_len d_model"]
    ) -> Float[Tensor, "batch_size seq_len d_model"]:
        a = self.w1(x)
        silu = a * torch.sigmoid(a)
        return self.w2(silu * self.w3(x))
    

