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
