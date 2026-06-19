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
    

