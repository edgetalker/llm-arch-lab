import math
import torch
from torch import Tensor
import torch.nn as nn
import warnings

import einx
from einops import einsum, rearrange
from jaxtyping import Int, Float, Bool

from archlab.tokenizer.bpe_tokenizer import Tokenizer
import torch.utils.checkpoint as cp 

# Linear
class Linear(nn.Module):
    def __init__(
        self, 
        in_features: int,
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        std = math.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(self.weight, std=std, mean=0.0, a=-3*std, b=3*std)

    def forward(self, x: Float[Tensor, "... in_dim"]) -> Float[Tensor, "... out_dim"]:
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

        # torch.nn.init.trunc_normal_(self.weight, std=1, mean=0.0, a=-3, b=3)
        torch.nn.init.trunc_normal_(self.weight, std=0.02, mean=0.0, a=-0.06, b=0.06)

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
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

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms

        return (x * self.weight).to(in_dtype)
    
def silu(x: torch.Tensor):
    return x * torch.sigmoid(x)

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
    
    def forward(self, x):
        return self.w2(silu(self.w1(x))* self.w3(x))

class SpeedRunFFN(nn.Module):
    def __init__(
        self,
        d_model: int, 
        d_ff: int,
    ):
        super().__init__()
        self.W1 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)

    def forward(self, x):
        x = self.W1(x)
        x = torch.relu(x) ** 2   
        x = self.W2(x)
        return x
    
class RotaryEmbedding(nn.Module):
    def __init__(self, context_length: int, dim: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache", RotaryEmbedding._init_cache(context_length, dim, theta)
        )
        self._freq_cis_cache: Float[Tensor, "2 context_length half_dim"]

    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float) -> Float[Tensor, "2 context_length half_dim"]:
        assert dim % 2 == 0

        d = torch.arange(0, dim, 2) / dim
        freqs = torch.tensor(theta) ** -d
        t = torch.arange(context_length)

        freqs = einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(
        self, x: Float[Tensor, "... seq d"], pos_id: Int[Tensor, "... seq"] | None
    ) -> Float[Tensor, "... seq d"]:
        x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2).unbind(0)

        if pos_id is not None:
            cos, sin = einx.get_at("cos_sin [pos] half_dim, ... -> cos_sin ... half_dim", self._freq_cis_cache, pos_id)
        else:
            seq_len = x.size(-2)
            cos, sin = self._freq_cis_cache[:, :seq_len, :].unbind(0)

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2

        result = torch.concat((x1_rot, x2_rot), dim=-1)
        return result
    
def softmax(x: Float[Tensor, "..."], dim: int = -1) -> Float[Tensor, "..."]:
    x = x - torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x)
    return exp_x / torch.sum(x, dim=dim, keepdim=True)

def scaled_dot_product_attention(
    q: Float[Tensor, "... d_k"],
    k: Float[Tensor, "... d_k"],
    v: Float[Tensor, "... d_k"],
    mask: Bool[Tensor, "seq_q seq_k"] | None = None
) -> Float[Tensor, "... d_k"]:
    d_k = q.size(-1)
    score = einsum(q, k, "... q d, ... k d -> ... q k") / math.sqrt(d_k)
    if mask is not None:
        score = score.masked_fill(~mask, float('-inf'))
    attn = softmax(score)
    return einsum(attn, v, "... q k, ... k d -> ... q d")

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding | None = None,
    ):
        super().__init__()
        if positional_encoder is None:
            warnings.warn("No positional encoder provided", stacklevel=2)
            
        assert d_model % num_heads == 0
        self.d_model =d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.W_q = Linear(self.d_model, self.num_heads * self.d_k)
        self.W_k = Linear(self.d_model, self.num_heads * self.d_k)
        self.W_v = Linear(self.d_model, self.num_heads * self.d_v)

        self.W_o = Linear(self.num_heads * self.d_v, self.d_model)

        self.q_norm = RMSNorm(self.d_k)
        self.k_norm = RMSNorm(self.d_k)

        self.positional_encoder: RotaryEmbedding | None = positional_encoder 
    
    def forward(
        self, x: Float[Tensor, "batch_size seq_len d_model"],
        token_positions: Int[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "batch_size seq_len d_model"]:
        seq_len = x.size(-2)
        q, k, v= self.W_q(x), self.W_k(x), self.W_v(x)

        # Multi-head
        q = rearrange(q, "... seq (h d) -> ... h seq d", h = self.num_heads)
        k = rearrange(k, "... seq (h d) -> ... h seq d", h = self.num_heads)
        v = rearrange(v, "... seq (h d) -> ... h seq d", h = self.num_heads)

        # RoPE on K V
        if self.positional_encoder is not None:
            if token_positions is not None:
                token_positions = rearrange(token_positions, "... seq -> ... 1 seq")
            
            q = self.positional_encoder(q, token_positions)
            k = self.positional_encoder(k, token_positions)

        # QKNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Causal Mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = rearrange(attn, "... h seq d -> ... seq (h d)").contiguous()

        return self.W_o(attn)
    
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryEmbedding | None,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_model=d_model, 
            num_heads=num_heads, 
            positional_encoder=positional_encoder,
        )
        self.ln1 = RMSNorm(d_model=d_model)
        self.ffn = SpeedRunFFN(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model=d_model)
    
    def forward(self, x: torch.Tensor):
        # 使用 checkpoint 包装注意力层
        x = x + self.attn(self.ln1(x))

        # 使用 checkpoint 包装 FFN 层
        x = x + self.ffn(self.ln2(x))
        return x
    
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = 10000.0
    ):
        super().__init__()
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model)
        d_head = d_model // num_heads
        self.positional_encoder = (
            RotaryEmbedding(context_length, d_head, theta) if theta is not None else None
        )

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                             positional_encoder=self.positional_encoder)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self, x: Int[Tensor, "batch_size seq_len"],
    ) -> Float[Tensor, "... vocab_size"]:
        x = self.token_embeddings(x)
        
        for layer in self.layers:
            x = layer(x)

        return self.lm_head(self.ln_final(x))

################## Generate ##################

@torch.no_grad()
def _sample_next_token(
    logits: torch.Tensor,        # (1, vocab_size)
    temperature: float = 1.0,
    top_p: float | None = None,  # None 表示不做 top-p 截断
) -> torch.Tensor:
    if temperature == 0:
        return logits.argmax(dim=-1)
    else:
        logits = logits / temperature

    probs = softmax(logits)

    if top_p is not None:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        mask = cumulative_probs > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False

        sorted_probs[mask] = 0.0
        filtered_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        sampled_position = torch.multinomial(filtered_probs, num_samples=1)
        next_token = sorted_indices.gather(-1, sampled_position) 
    else:
        next_token = torch.multinomial(probs, num_samples=1)

    return next_token

@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float | None = 0.9,
    eot_token: str = "<|endoftext|>",
    device: str = "cuda",
) -> str:
    model.eval()

    prompt_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0) #(1, length)

    eot_ids = tokenizer.encode(eot_token)
    assert len(eot_ids) == 1, f"eot_token should encode to 1 id, got {eot_ids}"
    eot_id = eot_ids[0]
    
    context_length = model.context_length

    generated_ids = []
    for _ in range(max_new_tokens):
        input_window = input_ids[:, -context_length:]

        logits = model(input_window)    # (1, T, vocab_size)
        last_logits = logits[:, -1, :] # (1, vocab_size)

        next_token = _sample_next_token(last_logits, temperature, top_p)

        next_id = next_token.item()
        if next_id == eot_id:
            break

        generated_ids.append(next_id)
        input_ids = torch.cat([input_ids, next_token], dim=-1) # (1, T+1)
        
    return tokenizer.decode(generated_ids)

if __name__ == "__main__":
    model = TransformerLM(vocab_size=32000, context_length=1024, d_model=768, d_ff=3072, num_heads=12, num_layers=12)
    params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {params / 1e6:.2f}M") 
    input = torch.randint(0, 32000, (2, 1024))
    logits = model(input)
    print(logits.shape)