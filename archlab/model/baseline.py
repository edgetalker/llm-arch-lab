import math
import torch
from torch import Tensor
import torch.nn as nn

from einops import einsum, rearrange
from jaxtyping import Int, Float, Bool

from archlab.tokenizer.bpe_tokenizer import Tokenizer

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
        self, x: Float[Tensor, "... d_model"]
    ) -> Float[Tensor, "... d_model"]:
        act = self.w1(x)
        gate = act * torch.sigmoid(act)
        value = self.w3(x)
        return self.w2(gate * value)
    
def _rotate_pair(x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
    """
    Rotate pairs by 90°: (x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(-2)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None, 
    ):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        # Frequencies: theta_i = theta^(-2i/d_k) for i in [0, d_k/2)
        half_dim = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freq = theta ** (-half_dim / d_k)

        # Outer product: positions × frequencies
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq) #[max_seq_len, inv_freq]

        cos = torch.repeat_interleave(torch.cos(freqs), 2, dim=-1)
        sin = torch.repeat_interleave(torch.sin(freqs), 2, dim=-1)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self, x: Float[Tensor, "... d_k"],
        token_positions: Int[Tensor, "seq_len"]
    ) -> Float[Tensor, "... d_k"]:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        return x * cos + _rotate_pair(x) * sin
    
def _softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    x = x - x.amax(dim=dim, keepdim=True)
    return x.exp() / x.exp().sum(dim=dim, keepdim=True)

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
    attn = _softmax(score, dim=-1)
    return einsum(attn, v, "... q k, ... k d -> ... q d")

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model =d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = Linear(self.d_model, self.d_model)
        self.W_k = Linear(self.d_model, self.d_model)
        self.W_v = Linear(self.d_model, self.d_model)
        self.W_o = Linear(self.d_model, self.d_model)

        if max_seq_len is not None and theta is not None:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
        else:
            self.rope = None
    
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
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=q.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # Causal Mask
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device))
        attn = scaled_dot_product_attention(q, k, v, mask)
        attn = rearrange(attn, "... h seq d -> ... seq (h d)")
        return self.W_o(attn)
    
class TransformerBlock(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ffn = PositionwiseFFN(d_model=d_model, d_ff=d_ff)
        self.ln2 = RMSNorm(d_model=d_model)
    
    def forward(
        self, x: Float[Tensor, "... d_model"],
        token_positions: Int[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "... d_model"]:
        x = x + self.attn(self.ln1(x), token_positions)
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
        theta: float
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                             theta=theta, max_seq_len=context_length)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self, x: Int[Tensor, "batch_size seq_len"],
        token_positions: Float[Tensor, "..."] | None = None,
    ) -> Float[Tensor, "... vocab_size"]:
        x = self.token_embeddings(x)
        if token_positions is None:
            token_positions = torch.arange(x.size(-2), device=x.device)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
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

    probs = _softmax(logits, -1)

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
    lm = TransformerLM(vocab_size=256, context_length=16, d_model=64, num_layers=2, num_heads=4, d_ff=128, theta=10000.0)
    x = torch.randint(0, 256, (2, 16))
    logits = lm(x)
    print(logits.size())