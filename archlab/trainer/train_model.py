import torch
from torch import Tensor
from jaxtyping import Float

def cross_entropy(
    logits: Float[Tensor, "batch_size seq_len vocab_size"],
    targets: Float[Tensor, "batch_size seq_len"]
) -> float:
    max = logits.amax(dim=-1, keepdim=True)
    z = logits - max
    lse = max.squeeze(-1) + torch.log(torch.exp(z).sum(-1))

    tgt_logit = logits.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    log_prob = lse - tgt_logit

    return log_prob.mean()

