import torch
import math
import numpy as np
from einops import einsum


def softmax(x: torch.Tensor, dimension: int):
    x_shifted = x - torch.max(x, dim=dimension, keepdim=True).values
    exp_x = torch.exp(x_shifted)
    return exp_x / torch.sum(exp_x, dim=dimension, keepdim=True)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor):
    loss = 0.0
    """
    log_softmax = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)
    target_log_probs = torch.gather(
        log_softmax, dim=-1, index=targets.unsqueeze(-1)
    ).unsqueeze(1)
    loss = -target_log_probs.mean()
    """
    print(inputs.shape)
    input_shifted = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    exp_input_shifted = torch.exp(input_shifted)
    targets_logits = torch.gather(input_shifted, dim=-1, index=targets.unsqueeze(-1))
    loss -= (
        targets_logits - torch.log(exp_input_shifted.sum(dim=-1, keepdim=True))
    ).sum() / inputs.shape[0]

    return loss


def scaled_dot_product_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    attn_scores = einsum(Q, K, "... n d, ... m d -> ... n m") / math.sqrt(d_k)
    if mask is not None:
        attn_scores.masked_fill_(~mask, float("-inf"))

    attn_probs = softmax(attn_scores, dimension=-1)
    output = einsum(attn_probs, V, "... n m, ... m d -> ... n d")

    return output

def lr_cosine_schedule(t:int, lr_max:float, lr_min:float, Tw:int, Tc:int):
    if t < Tw:
        return lr_max * t / Tw

    elif t >= Tw and t <= Tc:
        return lr_min + 0.5 * (1 + math.cos((t - Tw) / (Tc - Tw) * math.pi)) * (lr_max - lr_min)

    else:
        return lr_min
    
def gradient_clipping(parameters, max_l2_norm: float, eps = 1e-6):
    grads = [p.grad for p in parameters if p.grad is not None]
    
    if len(grads) == 0:
        return
    
    total_norm = torch.sqrt(sum(torch.sum(grad ** 2) for grad in grads))
    
    if total_norm <= max_l2_norm:
        return
    
    clip_coef = max_l2_norm / (total_norm + eps)
    for grad in grads:
        grad.mul_(clip_coef)

def get_batch(dataset, batch_size:int, context_length:int, device:str):
    starts = np.random.randint(0, len(dataset)-context_length, size=batch_size)

    input = np.stack([dataset[start:start+context_length] for start in starts])
    target = np.stack([dataset[start+1:start+context_length+1] for start in starts])

    return (
        torch.as_tensor(input, dtype=torch.long, device=device),
        torch.as_tensor(target, dtype=torch.long, device=device)
    )

def save_checkpoint(model, optimizer, iteration, out): 
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration