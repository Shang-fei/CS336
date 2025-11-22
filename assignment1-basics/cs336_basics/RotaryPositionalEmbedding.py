import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
  def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
    super().__init__()
    freqs = (1.0 / theta**(torch.arange(0, d_k, 2)[:(d_k//2)].float() / d_k))
    position_idx = torch.arange(0, max_seq_len, device=device).float()
    freqs = torch.outer(position_idx, freqs)
    self.register_buffer('cos_cache', torch.cos(freqs), persistent=False)
    self.register_buffer('sin_cache', torch.sin(freqs), persistent=False)

  def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
    cos = self.cos_cache[token_positions]
    sin = self.sin_cache[token_positions]

    x_even = x[...,::2]
    x_odd = x[...,1::2]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    # Re-interleave
    out = torch.empty_like(x)
    out[..., ::2] = out_even
    out[..., 1::2] = out_odd

    return out

  





