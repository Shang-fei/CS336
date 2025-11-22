import torch
import torch.nn as nn
from einops import einsum

class RMSNorm(nn.Module):
  def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    super().__init__()
    self.eps = eps
    self.d_model = d_model
    factory_kwargs = {"device": device, "dtype": dtype}
    self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

  
  def forward(self, x: torch.Tensor):
    in_dtype = x.dtype
    x = x.to(torch.float32)

    RMS = (torch.sum(x**2, dim=-1, keepdim=True)/self.d_model + self.eps)**0.5
    result = einsum(x/RMS, self.weight, "... d_model, d_model -> ... d_model")
    
    return result.to(in_dtype)