import torch
import torch.nn as nn 

from einops import einsum

class Linear(nn.Module):
  def __init__(self, in_features, out_features, device=None, dtype=None):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    factory_kwargs = {"device": device, "dtype": dtype}
    self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
    std = 2 / (self.in_features + self.out_features)
    nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3*std, b=3*std)
  
  def forward(self, x: torch.Tensor):
    return einsum(x, self.weight, "... din, dout din -> ... dout")
    