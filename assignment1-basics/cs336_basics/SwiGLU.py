import torch
import torch.nn as nn
from .Linear import Linear

def SiLU(x: torch.Tensor):
  return x * torch.sigmoid(x)

class SiwGLU(nn.Module):
  def __init__(self, d_model: int, d_ff: int, device = None, dtype = None):
    super().__init__()
    factory_kwargs = {"device": device, "dtype": dtype}
    self.linear1 = Linear(d_model, d_ff, **factory_kwargs)
    self.linear2 = Linear(d_ff, d_model, **factory_kwargs)
    self.linear3 = Linear(d_model, d_ff, **factory_kwargs)

  def forward(self, x):
    return self.linear2(SiLU(self.linear1(x)) * self.linear3(x))


