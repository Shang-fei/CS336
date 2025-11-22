import torch
import math
from collections.abc import Callable, Iterable
from typing import Optional

class AdamW(torch.optim.Optimizer):
  def __init__(self, params, lr=1e-3, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8,):
    defaults = {"lr": lr, "weight_decay": weight_decay, "beta1":betas[0], "beta2":betas[1], "eps":eps}
    super().__init__(params, defaults)
  
  def step(self, closure: Optional[Callable] = None):
    loss = None if closure is None else closure()
    for group in self.param_groups:
      lr, beta1, beta2, eps, weigh_decay = group["lr"], group["beta1"], group["beta2"], group["eps"], group["weight_decay"]
      
      for p in group["params"]:
        if p.grad == None:
          continue
          
        state = self.state[p]
        t = state.get("t", 1)
        m = state.get("m", torch.zeros_like(p.data))
        n = state.get("n", torch.zeros_like(p.data))

        grad = p.grad.data
        m = (beta1 * m + (1 - beta1) * grad)
        n = (beta2 * n + (1 - beta2) * grad ** 2)

        lr_t = lr * (1 - beta2**t)**0.5 / (1 - beta1**t)
        p.data -= lr_t * m / (n**0.5 + eps)
        p.data -= lr * weigh_decay * p.data

        state["t"] = t + 1
        state["m"] = m
        state["n"] = n
      
    return loss



    
