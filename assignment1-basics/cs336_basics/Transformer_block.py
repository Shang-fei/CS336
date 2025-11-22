import torch
from .RMSNorm import RMSNorm
from .SwiGLU import SiwGLU
from .multihead_self_attention import Multihead_self_attention

class Transformer_block(torch.nn.Module):
  def __init__(self, d_model:int, d_ff:int, num_heads:int, sequence_length:int, use_rope=False, theta=None, max_seq_len=None, device=None, dtype=None):
    super().__init__()
    self.d_model = d_model
    self.d_ff = d_ff
    self.num_heads = num_heads
    self.rmsnorm1 = RMSNorm(d_model, device=device, dtype=dtype)
    self.rmsnorm2 = RMSNorm(d_model, device=device, dtype=dtype)
    self.self_attn = Multihead_self_attention(d_model, num_heads, sequence_length, use_rope=use_rope, theta=theta, max_seq_len=max_seq_len, device=device, dtype=dtype)
    self.ffn = SiwGLU(d_model, d_ff, device=device, dtype=dtype)
  
  def forward(self, x, token_positions):
    attn_out = self.self_attn(self.rmsnorm1(x), token_positions=token_positions)
    x = x + attn_out                       

    ff_out = self.ffn(self.rmsnorm2(x))
    x = x + ff_out                 
    return x
