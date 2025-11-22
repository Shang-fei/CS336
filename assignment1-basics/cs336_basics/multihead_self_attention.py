import torch
import torch
from .Linear import Linear
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding

from einops import rearrange
from .util import scaled_dot_product_attention
from .RotaryPositionalEmbedding import RotaryPositionalEmbedding


class Multihead_self_attention(torch.nn.Module):
  def __init__(self, d_model: int, num_heads: int, sequence_length:int, theta=None, max_seq_len=None, use_rope=False, device=None, dtype=None):
    super().__init__()
    self.d_model = d_model
    self.num_heads = num_heads
    self.use_rope = use_rope
    self.d_k = self.d_model // self.num_heads
    self.kwargs = {"device": device, "dtype": dtype}
    self.k_proj, self.q_proj, self.v_proj, self.o_proj = [Linear(d_model, d_model, **self.kwargs) for _ in range(4)]
    mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device))
    self.register_buffer("causal_mask", mask, persistent=False)
    if use_rope:
      self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)


  def forward(self , x, token_positions=None):
    mask = self.causal_mask
    Q, K, V = [rearrange(proj(x), "... sequence_length (num_heads d) -> ... num_heads sequence_length d", num_heads = self.num_heads)
                for proj in [self.q_proj, self.k_proj, self.v_proj]]
    if self.use_rope:
      Q, K = self.rope(Q, token_positions), self.rope(K, token_positions)
    attn_output = scaled_dot_product_attention(Q, K, V, mask)
    attn_output = rearrange(attn_output, "... num_heads sequence_length d -> ... sequence_length (num_heads d)", num_heads = self.num_heads)
    return self.o_proj(attn_output)


     
    


    
