import torch
from .RMSNorm import RMSNorm
from .Linear import Linear
from .Transformer_block import Transformer_block
from .Embedding import Embedding

class Transformer_ln(torch.nn.Module):
  def __init__(self, d_model:int, d_ff:int, num_heads:int, sequence_length:int, vocab_size:int, context_length:int, num_layer:int, use_rope=None, theta=None, device=None, dtype=None):
    super().__init__()
    self.embedding = Embedding(vocab_size, d_model)
    self.blocks = torch.nn.ModuleList([
      Transformer_block(d_model=d_model, d_ff=d_ff, num_heads=num_heads, sequence_length=sequence_length, use_rope=True, theta=theta, max_seq_len=context_length, device=device, dtype=dtype)
      for _ in range(num_layer)
    ])
    self.rmsnorm = RMSNorm(d_model, device=device, dtype=dtype)
    self.ffn = Linear(d_model, vocab_size, device=device, dtype=dtype)


  def forward(self, token_ids):
    batch, sequence_length = token_ids.shape[0], token_ids.shape[1]
    x = self.embedding(token_ids)
    token_position = torch.arange(0, sequence_length).expand(batch, sequence_length)
    for block in self.blocks:
      x = block(x, token_position)
    x = self.rmsnorm(x)
    x = self.ffn(x)
    return x

    
