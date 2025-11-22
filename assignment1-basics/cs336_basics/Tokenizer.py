import regex as re
import pickle
from collections.abc import Iterator, Iterable

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def split_by_specail_token(special_pattern, text: str) -> list[str]:
  if not special_pattern:
    return [text]
  chunks = special_pattern.split(text) 
  return [c for c in chunks if c]

def apply_merges(bytes2id, ranks, text: str) -> list[int]:
  tokens: list[int] = []
  for m in PAT.finditer(text):
    word_str = m.group(0).encode("utf-8")
    current_token_bytes: list[bytes] = [bytes([b]) for b in word_str]
    while len(current_token_bytes) >= 2:
      bytes_pairs = []
      for i in range(len(current_token_bytes)-1):
        bytes_pairs.append((current_token_bytes[i], current_token_bytes[i+1]))

      merge_pair, merge_rank = None, len(ranks)
      for pair in bytes_pairs:
        current_merge_rank = ranks.get(pair, len(ranks)) 
        if current_merge_rank != -1 and current_merge_rank < merge_rank:
          merge_pair = pair
          merge_rank = current_merge_rank
      
      if not merge_pair:
        break

      i = 0
      new_token_bytes = []
      while(i < len(current_token_bytes)):
        if(i + 1 < len(current_token_bytes) and (current_token_bytes[i], current_token_bytes[i+1]) == merge_pair):
          new_token_bytes.append(merge_pair[0] + merge_pair[1])
          i += 2
        else:
          new_token_bytes.append(current_token_bytes[i])
          i += 1
      
      current_token_bytes = new_token_bytes  

    for token_bytes in current_token_bytes:
      tokens.append(bytes2id[token_bytes]) 
  return tokens

def initialize_special_pattern(specials_tokens: list[str]):
  if specials_tokens:
    specials_tokens = sorted(specials_tokens, key=len, reverse=True)
    delimiter = '|'.join(re.escape(token) for token in specials_tokens)
    specails_pattern = re.compile(f"({delimiter})")
    return specails_pattern
  else:
    return None

class Tokenizer:
  def __init__(self, vocab, merges, special_tokens = None):
    self.vocab = vocab
    self.merges = merges
    self.special_tokens = special_tokens if special_tokens else []
    self.ranks = {pair: i for i, pair in enumerate(self.merges)}
    self.special_pattern = initialize_special_pattern(self.special_tokens)
    self.special_tokens_bytes = [token.encode("utf-8") for token in self.special_tokens]

    self.bytes2id = {v : k for k, v in vocab.items()}
    for special_token_bytes in self.special_tokens_bytes:
      if special_token_bytes not in vocab.values():
          new_id = len(self.vocab)
          self.vocab[new_id] = special_token_bytes
          self.bytes2id[special_token_bytes] = new_id

  @classmethod
  def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None): 
    with open(vocab_filepath, "rb") as f:
      vocab = pickle.load(f)
    
    with open(merges_filepath, 'rb') as f:
      merges = pickle.load(f)

    return cls(vocab, merges, special_tokens)

  def encode(self, text: str) -> list[int]:
    tokens: list[int] = []
    chunks = split_by_specail_token(self.special_pattern, text)
    for item in chunks:
      if self.special_tokens and item in self.special_tokens:
        item_bytes = item.encode("utf-8")
        tokens.append(self.bytes2id[item_bytes])
      else:
        tokens.extend(apply_merges(self.bytes2id, self.ranks, item))
    return tokens

  def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    for text in iterable:
      encoded_list = self.encode(text)
      yield from encoded_list
  
  def decode(self, ids: list[int]) -> str:
    return b''.join([self.vocab[t] for t in ids]).decode('utf-8',errors='replace')