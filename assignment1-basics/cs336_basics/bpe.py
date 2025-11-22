import regex as re
import multiprocessing as mp
from collections import defaultdict, Counter
from functools import partial
from cs336_basics.pretokenization_example import find_chunk_boundaries

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def initialize_special_pattern(specails_tokens: list[str]):
  if specails_tokens:
    delimiter = '|'.join(re.escape(token) for token in specails_tokens)
    specails_pattern = re.compile(delimiter)
    return specails_pattern
  else:
    return None

def initialize_vocabulary(specials_tokens: list[str]) -> dict[int, bytes]:
  specials_bytes_list = [token.encode("utf-8") for token in specials_tokens]
  utf8_bytes_list = [bytes([i]) for i in range(256)]
  init_vocab_bytes_list = specials_bytes_list + utf8_bytes_list
  init_vocab = {vocab_index: vocab_item for vocab_index, vocab_item in enumerate(init_vocab_bytes_list)}
  return init_vocab

def remove_special_tokens(text: str, special_pattern:re.Pattern | None) -> list[str]:
  if not special_pattern:
    return [text]
  chunks = special_pattern.split(text)
  return [c for c in chunks if c]

def pre_tokenization(word: str) -> dict[tuple[bytes], int]:
  pre_token_counts = Counter()
  for m in PAT.finditer(word):
      word_split = m.group(0).encode("utf-8")
      byte_sequence = tuple(bytes([b]) for b in word_split)
      pre_token_counts[byte_sequence] += 1
  return pre_token_counts

def process_chunk_worker(chunk: str, special_pattern: re.Pattern | None) -> Counter:
  text_segments = remove_special_tokens(chunk, special_pattern)
  chunk_pre_token_counts = Counter()
  for segment in text_segments:
    segment_token_counts = pre_tokenization(segment)
    chunk_pre_token_counts.update(segment_token_counts)
  return chunk_pre_token_counts

def merge_dict(dict_list: list[dict]) -> defaultdict[tuple[bytes], int]: 
  result = defaultdict(int)
  for per_dict in dict_list:
    for k, v in per_dict.items():
      result[k] += v
  return result

def count_pair(pre_token_counts: Counter) -> tuple[Counter, defaultdict[tuple[bytes, bytes], set]]:
  pair2word = defaultdict(set)
  pair_bytes_counts = Counter()
  for vocab_item, vocab_count in pre_token_counts.items():
    if len(vocab_item) < 2:
      continue
    for start, end in zip(vocab_item[:-1], vocab_item[1:]):
      pair_bytes_counts[(start, end)] += vocab_count
      pair2word[(start, end)].add(vocab_item)
  return pair_bytes_counts, pair2word

def update_data(pre_token_counts: Counter, pair_bytes_counts: Counter, pair2word: defaultdict[tuple[bytes, bytes], set],  merges_pair: tuple):
  modified_tokens = pair2word[merges_pair]
  for old_token in modified_tokens.copy():

    i = 0
    new_token = []
    while(i < len(old_token)):
      if i + 1 < len(old_token) and (old_token[i], old_token[i+1]) == merges_pair:    
        new_token.append(merges_pair[0] + merges_pair[1])
        i += 2
      else:
        new_token.append(old_token[i])
        i += 1
    cnt = pre_token_counts.pop(old_token)
    pre_token_counts[tuple(new_token)] += cnt

    for start, end in zip(old_token[:-1], old_token[1:]):
      pair_bytes_counts[(start, end)] -= cnt
      pair2word[(start, end)].discard(old_token)
      if pair_bytes_counts[(start, end)] <= 0:
        del pair_bytes_counts[(start, end)]

    for start, end in zip(new_token[:-1], new_token[1:]):
      pair_bytes_counts[(start, end)] += cnt
      pair2word[(start, end)].add(tuple(new_token))

def merges_vocabulary(vocab: dict, merges: list, pre_token_counts: Counter, pair_bytes_counts: Counter, pair2word: defaultdict[tuple[bytes, bytes], set]):
  max_pair, _  = max(pair_bytes_counts.items(), key=lambda x: (x[1], x[0]))

  vocab[len(vocab)] = b''.join(max_pair)
  merges.append(max_pair)
  update_data(pre_token_counts, pair_bytes_counts, pair2word, max_pair)

def read_chunks(input_file, num_processes, special_tokens):
  with open(input_file, "rb") as f:
    boundaries = find_chunk_boundaries(f, num_processes, special_tokens)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
      f.seek(start)
      chunk_data = f.read(end - start).decode("utf-8", errors="ignore")
      yield chunk_data

def train_bpe(input_file, vocab_size: int, specials_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  special_pattern = initialize_special_pattern(specials_tokens)
  merges: list[tuple[bytes, bytes]] = []
  vocab = initialize_vocabulary(specials_tokens)

  pre_token_counts = Counter()
  worker = partial(process_chunk_worker, special_pattern=special_pattern)

  chunks = read_chunks(input_file, num_processes=8, special_tokens=b'<|endoftext|>')
  with mp.Pool() as pool:
      for chunk_counts in pool.imap_unordered(worker, chunks):
        pre_token_counts.update(chunk_counts)

  pair_bytes_counts, pair2word = count_pair(pre_token_counts)

  merges_num = vocab_size - len(vocab)
  for i in range(merges_num):
    merges_vocabulary(vocab, merges, pre_token_counts, pair_bytes_counts, pair2word)
    
  return vocab, merges


