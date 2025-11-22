import os
import pickle
import pathlib

from tests.adapters import run_train_bpe

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
INPUT_FILE = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")

TOKENIZER_DIR = pathlib.Path(__file__).resolve().parent.parent / "tokenizer"
VOCAB_FILE = os.path.join(TOKENIZER_DIR, "TinyStoriesV2_vocab.pkl")
MERGES_FILE = os.path.join(TOKENIZER_DIR, "TinyStoriesV2_merges.pkl")

if __name__ == '__main__':
  vocab_size = 10000
  special_tokens = ["<|endoftext|>"]

  vocab, merges = run_train_bpe(INPUT_FILE, vocab_size, special_tokens)

  os.makedirs(TOKENIZER_DIR, exist_ok = True)
  with open(VOCAB_FILE, "wb") as f:
    pickle.dump(vocab, f)

  with open(MERGES_FILE, "wb") as f:
    pickle.dump(merges, f)

  longest_token = max(vocab.values(), key=len)
  print("最长token:", longest_token, "长度:", len(longest_token))

