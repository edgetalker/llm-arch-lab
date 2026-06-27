"""
Encode raw text files into uint16 token-id binaries for training.

Run with:
    uv run python -m archlab.script.prepare_data
"""
import time
from pathlib import Path

import numpy as np

from archlab.tokenizer.bpe_tokenizer import Tokenizer


# ============ 配置 ============
TOKENIZER_VOCAB  = 'archlab/tokenizer/tinystory/vocab.pkl'
TOKENIZER_MERGES = 'archlab/tokenizer/tinystory/merges.pkl'
SPECIAL_TOKENS   = ["<|endoftext|>"]
TOKEN_DTYPE      = np.uint16

JOBS = [
    ("data/TinyStoriesV2-GPT4-valid.txt", "data/tinystories_val.bin"),
    # ("data/TinyStoriesV2-GPT4-train.txt", "data/tinystories_train.bin"),
]
# ==============================


def encode_file(tokenizer: Tokenizer, input_txt: str, output_bin: str) -> None:
    input_path  = Path(input_txt)
    output_path = Path(output_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    input_size_mb = input_path.stat().st_size / 1e6
    print(f"\n=== {input_path.name} → {output_path.name} ===")
    print(f"input size: {input_size_mb:.1f} MB")
    
    t0 = time.time()
    ids: list[int] = []
    last_report = t0
    
    with open(input_path, "r", encoding="utf-8") as f:
        for tid in tokenizer.encode_iterable(f):
            ids.append(tid)
            
            # 每 5 秒打一次进度
            now = time.time()
            if now - last_report >= 5.0:
                elapsed = now - t0
                rate = len(ids) / elapsed
                print(f"  [{elapsed:6.1f}s] {len(ids):>12,} tokens  ({rate/1e3:.1f}k tok/s)")
                last_report = now
    
    encode_time = time.time() - t0
    
    # 写盘
    arr = np.array(ids, dtype=TOKEN_DTYPE)
    assert arr.max() < np.iinfo(TOKEN_DTYPE).max, \
        f"token id {arr.max()} overflows {TOKEN_DTYPE}"
    arr.tofile(output_path)
    
    total_time = time.time() - t0
    print(f"\n  done in {total_time:.1f}s (encode {encode_time:.1f}s + write {total_time-encode_time:.1f}s)")
    print(f"  tokens:      {len(arr):,}")
    print(f"  output size: {output_path.stat().st_size / 1e6:.1f} MB")
    print(f"  range:       [{arr.min()}, {arr.max()}]")
    print(f"  compression: {input_size_mb / (output_path.stat().st_size / 1e6):.2f}x")


def main():
    print(f"loading tokenizer from {TOKENIZER_VOCAB}")
    tokenizer = Tokenizer.from_files(
        TOKENIZER_VOCAB, TOKENIZER_MERGES, SPECIAL_TOKENS
    )
    print(f"vocab size: {len(tokenizer.vocab)}")
    assert len(tokenizer.vocab) < np.iinfo(TOKEN_DTYPE).max, \
        f"vocab size {len(tokenizer.vocab)} exceeds {TOKEN_DTYPE} range"
    
    t_start = time.time()
    for input_txt, output_bin in JOBS:
        encode_file(tokenizer, input_txt, output_bin)
    
    print(f"\n{'='*50}")
    print(f"all done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()