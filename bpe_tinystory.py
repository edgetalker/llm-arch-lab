"""
Train a BPE tokenizer on a text corpus.

Run with:
    uv run python -m archlab.script.train_tokenizer
"""
import pickle
import time
from pathlib import Path

from archlab.trainer.train_tokenizer import train_bpe


# ============ 配置 ============
INPUT_TXT = "data/TinyStoriesV2-GPT4-train.txt"   
VOCAB_SIZE = 10000                                  
SPECIAL_TOKENS = ["<|endoftext|>"]
OUTPUT_DIR = Path("trainer/tokenizer")
# ==============================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 输入文件检查
    input_path = Path(INPUT_TXT)
    assert input_path.exists(), f"input file not found: {input_path.resolve()}"
    print(f"input:      {input_path.resolve()}")
    print(f"input size: {input_path.stat().st_size / 1e6:.1f} MB")
    print(f"vocab_size: {VOCAB_SIZE}")
    print(f"special:    {SPECIAL_TOKENS}")
    print()
    
    # 训练
    t0 = time.time()
    vocab, merges = train_bpe(
        input_path=str(input_path),
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )
    elapsed = time.time() - t0
    print(f"\ntrained in {elapsed:.1f}s")
    print(f"final vocab size: {len(vocab)}")
    print(f"merges learned:   {len(merges)}")
    
    # 完整性 sanity check
    assert len(vocab) <= VOCAB_SIZE, f"vocab exceeded budget: {len(vocab)} > {VOCAB_SIZE}"
    assert len(merges) == len(vocab) - 256 - len(SPECIAL_TOKENS), \
        f"merges count mismatch: {len(merges)} vs {len(vocab) - 256 - len(SPECIAL_TOKENS)}"
    
    # 存盘 (pickle 二进制,兼容你的 Tokenizer.from_files)
    vocab_path  = OUTPUT_DIR / "vocab.pkl"
    merges_path = OUTPUT_DIR / "merges.pkl"
    
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)
    
    print(f"\nsaved:")
    print(f"  {vocab_path}  ({vocab_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {merges_path} ({merges_path.stat().st_size / 1024:.1f} KB)")
    
    # 打印一些 vocab 样本,直观看一眼学到了什么
    print(f"\nfirst 5 merges learned:")
    for i, (a, b) in enumerate(merges[:5]):
        print(f"  {i}: {a!r} + {b!r} → {(a + b)!r}")
    print(f"\nlast 5 merges learned:")
    for i, (a, b) in enumerate(merges[-5:]):
        print(f"  {len(merges) - 5 + i}: {a!r} + {b!r} → {(a + b)!r}")


if __name__ == "__main__":
    main()