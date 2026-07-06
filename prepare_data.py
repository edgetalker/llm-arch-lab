import time
import os
from pathlib import Path
import numpy as np

# 导入你优化后的高性能 Tokenizer
from archlab.tokenizer.bpe_tokenizer import Tokenizer


# ============ 配置 ============
TOKENIZER_VOCAB  = 'archlab/tokenizer/tinystory/vocab.pkl'
TOKENIZER_MERGES = 'archlab/tokenizer/tinystory/merges.pkl'
SPECIAL_TOKENS   = ["<|endoftext|>"]
TOKEN_DTYPE      = np.uint16

# 推荐核心数：使用 CPU 最大核心数，也可设为特定值 (如 8, 16 等)
NUM_WORKERS      = os.cpu_count() or 1 

JOBS = [
    #("data/owt_valid.txt", "data/owt_valid.bin"),
    ("data/TinyStoriesV2-GPT4-train.txt", "data/tinystories_train.bin"),
]
# ==============================


def encode_file(tokenizer: Tokenizer, input_txt: str, output_bin: str) -> None:
    input_path  = Path(input_txt)
    output_path = Path(output_bin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    input_size_gb = input_path.stat().st_size / 1e9
    print(f"\n=== {input_path.name} → {output_path.name} ===")
    print(f"input size: {input_size_gb:.2f} GB")
    print(f"using {NUM_WORKERS} process workers for parallel encoding...")
    
    t0 = time.time()
    
    # 工业级流式持久化：避免一次性在内存中堆积几亿个 Python Int 对象。
    # 我们每攒够 1000 万个 token，就将其打包成 numpy array 追加写入磁盘。
    chunk_buffer = []
    chunk_limit = 10_000_000 
    total_tokens_count = 0
    last_report = t0
    
    # 确保写入新文件（覆盖旧文件）
    if output_path.exists():
        output_path.unlink()

    # 最大 Token ID 溢出校验
    max_vocab_id = len(tokenizer.vocab) - 1
    assert max_vocab_id < np.iinfo(TOKEN_DTYPE).max, \
        f"vocab size {max_vocab_id + 1} overflows {TOKEN_DTYPE}"

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f, open(output_path, "ab") as f_out:
        # 激活多进程并行迭代
        token_stream = tokenizer.encode_iterable(
            f, 
            chunk_lines=20000,   # 每个子进程任务分配 2 万行，平衡负载与 IPC
            num_workers=NUM_WORKERS
        )
        
        for tid in token_stream:
            chunk_buffer.append(tid)
            
            # 当缓冲区满时，序列化并追加写入磁盘，清空内存
            if len(chunk_buffer) >= chunk_limit:
                arr = np.array(chunk_buffer, dtype=TOKEN_DTYPE)
                arr.tofile(f_out)
                total_tokens_count += len(chunk_buffer)
                chunk_buffer = []
                
            # 每 5 秒打一次进度
            now = time.time()
            if now - last_report >= 5.0:
                elapsed = now - t0
                current_total = total_tokens_count + len(chunk_buffer)
                rate = current_total / elapsed
                print(f"  [{elapsed:6.1f}s] {current_total:>12,} tokens  ({rate/1e3:.1f}k tok/s)")
                last_report = now
                
        # 写入最后不足一个块的残余数据
        if chunk_buffer:
            arr = np.array(chunk_buffer, dtype=TOKEN_DTYPE)
            arr.tofile(f_out)
            total_tokens_count += len(chunk_buffer)
            chunk_buffer = []

    encode_time = time.time() - t0
    total_time = encode_time  # 采用流式边跑边写，写盘时间几乎被完全隐藏

    # 打印最终统计与校验
    final_bin_size_mb = output_path.stat().st_size / 1e6
    print(f"\n  done in {total_time:.1f}s (overlapped encode & write)")
    print(f"  tokens:      {total_tokens_count:,}")
    print(f"  output size: {final_bin_size_mb:.1f} MB")
    print(f"  compression: { (input_size_gb * 1000) / final_bin_size_mb:.2f}x")


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