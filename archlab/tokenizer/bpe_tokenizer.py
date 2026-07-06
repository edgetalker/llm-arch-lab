import os
import pickle
import regex as re
from typing import Iterator, Iterable
from concurrent.futures import ProcessPoolExecutor

def _parallel_encode_worker(tokenizer_instance: "Tokenizer", text_chunk: str) -> list[int]:
    """子进程执行的实际编码工作"""
    return tokenizer_instance.encode(text_chunk)

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens= sorted(
            special_tokens if special_tokens is not None else [],
            key = len,
            reverse=True
        )
        self.lookup = {v: k for k, v in vocab.items()}
        self.pretoken_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        self.cache: dict[bytes, list[int]] = {}
        self.byte_to_bytes = [bytes([i]) for i in range(256)]
    
    @classmethod
    def from_files(
        cls, vocab_filepath: str, 
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab,merges, special_tokens)
    
    def _merge_word(
        self,
        word: tuple[bytes, ...],
        best_pair: tuple[bytes, ...]
    ):
        new_word = []
        new_tok = best_pair[0] + best_pair[1]
        i = 0
        while i < len(word):
            if i+1 < len(word) and (word[i], word[i+1]) == best_pair:
                new_word.append(new_tok)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return  tuple(new_word)
        
    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
            chunks = re.split(pattern, text)
        else:
            chunks = [text]
        
        encode_list = []
        for split_chunk in chunks:
            if split_chunk in self.special_tokens:
                encode_list.append(self.lookup[split_chunk.encode('utf-8')])
            else:
                for match in re.finditer(self.pretoken_pattern, split_chunk):
                    c = match.group().encode("utf-8")
                    # cache 命中
                    if c in self.cache:
                        encode_list.extend(self.cache[c])
                        continue
                    # cache 不命中
                    word = tuple(self.byte_to_bytes[b] for b in c)
                    while True:
                        candidates = [
                            (self.merge_rank[(word[i], word[i+1])], i)
                            for i in range(len(word)-1)
                            if (word[i], word[i+1]) in self.merge_rank
                        ]
                        if not candidates:
                            break
                        min_rank, _ = min(candidates)
                        best_pair = self.merges[min_rank]
                        word = self._merge_word(word, best_pair)
                    
                    ids = [self.lookup[token] for token in word]
                    self.cache[c] = ids

                    encode_list.extend(ids)
        return encode_list
    
    def encode_iterable(
        self, 
        iterable: Iterable[str],
        chunk_lines: int = 5000,
        num_workers: int | None = None
    ) -> Iterator[int]:
        if not num_workers:
            num_workers = os.cpu_count() or 1
        
        if num_workers == 1:
            batch = []
            for line in iterable:
                batch.append(line)
                if len(batch) > 2000:
                    yield from self.encode("".join(batch))
                    batch = []
            if batch:
                yield from self.encode("".join(batch))
            return
        
        def chunk_generator():
            current_chunk = []
            for line in iterable:
                current_chunk.append(line)
                if len(current_chunk) >= chunk_lines:
                    yield "".join(current_chunk)
                    current_chunk = []
            if current_chunk:
                yield "".join(current_chunk)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 维持一个未来任务队列，保证流式处理大文件时的内存安全
            futures = []
            
            # 预热任务（防止一次性把 11GB 文件全读进内存，保持队列长度为 2 * num_workers）
            chunk_iter = chunk_generator()
            for _ in range(num_workers * 2):
                try:
                    chunk = next(chunk_iter)
                    # 提交任务时，将 self（分词器实例）一同打包传入子进程
                    futures.append(executor.submit(_parallel_encode_worker, self, chunk))
                except StopIteration:
                    break
            
            # 消费已完成的任务，并不断推入新任务
            while futures:
                # 阻塞等待最先提交的任务完成，保证输出的 Token 顺序与输入文件完全一致
                finished_future = futures.pop(0)
                encoded_ids = finished_future.result()
                yield from encoded_ids
                
                # 尝试推入下一个大块任务
                try:
                    next_chunk = next(chunk_iter)
                    futures.append(executor.submit(_parallel_encode_worker, self, next_chunk))
                except StopIteration:
                    pass
        

    def decode(self, ids: list[int]) -> str:
        bytes_squence = b"".join(self.vocab[id] for id in ids)
        return bytes_squence.decode("utf-8", errors="replace")