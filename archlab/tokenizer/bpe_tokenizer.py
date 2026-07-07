import os
import pickle
import regex as re
from typing import Iterator, Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[int, int]],
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab
        self.merges = merges

        base_id = 256 + len(special_tokens) if special_tokens else 256
        self.merge_id =  {pair: base_id + i for i, pair in enumerate(merges)}
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

        self.special_tokens= sorted(
            special_tokens if special_tokens is not None else [],
            key = len,
            reverse=True
        )
        self.lookup = {v: k for k, v in vocab.items()}
        self.pretoken_pattern = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        self._encode_word = lru_cache(maxsize=500000)(self._encode_word_impl)
    
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
        
    def _encode_word_impl(self, raw: bytes) -> tuple[int, ...]:
        word = tuple(raw)
        while True:
            candidates = []
            for i in range(len(word)-1):
                pair = (word[i], word[i+1])
                rank = self.merge_rank.get(pair)
                if rank is not None:
                    candidates.append((rank, i)) 

            if not candidates:
                break

            min_rank, idx = min(candidates)
            best_pair = self.merges[min_rank]
            new_id = self.merge_id[best_pair]
            word = word[:idx] + (new_id,) + word[idx+2:]
        return word 

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
                for match in self.pretoken_pattern.finditer(split_chunk):
                    raw = match.group().encode("utf-8")
                    encode_list.extend(self._encode_word(raw))
        return encode_list
    
    def encode_iterable(
        self, 
        iterable: Iterable[str],
        chunk_lines: int = 5000
    ) -> Iterator[int]:
        batch = []
        for line in iterable:
            batch.append(line)
            if len(batch) > chunk_lines:
                yield from self.encode("".join(batch))
                batch = []
        if batch:
            yield from self.encode("".join(batch))
        return
        
    def decode(self, ids: list[int]) -> str:
        bytes_sequence = b"".join(self.vocab[id] for id in ids)
        return bytes_sequence.decode("utf-8", errors="replace")