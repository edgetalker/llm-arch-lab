import pickle
import regex as re
from typing import Iterator

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
                    pretoken = tuple(bytes([i]) for i in c) #(b't', b'h', b'e')
                    word = pretoken
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
                    
                    for token in word:
                        encode_list.append(self.lookup[token])
        return encode_list
    
    def encode_iterable(self, iterable) -> Iterator[int]:
        for line in iterable:
            yield from self.encode(line)

    def decode(self, ids: list[int]) -> str:
        bytes_squence = b"".join(self.vocab[id] for id in ids)
        return bytes_squence.decode("utf-8", errors="replace")