import os
import regex as re
import heapq
from typing import BinaryIO
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"
    
    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096 # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position) # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size) # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            
            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
        
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(
    input_path: str,
    start: int,
    end: int, 
    split_re: re.Pattern
) -> Counter:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    docs = split_re.split(chunk) if split_re else [chunk]
    local_counter = Counter()
    for doc in docs:
        for match in PAT.finditer(doc):
            key = tuple(match.group().encode("utf-8")) # b'abc' -> (97, 98, 99)
            local_counter[key] += 1
    return local_counter

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]      
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    
    pattern = "|".join(re.escape(t) for t in special_tokens)
    split_re = re.compile(pattern) if special_tokens else None

    # Pretokenization
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 16, special_tokens[0].encode("utf-8"))

    args = [(input_path, s, e, split_re) for s, e in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=cpu_count()) as pool:
        local_counters = pool.starmap(process_chunk, args)

    frequency_table = Counter()
    for lc in local_counters:
        frequency_table.update(lc)

    # Vocabulary initialization
    vocab = {i: bytes([i]) for i in range(256)}
    start_index = len(vocab)
    for i, tokens in enumerate(special_tokens):
        vocab[start_index + i] = tokens.encode("utf-8")
    index = len(vocab)
    merges = [] # (pair_id1, pair_id2)

    # Construct the pair_counts、 pair_to_words
    pair_counts = defaultdict(int)
    pair_to_words = defaultdict(set)
    heap = []

    for word, freq in frequency_table.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    # pair to heap
    for pair, cnt in pair_counts.items():
        heapq.heappush(heap, (-cnt, pair[0], pair[1]))

    # Train Loop
    while index < vocab_size and heap:
        best_pair = None
        while heap:
            neg_cnt, p0, p1 = heapq.heappop(heap)
            pair = (p0, p1)
            if pair_counts.get(pair, 0) == -neg_cnt:
                best_pair = pair
                break
        
        if not best_pair:
            break

        new_id = index
        new_byte = vocab[best_pair[0]] + vocab[best_pair[1]]
        vocab[new_id] = new_byte
        index += 1
        merges.append(best_pair)

        affected_words = pair_to_words.pop(best_pair, set())
        for old_word in list(affected_words):
            freq = frequency_table.pop(old_word)

            for i in range(len(old_word)-1):
                p = (old_word[i], old_word[i+1])
                new_cnt = pair_counts[p] - freq
                if new_cnt > 0:
                    pair_counts[p] = new_cnt
                    heapq.heappush(heap, (-new_cnt, p[0], p[1]))
                else:
                    del pair_counts[p]
                pair_to_words[p].discard(old_word)
            
            new_word = []
            i = 0
            while i < len(old_word):
                if (i+1 < len(old_word) and (old_word[i], old_word[i+1]) == best_pair):
                    new_word.append(new_id)
                    i += 2
                else:
                    new_word.append(old_word[i])
                    i += 1
            new_word = tuple(new_word)

            frequency_table[new_word] += freq
            for i in range(len(new_word)-1):
                p = (new_word[i], new_word[i+1])
                new_cnt = pair_counts[p] + freq
                pair_counts[p] = new_cnt
                pair_to_words[p].add(new_word)
                heapq.heappush(heap, (-new_cnt, p[0], p[1]))
            
    return vocab, merges