import os
import regex as re
from typing import BinaryIO
from collections import Counter, defaultdict
from multiprocessing import Pool

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
        
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk(input_path, start, end, special_tokens):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    pattern = "|".join(re.escape(t) for t in special_tokens)
    docs = re.split(pattern, chunk) if special_tokens else [chunk]

    local_counter = Counter()
    for doc in docs:
        for match in re.finditer(PAT, doc): 
            c = match.group().encode("utf-8")
            key = tuple(bytes([i]) for i in c)
            local_counter[key] += 1
    return local_counter

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]      
):
    # Vocabulary initialization
    vocab = {i: bytes([i]) for i in range(256)}
    start_index = len(vocab)
    for i, tokens in enumerate(special_tokens):
        vocab[start_index + i] = tokens.encode("utf-8")

    # Pretokenization
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, 16, special_tokens[0].encode("utf-8"))

    args = [(input_path, s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=4) as pool:
        local_counters = pool.starmap(process_chunk, args)

    frequency_table = Counter()
    for lc in local_counters:
        frequency_table.update(lc)
    
    # Merges
    def merge_word(
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
    
    index = len(vocab)
    merges = []

    # Construct the pair_counts、 pair_to_words
    pair_counts = Counter()
    pair_to_words = defaultdict(set)
    for word, freq in frequency_table.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    # Train Loop
    while index < vocab_size:
        if not pair_counts:
            break
        
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        affected_words = pair_to_words[best_pair]

        for old_words in list(affected_words):
            freq = frequency_table[old_words]
            new_words = merge_word(old_words, best_pair)

            for i in range(len(old_words)-1):
                pair = (old_words[i], old_words[i+1])
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                pair_to_words[pair].discard(old_words)
            
            for i in range(len(new_words)-1):
                pair = (new_words[i], new_words[i+1])
                pair_counts[pair] += freq
                pair_to_words[pair].add(new_words)

            del frequency_table[old_words]
            frequency_table[new_words] += freq

        # Update Vocabulary
        vocab[index] = best_pair[0] + best_pair[1]
        index += 1
        merges.append(best_pair)

    return vocab, merges
    
    


    