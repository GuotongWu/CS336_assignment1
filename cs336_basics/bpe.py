import os
import time
import heapq
import regex as re
import pickle as pkl
import numpy as np
from multiprocess import cpu_count, Pool 
from collections import defaultdict
from typing import BinaryIO, Iterable, Iterator
from tqdm import tqdm

num_processes = cpu_count()
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class PairElement():
    def __init__(self, pair, cnt):
        self.pair = pair
        self.cnt = cnt
    
    def __lt__(self, other):
        return (self.cnt, self.pair) > (other.cnt, other.pair)

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

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

def pre_tokenize(chunk: str, pat_special: str) -> dict[tuple[bytes], int]:
    split_chunk = re.split(pat_special, chunk)
    d = defaultdict(int)
    for spc in split_chunk: 
        it = re.finditer(PAT, spc)
        for word in it:
            word_bytes = tuple(bytes([b]) for b in word.group(0).encode("utf-8"))
            d[word_bytes] += 1
    return d

def _pre_tokenize_worker(args) -> dict[tuple[bytes], int]:
    input_path, start, end, pat_special = args
    with open(input_path, 'rb') as f:
        f.seek(start)
        data = f.read(end - start)
    chunk_str = (
        data.decode("utf-8", errors="ignore")
            .replace('\r\n', '\n')
            .replace('\r', '\n')
    )
    return pre_tokenize(chunk_str, pat_special)

def merge_pair(w: tuple[bytes], max_pair: tuple[bytes, bytes]):
    new_w = []
    i = 0
    while i < len(w):
        if i < len(w) - 1 and (w[i], w[i+1]) == max_pair:
            new_w.append(w[i] + w[i+1])
            i += 2
        else:
            new_w.append(w[i])
            i += 1
    return tuple(new_w)

def train_bpe(input_path:str, vocab_size:int, special_tokens:list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    total_start_time = time.time()
    special_tokens.sort(key=lambda sp_token: -len(sp_token))
    PAT_SPECIAL = "|".join([re.escape(token) for token in special_tokens])

    # 1. read files
    tqdm.write("🚀 Start to read files and chunk ...")
    read_start_time = time.time()

    with open(input_path, 'rb') as f:
        boundaries = find_chunk_boundaries(
                f, 48, "<|endoftext|>".encode("utf-8"))

    read_time = time.time() - read_start_time
    tqdm.write(f"✅ Read finished! Cost time: {read_time:.4f} seconds")

    # 2. pre_tokenization
    tqdm.write("📊 Start to pre_tokenize ...")
    pre_tokenize_start_time = time.time()
    
    word_count = defaultdict(int)
    tasks = [(input_path, s, e, PAT_SPECIAL) for s, e in zip(boundaries[:-1], boundaries[1:])]

    with Pool(processes=num_processes) as pool:
        for d in tqdm(pool.imap_unordered(_pre_tokenize_worker, tasks),
                      total=len(tasks), desc="pre_tokenize", unit="chunk"):
            for k, v in d.items():
                word_count[k] += v

    pre_tokenize_time = time.time() - pre_tokenize_start_time
    tqdm.write(f"✅ Pre_tokenization finished! Cost time: {pre_tokenize_time: .4f} seconds")        

    # 3. word count and initialization
    tqdm.write("📊 Start to count word and initialize ...")
    initial_start_time = time.time()

    pair_count = defaultdict(int)
    pair_to_word = defaultdict(list)
    for word, cnt in word_count.items():
        if len(word) < 2:
            continue
        pairs = [(w1, w2) for w1, w2 in zip(word[:-1], word[1:])]
        for pair in pairs:
            pair_count[pair] += cnt
            pair_to_word[pair].append(word)

    pq_pair_count = [PairElement(pair, val) for pair, val in pair_count.items()]
    heapq.heapify(pq_pair_count)

    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for spi in special_tokens:
        vocab[len(vocab)] = spi.encode("utf-8")

    merges = []
    
    initial_time = time.time() - initial_start_time
    tqdm.write(f"✅ Word count and initialization finished! Cost time: {initial_time: .4f} seconds")        
    
    # 4. bpe merge
    tqdm.write("🔄 Start to merge ...")
    merge_start_time = time.time()

    pbar = tqdm(total=vocab_size-len(vocab), desc="bpe merge", unit="merge", 
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") 

    while len(vocab) < vocab_size:
        max_pair_ele = heapq.heappop(pq_pair_count)
        max_pair, max_pair_count = max_pair_ele.pair, max_pair_ele.cnt
        if max_pair not in pair_count or max_pair_count != pair_count[max_pair]:
            continue

        merges.append(max_pair)
        vocab[len(vocab)] = max_pair[0] + max_pair[1]

        old_pair_count = defaultdict(int)
        new_pair_count = defaultdict(int)
        
        for w in pair_to_word[max_pair]:
            if w not in word_count:
                continue
            wcnt = word_count[w]

            for old_pair in zip(w[:-1], w[1:]):
                old_pair_count[old_pair] += wcnt

            new_w = merge_pair(w, max_pair)
            
            for new_pair in zip(new_w[:-1], new_w[1:]): 
                pair_to_word[new_pair].append(new_w)
                new_pair_count[new_pair] += wcnt
            word_count[new_w] = wcnt
            del word_count[w]
        
        all_pairs = set(old_pair_count.keys() | new_pair_count.keys())
        for pair in all_pairs:
            delta = new_pair_count.get(pair, 0) - old_pair_count.get(pair, 0)
            if delta != 0:
                new_count = pair_count.get(pair, 0) + delta
                if new_count > 0:
                    pair_count[pair] = new_count
                    heapq.heappush(pq_pair_count, PairElement(pair, new_count))
                else:
                    pair_count.pop(pair, None)
        pair_count.pop(max_pair, None)

        pbar.update(1)
    
    pbar.close()
    merge_time = time.time() - merge_start_time
    tqdm.write(f"✅ Merge finished! Cost time: {merge_time: .4f} seconds")        

    total_time = time.time() - total_start_time
    tqdm.write(f"✅ All tasks have been finised! Cost time: {total_time: .4f} seconds")        

    return vocab, merges


class BPE():
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], 
                 special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.vocab_inv = {word: i for i, word in vocab.items()}

        self.merges = merges
        self.merges_rank = {merges: i for i, merges in enumerate(merges)}

        
        self.special_tokens = special_tokens
        if special_tokens:
            self.special_tokens.sort(key=lambda sp_token: -len(sp_token))
            for sp_token in self.special_tokens:
                sp_token = sp_token.encode("utf-8")
                if sp_token not in self.vocab_inv:
                    index = len(self.vocab)
                    self.vocab[index] = sp_token
                    self.vocab_inv[sp_token] = index
            self.pat_special = "|".join([re.escape(token) for token in self.special_tokens])

        self.inf = 1 << 60

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None=None):
        with open(vocab_filepath, 'rb') as vf:
            raw_vocab = pkl.load(vf)

        vocab = {int(k): (v.encode("utf-8") if isinstance(v, str) else v)
                for k, v in raw_vocab.items()}

        with open(merges_filepath, 'rb') as mf:
            raw_merges = pkl.load(mf)

        merges = []
        for a, b in raw_merges:
            merges.append((
                a.encode("utf-8") if isinstance(a, str) else a,
                b.encode("utf-8") if isinstance(b, str) else b
            ))
        return cls(vocab, merges, special_tokens)

    @classmethod
    def from_vocab_merges_files(cls, vocab_merges_filepath: str, special_tokens: list[str] | None=None):
        with open(vocab_merges_filepath, 'rb') as f:
            vocab, merges = pkl.load(f)
        return cls(vocab, merges, special_tokens)

    def encode_token(self, token: str):
        if len(token) == 0:
            return []
        
        if token.encode("utf-8") in self.vocab_inv:
            return [self.vocab_inv[token.encode("utf-8")]]
        
        token = tuple([bytes([b]) for b in token.encode("utf-8")])
        pair_state = {}
        for pair in zip(token[:-1], token[1:]):
            pair_state[pair] = self.merges_rank.get(pair, self.inf)
        
        while True:
            high_pair = min(pair_state.keys(), key=lambda pair: pair_state[pair])
            rank = pair_state[high_pair]
            if rank == self.inf:
                break
            token = merge_pair(token, high_pair)
            pair_state = {}
            for new_pair in zip(token[:-1], token[1:]):
                pair_state[new_pair] = self.merges_rank.get(new_pair, self.inf)
        
        return [self.vocab_inv[word] for word in token]

    def encode(self, text: str) -> list[int]:
        # 1. special tokens
        sp_tokens_list = []
        if self.special_tokens:
            sp_tokens_iter = re.finditer(self.pat_special, text)
            for sp_token in sp_tokens_iter:
                sp_token = sp_token.group(0)
                sp_tokens_list.append(self.encode_token(sp_token)) 
        # 2. other tokens
        if self.special_tokens:
            chunks = re.split(self.pat_special, text)
        else:
            chunks = [text]
        chunks_list = []
        for chunk in chunks:
            split_tokens_list = []
            split_tokens_iter = re.finditer(PAT, chunk)
            for token in split_tokens_iter:
                token = token.group(0)
                split_tokens_list.extend(self.encode_token(token))
            chunks_list.append(split_tokens_list)
        # 3. merge 
        results_list = []
        for i in range(len(sp_tokens_list)):
            results_list.extend(chunks_list[i])
            results_list.extend(sp_tokens_list[i])
        if len(chunks_list) > len(sp_tokens_list):
            results_list.extend(chunks_list[-1])

        return results_list

    def encode_dictble(self, iterable: Iterable[str]) -> Iterator[int]:
        for encoded in tqdm(map(self.encode, iterable)):
            yield from encoded 

    def decode(self, ids: list[int]) -> str:
        byte_sequence = b''.join(self.vocab[index] for index in ids)
        return byte_sequence.decode("utf-8", errors='replace')



if __name__ == "__main__":
    vocab_merges_filepath = "/home/wgt/projects/assignment1-basics/data/TinyStoriesV2-GPT4-train-vocab-merges-10000.pkl"
    special_tokens = ["<|endoftext|>"]
    # special_tokens = None
    tokenizer = BPE.from_vocab_merges_files(vocab_merges_filepath, special_tokens)
    input_path = "/home/wgt/projects/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    
    all_ids = []
    with open(input_path, "r") as f:
        for _id in tokenizer.encode_iterable(f):
            all_ids.append(_id)
    name = input_path.split(".")[0].split("/")[-1]
    all_ids = np.array(all_ids, dtype=np.uint16)
    
    saved_path = os.path.join("/home/wgt/projects/assignment1-basics/data", f"{name}-encoding.npy")
    np.save(saved_path, all_ids)
    
    print("sucessfully encode!")