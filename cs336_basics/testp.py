from collections import Counter
import regex as re

from cs336_basics.bpe import (
    PAT,
    word2bytes,
    parallel_pretokenize,
)

path = "tests/fixtures/corpus.en"
special_tokens = ["<|endoftext|>"]

# 单进程统计
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

pattern = "|".join(
    re.escape(token)
    for token in sorted(special_tokens, key=len, reverse=True)
)

segments = re.split(pattern, text)
single_counts = Counter()

for segment in segments:
    for match in PAT.finditer(segment):
        single_counts[word2bytes(match.group())] += 1

# 多进程统计
parallel_counts = parallel_pretokenize(
    path,
    special_tokens,
    num_processes=4,
)

assert single_counts == parallel_counts
print("Single-process and multiprocessing counts match!")