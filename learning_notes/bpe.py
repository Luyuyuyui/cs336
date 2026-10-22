"""
def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

1.初始化词表：理解 bytes 和 token ID 的映射
    def init_vocab() -> dict[int, bytes]:
2.预分词：正则切分文本，统计 pre-token 频率
    def pretokenizer(text: str) -> list[str]:
3.统计 pair 频率：找到相邻 token 的出现次数
    def count_pairs(tokens: list[str]) -> Counter:
4.实现 merge：将选中的 pair 合并成新 token
    def merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
5.完成 BPE 训练循环：不断更新词表和 merges
"""


"""
1.初始化词表
"""
def init_vocab():
    vocab={}
    for i in range(256):
        vocab[i] = bytes([i])
        # bytes([i]) 将整数 i 转换为单字节的 bytes 对象
    return vocab
vocab = init_vocab()


"""
2.预分词
"""

import regex as re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenizer(text):
    return re.findall(PAT, text)

# text="123123.This is a test.你好！luyu! my first BPE tokenizer.这是我的第一个分词器。"
text="hello hello hello  wolrd"
print(pretokenizer(text))
#保存每种pre token出现的次数

def count_pretokens(text):
    pre_tokens = pretokenizer(text)
    return Counter(pre_tokens)

print(count_pretokens(text))

#字符串转字节
def word2bytes(text):
    res=[]
    for byte in text.encode('utf-8'):
        res.append(bytes([byte]))
    return tuple(res)

print(word2bytes(text))


"""
3. 统计 pair 频率
"""

from collections import defaultdict
# 不需要先判断 key 是否存在

def count_pairs(word_freqs):
    pair_freqs=defaultdict(int)

    for word,freq in word_freqs.items():
        tokens=word2bytes(word)
        # 构造pair并统计频率
        for i in range(len(tokens)-1):
            pair=(tokens[i],tokens[i+1])
            pair_freqs[pair]+=freq

        best_pair=max(pair_freqs,
                    key=lambda pair:(pair_freqs[pair],pair))
    return pair_freqs,best_pair

# word_freqs = {
#     (b'a', b'b', b'c'): 5,
#     (b'a', b'b'): 3,
# }

print(dict(count_pairs(word_freqs)))


"""
4.合并高频pair
"""

def merge_pair(word, pair):
    res=[]
    i = 0

    while i<len(word):
        if (word[i],word[i+1])==pair and i<len(word)-1:
            res.append(word[i]+word[i+1])
            i+=2
        else:
            res.append(word[i])
            i+=1
    return tuple(res)

