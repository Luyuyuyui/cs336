from collections import Counter,defaultdict
import regex as re
from concurrent.futures import ProcessPoolExecutor
from cs336_basics.pretokenization_example import find_chunk_boundaries



PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

"""
一些工具函数
"""
def init_vocab():
    vocab={}
    for i in range(256):
        vocab[i] = bytes([i])
    return vocab

def pretokenizer(text):
    return re.findall(PAT, text)

def word2bytes(text):
    res=[]
    for byte in text.encode('utf-8'):
        res.append(bytes([byte]))
    return tuple(res)

def count_pairs(word_freqs):
    pair_freqs=defaultdict(int)

    for word,freq in word_freqs.items():
        for i in range(len(word)-1):
            pair=(word[i],word[i+1])
            pair_freqs[pair]+=freq
    return pair_freqs

def get_best_pair(pair_freqs):
    best_pair=max(pair_freqs,
                    key=lambda pair:(pair_freqs[pair],pair))
    return best_pair

def merge_word(word,pair):
    res=[]
    i= 0
    while(i<len(word)):
        if i + 1 < len(word) and (word[i], word[i+1]) == pair:
            res.append(word[i]+word[i+1])
            i+=2
        else:
            res.append(word[i])
            i+=1

    return tuple(res)


def init_pair_stats(word_freqs):
    pair_freqs=count_pairs(word_freqs)
    pair_to_words=defaultdict(set)
    for word in word_freqs:
        for pair in set(zip(word,word[1:])):#把单词切成相邻两两组合,存入
            pair_to_words[pair].add(word)
    return pair_freqs,pair_to_words

def merge_pair_incrementally(
    best_pair,
    word_freqs,
    pair_freqs,
    pair_to_words,
):
    """
    合并 best_pair，增量维护:
      - word_freqs:     {word_tuple: freq}
      - pair_freqs:     {pair: total_freq}
      - pair_to_words:  {pair: set(word_tuple)}
    返回: 本次合并中 pair 频率发生变化的集合 (set of pairs)
    """
    affected_words=list(pair_to_words.get(best_pair,()))# 定位 word
    changes=defaultdict(int)

    for word in affected_words:
        freq=word_freqs.pop(word,0)
        if freq==0:
            continue
        old_pairs=Counter(zip(word,word[1:]))
        new_word=merge_word(word,best_pair)
        new_pairs=Counter(zip(new_word,new_word[1:]))

        #删掉旧的word的pair贡献
        for pair,cnt in old_pairs.items():
            changes[pair]-=cnt*freq
            pair_to_words[pair].discard(word)

        #加上新的word的pair贡献
        word_freqs[new_word]+=freq
        for pair,cnt in new_pairs.items():
            changes[pair]+=cnt*freq
            pair_to_words[pair].add(new_word)

    for pair,delta in changes.items():
        if delta==0:
            continue
        new_freq=pair_freqs.get(pair,0)+delta
        if new_freq<0:
            raise RuntimeError(f"Negative frequency for pair {pair}: {new_freq}")
        if new_freq==0:
            pair_freqs.pop(pair,None)

            words=pair_to_words.get(pair)
            if words is not None and not words:
                pair_to_words.pop(pair,None)
        else:
            pair_freqs[pair]=new_freq


def process_chunk(args):
    """
    输入：文件路径 + 自己的字节区间 [start, end) + 特殊 token 列表
    处理文件的一个 chunk，返回该 chunk 的 word_freqs
    思路：把一个大文件切成 N 段，每个进程读自己那段、各自统计局部词频，最后把局部 Counter 相加
    """
    input_path, start, end, special_tokens = args
    with open(input_path,"rb")as f:
        f.seek(start)
        chunk=f.read(end-start).decode("utf-8")

    if special_tokens:
        pattern="|".join(
            re.escape(token)
            for token in sorted(special_tokens,key=len,reverse=True)
        )
        seg=re.split(pattern,chunk)
    else:
        seg=[chunk]

    #对每段做预分词
    cnts=Counter()
    for s in seg:
        for match in re.finditer(PAT,s):
            cnts[word2bytes(match.group())]+=1
    return cnts

def parallel_pretokenize(
        input_path,
        special_tokens,
        num_processes=4,
):
    #没有特殊 token 的情况，直接预分词
    if not special_tokens:
        with open(input_path,"r",encoding="utf-8")as f:
            text=f.read()
        cnts=Counter()
        for word in pretokenizer(text):
            cnts[word2bytes(word)]+=1
        return cnts

    split_token=special_tokens[0].encode("utf-8")
    with open(input_path,"rb")as f:
        boundaries =find_chunk_boundaries(f,num_processes,split_token)
        tasks=[
            (input_path,start,end,special_tokens)
            for start,end in zip(boundaries[:-1],boundaries[1:])
        ]
    word_freqs=Counter()

    # 把每个 task 分给一个 worker，按提交顺序产出结果
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for local_counts in executor.map(
            process_chunk,
            tasks,
        ):
            word_freqs.update(local_counts)
    return word_freqs


           

"""
训练循环
"""

def train_bpe(input_path, vocab_size, special_tokens)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # Step 1: 初始化词表
    vocab = init_vocab()
    merges = []

    # Step 2: 添加特殊 token
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    if len(vocab) > vocab_size:
        raise ValueError("Vocabulary size is too small !")

    # Step 3: 读取语料
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Step 4: 预分词
    # 先按特殊 token 分割文本
    if special_tokens:
        pattern="|".join(
            re.escape(token)
            for token in sorted(special_tokens,key=len,reverse=True)
        )
        seg=re.split(pattern,text)
    else:
        seg=[text]

    # 统计每个分词的频率
    word_freqs = parallel_pretokenize(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=4,
    )
    # word_freqs=Counter()
    # for s in seg:
    #     for match in re.finditer(PAT, s):
    #         word = match.group()
    #         byte_word = word2bytes(word)
    #         word_freqs[byte_word] += 1

    # Step 5: 计算合并次数
    pair_freqs, pair_to_words = init_pair_stats(word_freqs)

    num_merges = vocab_size - len(vocab)

    # Step 6: BPE 训练循环
    for i in range(num_merges):
        
        # count_pairs()
        # pair_freqs = count_pairs(word_freqs)
        if not pair_freqs:
            break
        best_pair = get_best_pair(pair_freqs)

        merge_pair_incrementally(best_pair, word_freqs, pair_freqs, pair_to_words)
        # 生成新的 token 并加入 vocab
        # new_token=best_pair[0]+best_pair[1]
        # vocab[len(vocab)] = new_token

        # 将 pair 加入 merges
        merges.append(best_pair)

        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        #使用 merge_word() 更新 word_freqs
        # new_word_freqs=Counter()
        # for word,freq in word_freqs.items():
        #     new_word=merge_word(word,best_pair)
        #     new_word_freqs[new_word]+=freq

        # word_freqs=new_word_freqs

    return vocab, merges