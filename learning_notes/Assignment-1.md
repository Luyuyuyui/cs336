# CS336 Assignment 1：BPE Tokenizer Training 实验报告

## 1. BPE Training Implementation

### 1.1 Algorithm Overview

本实验实现了 Byte Pair Encoding（BPE）分词器训练算法。算法首先将训练语料划分为 pre-token，再将每个 pre-token 编码为 UTF-8 字节序列。初始词表包含 256 个单字节 token，并加入指定的特殊 token。

在每一轮训练中，算法统计所有 pre-token 中相邻 token pair 的加权频率，选择频率最高的 pair，将其合并为一个新 token，并将合并规则记录到有序的 merges 列表中。训练持续进行，直到达到目标词表大小或不存在可合并的 pair。

Pair \((a,b)\) 的频率定义为：

\[
f(a,b)=\sum_w c(w)n_w(a,b)
\]

其中，\(c(w)\) 表示 pre-token \(w\) 的出现次数，\(n_w(a,b)\) 表示 pair \((a,b)\) 在该 pre-token 中出现的次数。

当多个 pair 的频率相同时，按照字节序选择较大的 pair，以保证合并规则符合规定的确定性排序要求。

### 1.2 Pre-tokenization

预分词阶段使用正则表达式将文本划分为 pre-token，并在普通预分词之前按照特殊 token 的边界分割文本。每个 pre-token 随后被转换为 UTF-8 字节序列，并通过 Counter 统计其出现频率。

该设计将文本预处理与后续 BPE 合并分离，避免不同 pre-token 之间发生合并。

### 1.3 Incremental Pair Frequency Updates

朴素 BPE 实现需要在每一轮重新统计所有 pair 的频率，并重新处理全部 pre-token，造成大量重复计算。

为了减少开销，本实验维护两个数据结构：

- `pair_freqs`：记录每个相邻 pair 的全局加权频率。
- `pair_to_words`：记录包含指定 pair 的 pre-token 集合。

每次合并时，算法只处理包含当前最高频 pair 的 pre-token。对于每个受影响的 pre-token，先减去旧序列中所有 pair 的频率贡献，再加入新序列产生的 pair 贡献，并同步更新反向索引。

该方法避免每轮重新扫描全部 pre-token。当前实现仍通过扫描 `pair_freqs` 选择最高频 pair，没有使用最大堆，因此 pair 选择仍存在进一步优化空间。

### 1.4 Parallel Pre-tokenization

为了加速较大语料的预分词，本实验进一步采用多进程处理。首先根据特殊 token 的边界划分语料文件，再将不同文本块分配给多个工作进程。每个进程独立执行预分词并返回局部 Counter，主进程最终汇总各进程的统计结果。

通过比较单进程与多进程得到的 pre-token 频率统计，验证并行处理是否保持统计结果一致。

## 2. TinyStories Experiment

### 2.1 Experimental Setup

| 项目 | 设置 |
|---|---|
| Dataset | TinyStories |
| Vocabulary size | 10,000 |
| Special token | `<\|endoftext\|>` |
| Hardware | [填写 CPU、内存及服务器配置] |
| Number of processes | [填写实际进程数] |

### 2.2 Training Performance

正式训练完成后，记录以下指标：

| 指标 | 实验结果 |
|---|---|
| Training time | [填写实际训练时间] |
| Peak memory usage | [填写实际峰值内存] |
| Final vocabulary size | [填写实际词表大小] |
| Number of merge operations | [填写实际合并次数] |

训练时间与内存使用量用于评估当前 BPE 实现处理大规模文本语料的效率。由于当前实现仍使用 `max()` 搜索最高频 pair，其整体性能还可能受到 pair 选择阶段的限制。

### 2.3 Longest Token Analysis

训练完成后，对词表中的普通 token 按字节长度进行排序，找到最长 token，并排除特殊 token 的影响。

| 指标 | 实验结果 |
|---|---|
| Longest token | [填写实际 token] |
| Token ID | [填写实际 ID] |
| Length in bytes | [填写实际字节长度] |

最长 token 可能对应高频词、常见短语或训练语料中重复出现的文本模式。需要结合实际 token 内容判断其形成原因，而不能仅根据 token 长度推断其频率或语义。

## 3. Profiling and Optimization

为了分析 BPE 训练的性能瓶颈，将训练过程拆分为以下阶段：

1. 文件读取与预分词；
2. 初始 pair 频率统计；
3. 最高频 pair 搜索；
4. 受影响 pre-token 的合并；
5. pair 频率和反向索引更新。

记录各阶段的耗时后，可以判断多进程预分词和增量更新是否有效降低了总运行时间。

当前实现通过 `pair_to_words` 减少每轮合并时需要处理的 pre-token 数量，但最高频 pair 搜索仍需遍历当前所有 pair。因此，如果 Profiling 显示该阶段占比较高，可以进一步考虑最大堆及懒删除等优化方法。

| 阶段 | 耗时 | 占总时间比例 |
|---|---:|---:|
| Pre-tokenization | [填写] | [填写] |
| Initial pair counting | [填写] | [填写] |
| Best-pair selection | [填写] | [填写] |
| Merge and incremental updates | [填写] | [填写] |

## 4. OpenWebText Experiment

### 4.1 Experimental Setup

| 项目 | 设置 |
|---|---|
| Dataset | OpenWebText |
| Vocabulary size | 32,000 |
| Special token | `<\|endoftext\|>` |
| Hardware | [填写实际配置] |
| Number of processes | [填写实际进程数] |

### 4.2 Longest Token Analysis

记录 OpenWebText 训练后得到的最长普通 token，并分析其内容。

| 指标 | 实验结果 |
|---|---|
| Longest token | [填写实际 token] |
| Token ID | [填写实际 ID] |
| Length in bytes | [填写实际字节长度] |
| Training time | [填写实际训练时间] |
| Peak memory usage | [填写实际峰值内存] |

OpenWebText 包含网页文本，其词表可能学习到常见英文词语、标点组合、网址片段或其他重复文本模式。上述内容只是分析方向，最终结论应依据实际训练得到的 token。

### 4.3 Comparison with TinyStories

TinyStories 与 OpenWebText 的文本分布和词表大小不同，因此二者学习到的合并规则也可能不同。

TinyStories 主要包含故事文本，其高频模式可能集中于叙事性词汇、常见短语及对话表达；OpenWebText 包含更广泛的网页内容，可能具有更加多样的词汇、符号及格式模式。

比较两种 tokenizer 时，应结合实际最长 token、词表内容及训练时间进行分析。由于词表大小分别为 10,000 和 32,000，不能仅根据最长 token 的差异判断某个 tokenizer 更优。

## 5. Conclusion

本实验实现了基于字节的 BPE 训练算法，并通过增量维护 pair 频率和反向索引减少了合并阶段的重复计算。在此基础上，多进程预分词为较大规模语料的训练提供了并行处理能力。

后续性能分析将重点关注最高频 pair 搜索的时间开销，并结合 TinyStories 与 OpenWebText 的实验结果，分析不同文本分布对学习到的 token 及训练效率的影响。

本报告中的训练时间、峰值内存、最长 token 和 Profiling 数据应以实际运行结果为准。
