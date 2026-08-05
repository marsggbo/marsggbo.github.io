---
layout: post
title: "Megatron-Deepspeed：用Wikipedia数据集训练GPT实战笔记"
date: '2023-11-26'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/668913089
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/668913089>

更详细的查看GitHub

[https://github.com/marsggbo/Megatron-DeepSpeed/blob/main/tutorials/gpt2\_wikipedia.md](https://link.zhihu.com/?target=https%3A//github.com/marsggbo/Megatron-DeepSpeed/blob/main/tutorials/gpt2_wikipedia.md)

## **一. 下载数据集**

1. 下载Wikipedia压缩数据集（[enwiki-latest-pages-articles.xml.bz2](https://link.zhihu.com/?target=https%3A//dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2)）
2. 使用[wikiextractor](https://link.zhihu.com/?target=https%3A//github.com/attardi/wikiextractor)工具将数据集解压缩

```bash
pip install wikiextractor
python -m wikiextractor.WikiExtractor --json enwiki-latest-pages-articles.xml.bz2
```

解压缩后会得到一个文件夹`text`，结构如下：

```
text
├── AA
  ├── wiki_00
  ├── wiki__01
  ├── ...
├── AB
├── AC
├── AD
├── AE
├── ...
├── GD
└── GE
```

文件夹包含多个子文件夹，每个子文件夹包含多个json格式的数据集，即`wiki_00`其实是json格式的文件

3. 对解压后的数据集做预处理

我们在训练GPT的时候，解压后的数据集还不能直接拿来用，我们还需要用Megatron-Deepspeed提供的 [tools/preprocess\_data.py](https://zhuanlan.zhihu.com/p/668913089/Megatron-DeepSpeed/tools/preprocess_data.py) 对`text`目录下的数据集做预处理，最终会得到两个二进制文件，后缀分别是`bin`和`idx`。

不过`tools/preprocess_data.py`只能对单个的json文件做处理，而第二步中我们有几十万个json文件，这个该怎么办呢？一种处理办法就是把第三步中的所有json文件合并到一个json文件中去，最后再对后并后的文件做预处理就可以了。在于处理之前，你需要先运行下面大命令下载GPT相关的文件，这主要是用来预处理的

```bash
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
```

下载好后，执行下面的代码即可

```bash
#!/bin/bash  
  
# 设置ROOT路径  
ROOT="/data/personal/nus-hx/Wikipedia/text"  

# 检查是否存在 wiki_all.json 文件，如果存在则删除  
if [ -f "$ROOT/wiki_all.json" ]; then  
    rm "$ROOT/wiki_all.json"  
fi  
  
# 创建一个空的 wiki_all.json 文件  
touch "$ROOT/wiki_all.json"
  
# 遍历ROOT路径下所有的文件  
find $ROOT -type f -name "*" -print0 | while IFS= read -r -d $'\0' file; do  
    # 将所有文件内容追加到wiki_all.json文件中  
    cat "$file" >> "$ROOT/wiki_all.json"  
done  

cd /path/to/Megatron-Deepspeed
python tools/preprocess_data.py \
--input "$ROOT/wiki_all.json" \
--output-prefix my-gpt2 \
--dataset-impl mmap \
--tokenizer-type GPT2BPETokenizer   \
--append-eod  \
--vocab-file gpt2-vocab.json \
--merge-file gpt2-merges.txt  \
--workers 16 \
--partitions 16
```

参考：[https://github.com/NVIDIA/Megatron-LM/issues/117](https://link.zhihu.com/?target=https%3A//github.com/NVIDIA/Megatron-LM/issues/117)

## **二. 运行代码**

最好基于文章最开头给的链接运行代码，因为官方的代码有bug，跑不起来

```
bash ./examples/pretrain_gpt.sh
```