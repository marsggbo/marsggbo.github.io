---
layout: post
title: Latex表格太宽怎么办？用 sidewaystable ！
date: '2022-09-29'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/569475129
related_posts: false
toc:
  sidebar: left
---

```
\begin{sidewaystable}[!tbp]
    \caption{Example of SideWaysTable}
    \centering
    \begin{tabular}{c|c|c|c|c|c|c|c|c|c|c|c|c}
        \hline
        AAAAA & BBBB & CCCC & DDDD & EEEE & FFFF & GGGG & HHHH & IIII & JJJJ & KKKK & LLLL & MMMM \\
        AAAAA & BBBB & CCCC & DDDD & EEEE & FFFF & GGGG & HHHH & IIII & JJJJ & KKKK & LLLL & MMMM \\
        AAAAA & BBBB & CCCC & DDDD & EEEE & FFFF & GGGG & HHHH & IIII & JJJJ & KKKK & LLLL & MMMM \\
        \hline
    \end{tabular}
    \label{table3:sidewaystable_example}
\end{sidewaystable}
```

效果图如下：

![](/assets/img/marsggbo/2022-09-29-Latex表格太宽怎么办用-sidewaystable/de07a442.jpg)