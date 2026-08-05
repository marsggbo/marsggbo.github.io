---
layout: post
title: Latex表格太长，一页放不下怎么办？用 longtable
date: '2022-09-29'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/569424468
related_posts: false
toc:
  sidebar: left
---
{% raw %}
```
\usepackage{longtable}


\begin{longtable}{l|c|c}
% \centering
\caption[Short Caption]{Full Caption Content.}
\label{table:longtable_example} \\

% 下面是表头
\hline Col1 & Col2 & Col3 \\  \hline 
\endfirsthead

% 下面数字3的意思是表格的列数
\multicolumn{3}{c}%
{{\bfseries \tablename\ \thetable{} -- continued from previous page}} \\
\hline Col1 & Col2 & Col3 \\  \hline  
% 注意这里把表头复制了一遍，因为在新的页面也会展示一下表头，不然表格不方便阅读
\endhead

\hline \multicolumn{3}{r}{{Continued on next page}} \\ \hline
\endfoot

\hline \hline
\endlastfoot

% 下面就是真正的表格数据了，注意不用再写表头了
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 \\
d1 & d2 & d3 
\end{longtable}
```
{% endraw %}

效果如下图，可以看到表格被换分到了两页，当然如果你的表格特别特别长，也会默认划分到多个连续的页面

![](/assets/img/marsggbo/2022-09-29-Latex表格太长一页放不下怎么办用-longtable/92607699.jpg)
