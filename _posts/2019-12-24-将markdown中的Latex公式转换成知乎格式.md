---
layout: post
title: "将markdown中的Latex公式转换成知乎格式"
date: 2019-12-24
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/99057715
related_posts: false
toc:
  sidebar: left
---

我们平时在写markdown时常常会用到latex公式，然而直接将markdown上传至知乎的话，知乎并不能正常解析公式，因为他还无法识别$$。

通过分析知乎网站代码可以看到知乎的处理方式是这样的：

假设你的latex代码如下：

```
$$\frac{a}{b}$$
```

知乎的显示数学代码的方式是转换成html中的`img`标签：

```
<img src="https://www.zhihu.com/equation/tex=\frac{a}{b}" eeimg="1">
```

当然有时它也会把latex公式转化成URL编码格式，即我们有时会看到有的URL中会出现%这些符号，这些其实就是转码之后的结果。

好，话不多说，直接送上代码。

BTW, 代码没有使用正则表达式（因为还不熟。。。），如果有哪些小伙伴用正则表达式实现了的，麻烦分享一下，谢谢！

> 源码地址：[zhihu\_markdown\_converter](https://link.zhihu.com/?target=https%3A//github.com/marsggbo/zhihu_markdown_converter)

### **MARSGGBO♥原创** **如有意合作，欢迎私戳** **邮箱:marsggbo@foxmail.com** **2019-12-24 10:12:42**