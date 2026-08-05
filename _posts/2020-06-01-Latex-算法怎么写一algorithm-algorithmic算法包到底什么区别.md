---
layout: post
title: "Latex 算法怎么写？（一）：algorithm, algorithmic算法包到底什么区别？"
date: 2020-06-01
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/145195565
related_posts: false
toc:
  sidebar: left
---

> 原文： [https://tex.stackexchange.com/questions/229355/algorithm-algorithmic-algorithmicx-algorithm2e-algpseudocode-confused](https://link.zhihu.com/?target=https%3A//tex.stackexchange.com/questions/229355/algorithm-algorithmic-algorithmicx-algorithm2e-algpseudocode-confused)

## **概括版本**

* **algorithm** - float wrapper for algorithms.
* **algorithmic** - first algorithm typesetting environment.
* **algorithmicx** - second algorithm typesetting environment.
* **algpseudocode** - layout for algorithmicx.
* **algorithm2e** - third algorithm typesetting environment.

我结合`algpseudocode`使用`algorithmicx`，因为他们比`algorithmic`高级。另外`algorithmicx`提供了和`algorithm2e`相同的功能，但是它的语法更加简洁易懂。

## **絮絮叨叨版本**

## **1. `algorithm`**

`algorithm`是 算法的**float warpper**，类似于`table`, `figure`这样的们命令，你可以在你的表格/图形上加一个数字，防止它被分成两页。**[官方文档](https://link.zhihu.com/?target=http%3A//mirror.ox.ac.uk/sites/ctan.org/macros/latex/contrib/algorithms/algorithms.pdf)**说明如下：

> When placed within the text without being encapsulated in a floating environment `algorithmic` environments may be split over a page boundary, greatly detracting from their appearance. In addition, it is useful to have algorithms numbered for reference and for lists of algorithms to be appended to the list of contents. The `algorithm` environment is meant to address these concerns by providing a floating environment for algorithms.

例子

```
\begin{algorithm}
    \caption{Algorithm caption}
    \label{alg:algorithm-label}
    \begin{algorithmic}
        ... Your pseudocode ...
    \end{algorithmic}
\end{algorithm}
```

## **2. `algorithmic`**

事先已经定义好一些常用的命令语句，有如`IF`,`WHILE`等。需要注意的是所有命令语句必须大写。官方文档说明如下：

> The `algorithmic` environment provides an environment for describing algorithms and the `algorithm` environment provides a “float” wrapper for algorithms (implemented using `algorithmic` or some other method at the users’s option). The reason for two environments being provided is to allow the user maximum flexibility.

## **3. `algorithmicx`**

这个可以理解成`algorithmic`的升级版，他可以让我们自定义一些命令，这个是`algorithmic`没有的功能。如果你不需要自定义命令，使用`algorithmic`就好了。另外因为二者的语法很相似,只有部分语法和细节有略微差异，可见下面的例子。官方文档说明如下：

> The package `algorithmicx` itself doesn’t define any algorithmic commands, but gives a set of macros to define such a command set. You may use only algorithmicx, and define the commands yourself, or you may use one of the predefined command sets

```
\begin{algorithm}
    \caption{Euclid’s algorithm}
    \label{euclid}
    \begin{algorithmic}[1] % The number tells where the line numbering should start
        \Procedure{Euclid}{$a,b$} \Comment{The g.c.d. of a and b}
            \State $r\gets a \bmod b$
            \While{$r\not=0$} \Comment{We have the answer if r is 0}
                \State $a \gets b$
                \State $b \gets r$
                \State $r \gets a \bmod b$
            \EndWhile\label{euclidendwhile}
            \State \textbf{return} $b$\Comment{The gcd is b}
        \EndProcedure
    \end{algorithmic}
\end{algorithm}
```

## **4. `algpseudocode`**

这个其实就是`algorithmicx`的**layout**(不知道咋翻译。。)， 它试图尽可能类似于`algorithmic`。 还有其他**layout**，例如：

* `algcompatible`(与`algorithmic`包完全兼容)，
* `algpascal`(目标是创建一个格式化的Pascal程序，可以用一些基本的替换规则将`Pascal`程序转换成`algpascal`算法描述)。
* `algc`(就像`algpascal`一样，但是为c语言设计的。）

官方文档说明如下：

> If you are familiar with the `algorithmic` package, then you’ll find it easy to switch. You can use the old algorithms with the `algcompatible` layout, but please use the `algpseudocode` layout for new algorithms. To use `algpseudocode`, simply use `\usepackage{algpseudocode}`. You don’t need to manually load the `algorithmicx` package, as this is done by `algpseudocode`.

总的来说就是如果你要是用这个包，直接`\usepackage{algpseudocode}`就可以了,因为它会自动导入`algorithmicx`，然后语法直接用`algorithmicx`的语法就行了。

## **5. `algorithm2e`**

这是另外一个和`algorithmic`和`algorithmicx`类似的算法包，官方文档说明如下：

> `Algorithm2e` is an environment for writing algorithms in LaTeX2e. An `algorithm` is defined as floating object like figures. It provides macros that allow you to create different sorts of key words, thus a set of predefined key words is given. You can also change the typography of the keywords.

例子

```
\begin{algorithm}[H]
    \SetAlgoLined
    \KwData{this text}
    \KwResult{how to write algorithm with \LaTeX2e }
    initialization\;
    \While{not at end of this document}{
        read current\;
        \eIf{understand}{
            go to next section\;
            current section becomes this one\;
            }{
            go back to the beginning of current section\;
        }
    }
\caption{How to write algorithms}
\end{algorithm}
```

### **微信公众号：AutoML机器学习**

**![](/assets/img/marsggbo/2020-06-01-Latex-算法怎么写一algorithm-algorithmic算法包到底什么区别/228f26c5.jpg)**

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
**2020-06-01 23:21:53**