---
layout: post
title: Python的super函数直观理解
date: '2021-03-12'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/356720970
related_posts: false
toc:
  sidebar: left
---

> `super`相关的介绍文章看了无数遍，每次看得都云里雾里的，没过多久就忘了，只模糊知道跟MRO有关，但是稍微一复杂就不知道怎么回事了，本篇文章主要记录我对super的理解，尽量以简单易懂的方式介绍，如果你看完这篇文章还是没懂。。。那么很抱歉，我尽力了hhhh

## **粗暴简单的理解**

**super的作用就是执行父类的方法**，虽然这句话不完全对，但是也差不多是那么个意思了。

比如以单继承为例

```
class A:
    def p(self):
        print('A')
class B(A):
    def p(self):
        super().p()
B().p()

>>> A
```

可以看到`B().p()`其实就是执行的`A.p`

## **万能模板解题技巧**

前面介绍的是最简单的情况，那万一复杂一点，比如多继承的情况呢？这种情况我们只要知道 MRO 序列，无论super怎么变化都不怕了，记住公式就完事了。MRO 序列简单理解就是记录了各个类继承的先后顺序，看下面的例子就明白了

> MRO 的介绍可以看这篇文章： [https://python3-cookbook.readthedocs.io/zh\_CN/latest/c08/p07\_calling\_method\_on\_parent\_class.html](https://link.zhihu.com/?target=https%3A//python3-cookbook.readthedocs.io/zh_CN/latest/c08/p07_calling_method_on_parent_class.html)

我们看下面的例子

```
class A:
    def p(self):
        print('A')
class B():
    def p(self):
        print('B')
class C(A,B):
    def p(self):
        print('C')
class D(C):
    def p(self):
        print('D')
a = A()
b = B()
c = C()
d = D()
```

四个类的MRO (可以通过查看`__mro__`属性获得，例如`A.__mro__`)) 分别是：

* A: (A, object)
* B: (B, object)
* C: (C, A, B, object)
* D: (D, C, A, B, object)

什么意思呢，我们以`A`类为例，它的MRO顺序是他自己和`object`，很好理解，因为python里一切都是对象，所以你可以看到四个类的终点都是`object`。那`C`类的 MRO 也好理解，第一个顺序永远是寄几，然后按照代码顺序依次是 `A,B`，最后是`object`。

相信看到这你应该知道MRO是什么意思了吧，那`super`是怎么用的呢？

super本身其实就是一个类，`super()`其实就是这个类的实例化对象，它需要接收两个参数 `super(class, obj)`,它返回的是`obj`的MRO中`class`类的父类（可能有点绕，待会看后面的栗子就好懂了）:

* `class`：就是类，这里你可以是`A,B,C`或者`D`
* `obj`：就是一个具体的实例对象，即`a,b,c,d`。我们经常在类的`__init__`函数里看到super的身影，而且一般都是写成这个样子的`super(className, self).__init__()`，`self`其实就是某个实例化的对象。

## **举几个栗子**

看到这里如果还是没明白，咱们就多看几个例子就完事了，建议你可以打开`ipython`输入如下示例，可以更加直观感受`super`：

## **栗子1**

首先我们看看下面的命令是什么意思呢？

```
super(C, d).p()
```

前面我们说过`super`的作用是 **返回的是`obj`的MRO中`class`类的父类**,在这里就表示**返回的是`d`的MRO中`C`类的父类**：

1. 返回的是`d`的MRO：`(D, C, A, B, object)`
2. 中`C`类的父类：`A`

那么`super(C, d)`就等价于`A`,那么`super(C, d).p()`会输出`A`

## **栗子2**

下面代码结果是什么呢？

```
super(A, c).p()
```

**返回的是`c`的MRO中`A`类的父类**：

1. 返回的是`c`的MRO：`(C, A, B, object)`
2. 中A类的父类：`B`

所以最后的输出是`B`

## **最后几个栗子**

> 注意：有的类里面没有`super()`

```
class A:
    def p(self):
        print('A')
class B(A):
    def p(self):
  super().p()
        print('B')
class C(B):
    def p(self):
        print('C')
class D(C):
    def p(self):
        print('D')

d = D()
d.p()
```

这个很简单，最后只会输出`D`

那如果D类改成如下样子呢？

```
class D(C):
    def p(self):
        super().p()
        print('D')
```

很简单，我们首先写出D的MRO为 (D,C,B,A,object)，缺省状态下,super()就表示前一个父类，这里就是C类，那么`super().p()`就会调用`C`的`p`函数，但是`C.p`里没有调用`super`，所以就与`A,B`类无关了，那么最终的输出就是`C,D`

我们再看看最复杂的继承情况

```
class A:
    def p(self):
        print('A')
class B(A):
    def p(self):
        super().p()
        print('B')
class C(B):
    def p(self):
        super().p()
        print('C')
class D(C):
    def p(self):
        super().p()
        print('D')
d = D()
d.p()
```

这里建议你自己在纸上画一下

![](/assets/img/marsggbo/2021-03-12-Python的super函数直观理解/b717e469.jpg)

所以最终结果是 `A B C D`

## **多继承**

如果一个类继承多个类，原理一样，我们只要知道了MRO就能知道执行顺序了，比如下面的例子：

```
class A:
    def __init__(self):
        print('A')
class B:
    def __init__(self):
        print('B')
class C(A,B):
    def __init__(self):
        super(C,self).__init__()
        print('C')
class D(B,A):
    def __init__(self):
        super(B,self).__init__()
        print('D')
print(C.__mro__)
print(D.__mro__)
print('initi C:')
c = C()
print('initi D:')
d = D()
```

输出结果为

```
(<class '__main__.C'>, <class '__main__.A'>, <class '__main__.B'>, <class 'object'>)
(<class '__main__.D'>, <class '__main__.B'>, <class '__main__.A'>, <class 'object'>)
initi C:
A
C
initi D:
A
D
```

结果解释，`C` 初始化后的打印结果应该很好理解，首先需要执行 `super(C,self)` 的`_init_` 函数，根据`C` 的MRO可以知道`super(C,self)` 返回的是`A` ，所以可以看到最后的确是先打印出了A，然后是C

我们再看`D` 初始化的结果，注意执行的是`super(B,self)` ，根据`D` 的MRO我们可以知道返回的是`A` 所以打印的结果也是 A，然后打印出了D

## **总结**

对于无论有多少个`super(class, obj)`，我们首先需要知道`obj`的MRO是什么，然后找到`class`的前一个class是什么就好了。

### **微信公众号：AutoML机器学习**

http://weixin.qq.com/r/HD8gOHzEmiHlrThb92oO (二维码自动识别)

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
**2021-03-12 21:40:18**