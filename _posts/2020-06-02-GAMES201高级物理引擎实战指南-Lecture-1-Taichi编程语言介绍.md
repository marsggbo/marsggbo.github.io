---
layout: post
title: GAMES201：高级物理引擎实战指南-Lecture 1 Taichi编程语言介绍
date: '2020-06-02'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/145392844
related_posts: false
toc:
  sidebar: left
---

> 常用链接  
>    
> GAMES 的主页 [http://games-cn.org](https://link.zhihu.com/?target=http%3A//games-cn.org) 【201相关链接在：导航栏/在线课程/GAMES201】  
> Taichi 论坛 forum.taichi.graphics/c/games201/ ； 记得在 GAMES201的版块发帖，其余板块用的是英文  
> 课程课件 [http://github.com/taichi-dev/games201/releases](https://link.zhihu.com/?target=http%3A//github.com/taichi-dev/games201/releases)  
> Taichi 主页 [http://github.com/taichi-dev/taichi](https://link.zhihu.com/?target=http%3A//github.com/taichi-dev/taichi)  
> Taichi 中文文档 [http://taichi.readthedocs.io/zh\_CN/latest](https://link.zhihu.com/?target=http%3A//taichi.readthedocs.io/zh_CN/latest)  
> 直播地址 [http://webinar.games-cn.org](https://link.zhihu.com/?target=http%3A//webinar.games-cn.org)

`Taichi`是基于python开发的高性能图形编程语言，为了解决python速度较慢的问题，特别设计了编译器，进一步提高了图形编程应用的生产力和可移植性。按Boss胡的说法是你只要装好了Taichi，别人的代码你copy过来基本上就可以直接用了，而不像OpenGL的一些你还得安装很多其他依赖库才能跑代码。

Taichi还有如下特点：

* 数据导向、并行性，megakernels
* 将数据结构和计算解耦
* Access spatially sparse tensors as if they are dense
* 支持可微编程

下面对Taichi做正式介绍。

## **1. 安装Taichi**

```
python3 -m pip install taichi
```

## **2. 初始化**

在执行任何Taichi操作之前都需要使用对Taichi做初始化操作,例如：

```
import taichi as ti
ti.init(arch=ti.cuda)
```

上面的`arch`参数可以制定如下几种backend：

* `ti.cpu`(默认值)
* `ti.x64/arm/cuda/opengl/metal`
* `ti.gpu`

## **3. 数据类型**

Taichi先支持如下几种

* Signed integers: `ti.i8 / i16 / i32 / i164`
* Unsigned integers: `ti.u8 / u16 / u32 / u64`
* Float-point numbers: `ti.f32 / f64`

`ti.i32`和`ti.f32`是最常用的数据类型，目前的布尔类型使用`ti.i32`表示。

## **4. Tensors**

Tensor也就是**张量**，它是Taichi里的第一公民（first-citizen）。其实在Pytorch和TensorFlow其实也用了这个概念，不过在Taichi有丢丢不一样：

* Tensor其实是一个多维的数组（arrays)
* Tensor里的一个元素可以是标量(scalar),向量(vector)，也可以是一个矩阵(matrix),但是在Pytorch和TensorFlow里，一个tensor里每一个元素就只是一个标量，这个在下面的代码示例可以体会到区别。
* Tensor里的元素只能通过`a[i,j,k]`这样的语法来访问，像`a[i][j][k]`这样我通过实验发现貌似只能访问tensor某个元素内的某个子元素，具体看下面的例子
* 越界访问是未定义的行为
* （高级功能）Tensor可以使空间稀疏的

举例子：

```
import taichi as ti
ti.init()
a = ti.var(dt=ti.f32, shape=(4,3))
b = ti.Vector(3, dt=ti.f32, shape=(2,2))
c = ti.Matrix(2,2, dt=ti.f32, shape=(4,3))

loss = ti.var(dt=ti.f32, shape=())

a[2,2] = 1
print(f"a[2,2]={a[2,2]}")

b[1,1] = [0,2,4]
print(f"b[1,1]={b[1,1][0],b[1,1][1],b[1,1][2]}")

loss[None] = 3
print(f"loss[None] = {loss[None]}")

>>>
a[2,2]=1.0
b[1,1]=(0.0, 2.0, 4.0)
loss[None] = 3.0
```

上面的变量`a`你不要把它理解成是一个标量，其实它是一个tensor，只不过它的每一个元素是标量而已。具体来说`a`的维度是$4\times3$，然后我们可以用`a[2,2]`来对指定位置赋值;

同理`b`也是一个大小为$2\times2$的tensor，只不过它的每一个元素是一个长度为3的vector。上面代码中通过`b[1,1]=[0,2,4]`来对[1,1]位置上的向量元素赋值，那么最后如何访问这个向量呢？如上面的代码所示，首先使用`a[i,j]`来定位到向量元素，然后再像`a[0]`来访问向量里某个具体的元素。

变量`c`是一个大小为$4\times3$的tensor，其中的每一个元素是一个大小为$2\times2$的矩阵。

变量`loss`是一个只有一个元素的tensor,即0d的tensor，所以`shape=()`,另外需要注意的是在访问loss的值(读/写)时需要`loss[None]=0`,因为我们都知道python语法里如果直接直接`loss=0`，那么这个`loss`就不再是原来的`loss`了。

## **5. Kernels**

在Taichi里，`kernel`简单理解就是用来计算的东西。Taichi的kernel语言是**compiled, statically-typed, lexically-scoped,parallel and differentiable**

你必须使用`@ti.kernel`对函数装饰后才能使用taichi的kernel功能。而且kernel的参数和返回值必须带有类型提示，例如

```
@ti.kernel
def hello(i: ti.i32):
 a = 40
 print("Hello world!", a+i)

hello(2) # "Hello world! 42"

@ti.kernel
def calc(c: ti.i32): -> ti.i32
 c += 1
 return c
```

## **6. Functions**

taichi里的functions可以被kernel或者其他的function调用，但是function不能调用kernel,kernel不能调用kernel。一个比较方便理解的方法就是`ti.func`类似于cuda编程里的local function，而`ti.kernel`则是global function。

例子：

```
@ti.func
def triple(x):
 return x * 3

@ti.kernel
def triple_array(a):
 for i in range(10):
  a[i] = triple(a[i])
```

## **7. 数学函数**

## **7.1 scalar math**

![](/assets/img/marsggbo/2020-06-02-GAMES201高级物理引擎实战指南-Lecture-1-Taichi编程语言介绍/3f90d171.jpg)

## **7.2 Matrices and linear algebra**

![](/assets/img/marsggbo/2020-06-02-GAMES201高级物理引擎实战指南-Lecture-1-Taichi编程语言介绍/6d1b1fab.jpg)

## **8. for-loops**

Taichi里有两种loop：

* **Range-for loops**：这个其实和python里的for循环没什么区别，Range-for loops可以内嵌；
* **Struct-for loops**：因为Taichi支持稀疏化张量，而这个for循环就是为了解决稀疏张量的读取问题。

## **8.1 range-for loops**

For循环里的最外层在taichi kernel里会自动并行化。例如下面第一个例子中的for循环会自动并行化，而第二个例子中由于有一个if判断语句，所以后面的for循环就不是并行化的了。

```
@ti.kernel
def foo():
 for i in range(10): # Parallelized :-)
 ...
 
@ti.kernel
def bar(k: ti.i32):
 if k > 42:
  for i in range(10): # Serial :-(
   ...
```

## **8.2 Struct-for loops**

```
import taichi as ti

ti.init(arch=ti.gpu)

n = 320

pixels = ti.var(dt=ti.f32, shape=(n * 2, n))

@ti.kernel
def paint(t: ti.f32):
 for i, j in pixels:# Parallized over all pixels
  pixels[i, j] = i * 0.001 + j * 0.002 + t
paint(0.3)
```

Struct-for loops会遍历如下所有的tensor坐标：(0, 0), (0, 1), (0, 2), ..., (0, 319), (1, 0), ..., (639, 319)

## **9. 原子操作(Atomic operations)**

在Taichi中，augmented assignments (比如 x[i] += 1) 会自动使用原子操作。

看下面的例子：

```
@ti.kernel
def sum():
 for i in x:
  # Approach 1: OK
  total[None] += x[i]
  
  # Approach 2: OK
  ti.atomic_add(total[None], x[i])
  
  # Approach 3: Wrong result (the operation is not atomic.)
  total[None] = total[None] + x[i]
```

可以看到上面三种操作里只有前两种是对的，最后一种是会有潜在问题的，因为它首先会先读取`total[None]`，然后再做加法操作，但是很有可能你在读取的时候已经有其他的县城对这个作了修改，换句话说第三种方式不是原子操作，所以是有问题的。

## **10. Taichi-scope v.s. Python-scope**

> Taichi-scope: 任何使用`ti.kernel`和`tf.func`装饰都属于taichi scope，该scope内的代码会使用Taichi编译器编译，然后可以运行在并行设备上。  
>    
>  Python-scope： taichi-scope以外的就是python scope

```
import taichi as ti
ti.init()
a=ti.var(dt=ti.f32,shape=(42,63)) # A tensor of 42x63 scalars
b=ti.Vector(3,dt=ti.f32,shape=4)#A tensor of 4x 3D vectors
C=ti.Matrix(2,2,dt=ti.f32,shape=(3,5)) # A tensor of 3x5 2x2 matrices
@ti.kernel
def foo():
 a[3,4]=1
 print('a[3,4]=',a[3,4])
 #”a[3,4]=1.000000”
 
 b[2]=[6,7,8]
 print('b[0]=',b[0],',b[2]=',b[2])
 #”b[0]=[[0.000000],[0.000000],[0.000000]],b[2]=[[6.000000],[7.000000],[8.000000]]”
 
 C[2,1][0,1]=1
 print('C[2,1]=',C[2,1])
 #C[2,1]=[[0.000000,1.000000],[0.000000,0.000000]]
foo()
```

## **11. Taichi 程序的阶段(phase)**

一个Taichi代码可以分成如下几个阶段：

1. 初始化： `ti.init()`
2. tensor分配：`ti.var, ti.Vector, ti.Matrix`
3. 计算（启动kernels， 获取python-scope的tensors）
4. 可选： 重启Taichi系统`ti.reset()`,包括了清除内存、销毁所有变量和kernels

目前版本中一旦启动kernel或者访问python-scope下的某个tensor后，就不能再allocate tensor。

```
import taichi as ti
ti.init()

n=320
pixels=ti.var(dt=ti.f32,shape=(n*2,n))

@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2-z[1]**2,z[1]*z[0]*2])
    
@ti.kernel
def paint(t:ti.f32):
    for i,j in pixels:#Parallized over all pixels
        c=ti.Vector([-0.8,ti.cos(t)*0.2])
        z=ti.Vector([i/n-1,j/n-0.5])*2
        iterations=0
        while z.norm()<20 and iterations<50:
            z=complex_sqr(z)+c
            iterations+=1
        pixels[i,j]=1-iterations*0.02

gui=ti.GUI("JuliaSet",res=(n*2,n))
for i in range(1000000):
    paint(i*0.03)
    gui.set_image(pixels)
    gui.show()
```

亲测有效：

![](/assets/img/marsggbo/2020-06-02-GAMES201高级物理引擎实战指南-Lecture-1-Taichi编程语言介绍/60c19126.jpg)

## **12. debug mode**

使用`ti.init(debug=True, arch=ti.cpu)`可以帮助检查诸如**访问越界**等问题。

```
import taichi as ti
ti.init(debug=True, arch=ti.cpu)
a = ti.var(ti.i32, shape=(10))
b = ti.var(ti.i32, shape=(10))
@ti.kernel
def shift():
 for i in range(10):
  a[i] = b[i + 1] # Runtime error in debug mode
  shift()
```

### **微信公众号：AutoML机器学习**

![](/assets/img/marsggbo/2020-06-02-GAMES201高级物理引擎实战指南-Lecture-1-Taichi编程语言介绍/029a9211.jpeg)

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**  **2020-06-02 15:41:06**
