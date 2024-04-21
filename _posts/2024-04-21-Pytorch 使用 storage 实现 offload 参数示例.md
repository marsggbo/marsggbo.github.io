---
title: Pytorch 使用 storage 实现 offload 参数示例
tags: 技术,pytorch,offload,torch.Storage
category: /techniques
layout: post
date: 2024-04-10 14:04:10
related_posts: false
toc:
  sidebar: left
---



在深入探讨 PyTorch 中的 `Storage` 类以及其在参数 offload 场景中的应用之前，让我们首先了解一下 PyTorch 和它的基础组件。PyTorch 是一个广泛使用的开源机器学习库，它不仅提供了强大的计算图功能和自动梯度计算，还允许开发者直接操作底层数据结构，这其中就包括 `Storage`。

# 1. 什么是 `torch.Storage`?

在 PyTorch 中，`Storage` 是一种容纳数据的一维数组，它可以看作是一个底层的内存块，其中存储着特定类型的数据。与 `Tensor` 的关系非常紧密，实际上，每个 `Tensor` 都有一个与之关联的 `Storage` 对象。`Tensor` 提供了一个高维视图来操作存储在 `Storage` 中的数据。

`Storage` 的一个关键特性是它的数据排列是连续的，这使得数据可以迅速地在设备之间传输，例如从 CPU 到 GPU，省去了频繁索引的操作。此外，`Storage` 可以存在于不同的设备上，如 CPU 或 CUDA（GPU）。

使用 storage 实现 offload 参数场景大致有如下：

- **模型训练时的内存优化**：
在深度学习模型训练过程中，特别是当使用的模型非常大，以至于单个 GPU 显存不足时，可以使用 offload 技术将部分数据暂时存储到 CPU 内存中，从而释放 GPU 显存用于计算。

- **数据预处理**：
在进行大规模数据处理时，可以将不活跃的数据段 offload 到 CPU，以保持 GPU 资源用于执行高优先级的任务。

- **长期数据存储**：
对于不需要频繁访问的大量数据，可以将其 offload 到 CPU 或其他存储系统，以减少昂贵的 GPU 存储资源的占用。

# 2. 理解 `Storage` 

## 2.1 简单例子
```python
import torch

x = torch.arange(3, dtype=torch.float32).cuda()
print(x.storage())
```
输出结果如下，可以看到打印出来的结果符合预期，有三个浮点数，storage 的类型是 `torch.storage.TypedStorage`

```python
 0.0
 1.0
 2.0
[torch.storage.TypedStorage(dtype=torch.float32, device=cuda:0) of size 3]
```

更一般地，我们还能打印看看无类型的 storage 是什么样的
```python
x_storage = x.storage()._untyped_storage
print(x_storage)
```
输出结果如下，可以看到总共有 12 个整数，这是因为前面我们使用的数据类型是 float32，也就是说每个数由 4 个字节（bytes）表示。因为 变量 x 总共有 3 个数，所有它的 storage 总共有 12 个字节。

```python
 0
 0
 0
 0
 0
 0
 128
 63
 0
 0
 0
 64
[torch.storage.UntypedStorage(device=cuda:0) of size 12]
```

这些值实际上是浮点数`0`、`1`、`2`在内存中的字节级表示。需要注意的是，上面输出结果并不是随机值，而是这些浮点数在 IEEE 754 标准下的二进制表达。我们可以逐个解释这些值如何来的。

## 2.2 浮点数的 IEEE 754 表示

对于类型 `float32`（即单精度浮点数），每个数字占用 4 个字节（32位），具体编码方式为：
- 1 位符号位（最高位）
- 8 位指数位
- 23 位尾数位

在解释这些值之前，我们先了解一下计算机中的 **小端序（Little Endian）** 存储方式：在这种存储方式中，低位字节存放在内存的低地址端，高位字节存放在高地址端。


以`Tensor[0., 1., 2.]` 为例，我们来看看这些值在内存中是如何表示的：

1. **数字 0 的浮点表示**：
   - 符号位：0
   - 指数位：全0（偏移量为127，因此全0表示指数-127）
   - 尾数位：全0
   - **二进制表示**：`00000000 00000000 00000000 00000000`
   - **十六进制表示**：`00 00 00 00`
   - **小端序下的字节表示**：`00 00 00 00`
   - **上面结果转化成十进制表示**： `0 0 0 0`

2. **数字 1 的浮点表示**：
   - 符号位：0
   - 指数位：127（偏移后为0，`01111111`）
   - 尾数位：全0（因为1.0的尾数部分无需额外存储）
   - **二进制表示**：`001111111 00000000000000000000000`
   - **十六进制表示**：`3F 80 00 00`
   - **小端序下的字节表示**：`00 00 80 3F` 
   - **上面结果转化成十进制表示**： `0 0 128 63` (`80` 十六进制转十进制是 `128`，`3F` 转十进制是 `63`)

3. **数字 2 的浮点表示**：
   - 符号位：0
   - 指数位：128（偏移后为1，`10000000`）
   - 尾数位：全0（因为2.0的尾数部分也无需额外存储）
   - **二进制表示**：`010000000 00000000000000000000000`
   - **十六进制表示**：`40 00 00 00`
   - **小端序下的字节表示**：`00 00 00 40`

# 3. 使用 Storage 实现参数 offload 到 cpu
前面例子中的变量`x`在 cuda上，为了实现 offload，我们需要在 cpu 上创建一个 storage，如下：

```python
offload_storage = torch.UntypedStorage(x.nbytes).pin_memory(x.device)
print(offload_storage.device)
print(offload_storage)
```

输出结果如下,可以看到`offload_storage`是在 cpu 上，目前其上面的值都是一些随机值。
```python
cpu
 208
 238
 22
 7
 0
 0
 0
 0
 208
 66
 20
 6
[torch.storage.UntypedStorage(device=cpu) of size 12]
```

接下来我们需要把 `x` offload 到 cpu 上，只需要对 storage 做 copy 操作即可，代码如下：

```python
offload_storage.copy_(x_storage)
print(offload_storage.device)
print(offload_storage)
```

输出结果如下：
```python
cpu
 0
 0
 0
 0
 0
 0
 128
 63
 0
 0
 0
 64
[torch.storage.UntypedStorage(device=cpu) of size 12]
```

可以看到`x`的值被成功拷贝到 cpu 上，但是这离实现 offload 还有一步之遥，我们接下来继续看一个简单的 offload 例子。

# 4. gpu 参数 和 cpu 参数互换


我们接着将探讨如何利用 Storage 实现 GPU 和 CPU 之间的数据互换，这对于处理大型数据集或进行复杂的数据处理任务时尤其有用。


假设我们有以下设置：
- 一个 CUDA `Tensor` 用于当前计算。
- 多个 CPU `Storage` 用于存储额外的数据集，这些数据集可能在不同时间被需求到 GPU。

## 4.1  初始化环境

首先，我们定义一个在 CUDA 上的 Tensor 和多个在 CPU 上的 Storage，准备用于数据交换：

```python
import torch

# 定义 CUDA Tensors (用于当前计算)
current_data = torch.tensor([0.0, 1.0], device='cuda')

# 定义 CPU Storages (用于存储额外数据)
extra_data1 = torch.FloatTensor([2.0, 3.0]).storage().pin_memory()
extra_data2 = torch.FloatTensor([4.0, 5.0]).storage().pin_memory()
extra_data3 = torch.FloatTensor([6.0, 7.0]).storage().pin_memory()

print("Initial CUDA Tensor (Current Data):")
print(current_data)

print("\nInitial CPU Storages (Extra Data):")
print("Extra Data 1:", list(extra_data1))
print("Extra Data 2:", list(extra_data2))
print("Extra Data 3:", list(extra_data3))
```

输出结果为：
```python
Initial CUDA Tensor (Current Data):
tensor([0., 1.], device='cuda:0')

Initial CPU Storages (Extra Data):
Extra Data 1: [2.0, 3.0]
Extra Data 2: [4.0, 5.0]
Extra Data 3: [6.0, 7.0]
```

## 4.2 使用缓冲区进行数据交换

接下来，我们将根据需要将 CPU 上的数据加载到 CUDA `Tensor` 中，同时将当前 CUDA `Tensor` 的数据存储回某个 CPU `Storage`，这可以申请一个 buffer 来作为中间变量，反正数据丢失。

```python
# 缓冲区定义
cpu_buffer = torch.FloatTensor(current_data.size()).storage().pin_memory()  # CPU buffer storage

# 场景1：将 current_data 保存到 extra_data1，从 extra_data1 加载新数据到 current_data
cpu_buffer.copy_(current_data.storage())  # Save current GPU data to CPU buffer
current_data.storage().copy_(extra_data1)  # Move from CUDA buffer to current_data
extra_data1.copy_(cpu_buffer)  # Move from CPU buffer to extra_data1 Storage

print("\nAfter Data Exchange Scenario 1:")
print(f"Updated Current Data on {current_data.device}:", current_data)
print(f"Updated Extra Data 1 on {extra_data1.device}:", list(extra_data1))

print("Extra Data 2:", list(extra_data2))
print("Extra Data 3:", list(extra_data3))
```

#### 输出结果

```python
After Data Exchange Scenario 1:
Updated Current Data on cuda:0: tensor([0., 1.], device='cuda:0')
Updated Extra Data 1 on cpu: [2.0, 3.0]
Extra Data 2: [4.0, 5.0]
Extra Data 3: [6.0, 7.0]
```

此示例清晰地展示了如何利用 PyTorch 的 Storage 类来有效管理内存资源，并通过使用 CPU 和 CUDA 缓冲区动态切换数据来优化应用性能。这种方法尤其适用于需要频繁在不同计算设备之间迁移数据的场景，从而保证计算效率和响应速度。

尽管可以通过 PyTorch 的 to('cpu') 或 to('cuda') 方法简单地在设备间迁移数据，使用 Storage 提供了更细粒度的控制。这在处理需要大量连续物理存储空间的复杂模型时显得尤为重要。

例如在混合专家模型（MoE）中，系统需要根据不同的请求动态调用不同的专家（模型）。每个专家可能包含的是由多层感知机 (MLP) 或更复杂结构组成的模型，其中每层的参数在内存中通常是不连续的。这种不连续性可能导致在将参数 offload 到 CPU 或重新加载到 GPU 时，因频繁的内存访问和索引操作而增加通信开销。






<footer style="color:white;;background-color:rgb(24,24,24);padding:10px;border-radius:10px;">
<h3 style="text-align:center;color:tomato;font-size:16px;" id="autoid-2-0-0">
<center>
<span>微信公众号：AutoML机器学习</span><br>
<img src="https://pic4.zhimg.com/80/v2-87083e55cd41dbef83cc840c142df48a_720w.jpeg" style="width:200px;height:200px">
</center>
<b>MARSGGBO</b><b style="color:white;"><span style="font-size:25px;">♥</span>原创</b><br>
<span>如有意合作或学术讨论欢迎私戳联系~<br>邮箱:marsggbo@foxmail.com</span>
<b style="color:white;"><br>
</b><p><b style="color:white;"></b>
</p></h3>
</footer>
