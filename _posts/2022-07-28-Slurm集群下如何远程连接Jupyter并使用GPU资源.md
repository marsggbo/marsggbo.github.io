---
layout: post
title: "Slurm集群下如何远程连接Jupyter并使用GPU资源？"
date: 2022-07-28
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/547117062
related_posts: false
toc:
  sidebar: left
---

## **背景**

Slurm集群一般是由一个主节点（master）和各个带有GPU资源的子节点组成的，每次要想使用GPU需要通过主节点跳转到子节点。那么如果我们想使用jupyter使用子节点的GPU应该怎么做呢？

我有试过连接子节点后直接运行`jupyter`命令，然后再本地电脑上打开`127.0.0.1:8888/token?=***`，但是总是失败。其实，原因是因为我们本地电脑监听的是master节点的端口，并不能监听到子节点的端口，所以我们还需要将本地电脑和子节点的端口做映射之后才可访问jupyter。废话不多说，直接看如下教程：

## **方法**

1. 新建一个终端连接集群中的某个节点,假设节点名是`v100`

假设你已经远程连接到你的集群的master节点，然后执行以下命令进入某个指定的带有GPU资源的节点

```
srun -N 1 -p v100 -t 1440 --pty "bash";
```

* `-p v100`表示连接名为`v100`的节点
* `-t 1440`表示1440分钟，1440=24\*60，即一天
* `--pty "bash"`表示进入交互界面

1. 查看节点IP

```
cat /etc/hosts
```

运行上述命令后会打印出主节点和所有子节点的IP，输出大概长这样

```
127.0.0.1               localhost.localdomain localhost localhost4.localdomain4 localhost4
::1                     localhost.localdomain localhost localhost6.localdomain6 localhost6
10.31.29.16           psgcluster.cm.cluster ** master.cm.cluster master localmaster.cm.cluster localmaster

10.31.11.21    wwmaster        wwmaster.psg.**.zone
10.10.0.1       hydra

# PSG Cluster
10.31.225.88    v99
10.31.225.89    v100
```

在这个例子中我们需要找到`v100`节点的ip，可以看到是`10.31.225.89`。记住这个IP，后面会用到。

1. 运行jupyter-lab

第一步运行后会进入`v100`节点，之后我们需要运行jupyter环境,指定一下端口号，这里以8889为例，你也可以设置其他端口

```
jupyter-lab --port 8889

...
 LabApp] JupyterLab application directory is /datasets/xihe/miniconda3/envs/hyperbox/share/jupyter/lab
[I 2022-07-27 20:49:19.295 ServerApp] jupyterlab | extension was successfully loaded.
[I 2022-07-27 20:49:19.296 ServerApp] Serving notebooks from local directory: /home/xihe
[I 2022-07-27 20:49:19.296 ServerApp] Jupyter Server 1.11.0 is running at:
[I 2022-07-27 20:49:19.296 ServerApp] http://cluster:8889/lab?token=0be46135c38dfaa32e6c9257d00cbcb1d19ec3cc5d93f548
[I 2022-07-27 20:49:19.296 ServerApp]  or http://127.0.0.1:8889/lab?token=0be46135c38dfaa32e6c9257d00cbcb1d19ec3cc5d93f548
```

1. 实现本地和子节点的端口映射

创建一个新的终端，使用ssh命令进行映射

```
ssh -L8889:10.31.225.89:8889 username@cluster.**.com
```

这样就完成了本地电脑和`v100`子节点在8889端口的映射，此时你在打开第3步输出的ip地址就可以访问jupyter啦，即`http://127.0.0.1:8889/lab?token=0be46135c38dfaa32e6c9257d00cbcb1d19ec3cc5d93f548`

### **微信公众号：AutoML机器学习**

http://weixin.qq.com/r/HD8gOHzEmiHlrThb92oO (二维码自动识别)

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
**2022-06-27 13:59:51**