---
layout: post
title: "本地机器如何访问服务器上的docker容器内的tensorboard？"
date: '2020-09-16'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/248113520
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/248113520>

本文介绍如何在本地机器访问服务器上的docker容器内的tensorboard。

## 1. 创建绑定端口的docker容器

假设你的Image名字为 `img_test`,你首先需要运行如下命令创建容器

```
docker run --runtime=nvidia -p 6666:6006 -it img_test /bin/bash
```

上面命令的意思是： - `--runtime=nvidia`：绑定NVIDIA GPU，这样在docker里就可以使用GPU了，如果没这需求可以不加这个命令 - `-p 6666:6006`: 将服务器的6666端口绑定至docker容器的6006端口

## 2. docker容器内启动tensorboard

假设上一步骤创建的容器名字是`container_test`，启动tensorboard服务

```
tensorboard --logdir ./path/to/your_files --port 6006
```

## 3. 本地ssh连接到服务器

假设你的服务器IP地址是`66.66.66.66`，你的用户名是 `niubi`，那么你可以执行以下命令连接到服务器

```
ssh -L 6006:127.0.0.1:6666 niubi@66.66.66.66
```

输入命令后需要你输入密码。

```
The authenticity of host '66.66.66.66 (66.66.66.66)' can't be established.
ECDSA key fingerprint is SHA256:AiJuoq7wFDoIG2hptEvyd8hLbnV+SN5dbzPFeyiSYqc.
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added '66.66.66.66' (ECDSA) to the list of known hosts.
niubi@66.66.66.66's password:
```

## 4. 打开浏览器访问tensorboard

上一步骤中密码输入之后就成功连接至服务器了，此时你只需要打开浏览器访问`http://127.0.0.1:6006`即可访问服务器里的docker容器的tensorboard服务了。

### 微信公众号：AutoML机器学习

![](/assets/img/marsggbo/2020-09-16-本地机器如何访问服务器上的docker容器内的tensorboard/029a9211.jpeg)

**MARSGGBO♥原创**  
如有意合作或学术讨论欢迎私戳联系~  
邮箱:marsggbo@foxmail.com