---
layout: post
title: 如何多级连跳远程SSH到Windows内部的Ubuntu虚拟机？
date: '2023-02-19'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/607538687
related_posts: false
toc:
  sidebar: left
---

## **1. 背景**

我学校有一台Windows电脑 (`x@a.b.c.d`)，里面安装了一个Ubuntu虚拟机（用户名为`y`），虚拟机里连着VPN。并且虚拟机的22端口映射到了主机的22端口。通过这个VPN我可以访问另一个远程的服务器（`z@e.f.g.h`）用于炼丹。

我现在在家，无法直接访问远程服务器，只能把学校电脑作为跳板才可以。下面是示意图

![](/assets/img/marsggbo/2023-02-19-如何多级连跳远程SSH到Windows内部的Ubuntu虚拟机/23834f60.jpg)

image

## **2. 在Windows上安装OpenSSH并配置服务器**

## **2.1 安装SSH Server**

假如你在机器A上安装SSH，一般情况下只有客户端，也就是说你只能去ssh到其它远端机器。但是你如果想在机器B上ssh到机器A是不行的，因为机器A并没有SSH服务器（Server）。SSH Server安装方式如下

![](/assets/img/marsggbo/2023-02-19-如何多级连跳远程SSH到Windows内部的Ubuntu虚拟机/f1d89d97.jpg)

image

## **2.2 SSH Server配置**

1. 打开SSH Server

首先按下快捷键 `Win + R`，然后输入`services.msc`，之后会进入到windows的服务管理界面。往下滑你可以找到OpenSSH SSH Server选项，双击它后点击`启动`，并将启动类型选项改为`自动`。

![](/assets/img/marsggbo/2023-02-19-如何多级连跳远程SSH到Windows内部的Ubuntu虚拟机/c7010704.jpg)

image

1. 防火墙增加入站规则，添加22端口

![](/assets/img/marsggbo/2023-02-19-如何多级连跳远程SSH到Windows内部的Ubuntu虚拟机/0d4a663a.jpg)

image

你可以通过如下命令查看22端口是否成功监听

```
netstat -an | findstr :22
```

![](/assets/img/marsggbo/2023-02-19-如何多级连跳远程SSH到Windows内部的Ubuntu虚拟机/3e66127d.jpg)

image

1. 修改`sshd_config`文件 `sshd_config`路径一般在`C:\ProgramData\ssh`文件夹里，其中`ProgramData`是隐藏文件夹，你需要在文件管理器里设置显示隐藏文件夹后就能看到了。之后用管理员权限打开`sshd_config`文件，你需要修改两个地方：

* 一个是把原本是注释状态的端口取消，即把#删掉即可

```
Port 22
```

* 另一个则是把最后两行注释掉，如下。这个耽误了我最长时间，之间一直无法成功ssh，直到把这个做了之后就可以了。

```
#Match Group administrators
#       AuthorizedKeysFile __PROGRAMDATA__/ssh/administrators_authorized_keys
```

如果你修改了上面提到的路径的`sshd_config`还是没用，你也可以试着把`C:\Windows\System32\OpenSSH`路径下的`sshd_config_default`仿照上面的方法修改一下。

## **开始SSH**

假设你远程Windows的用户名是x,IP是a.b.c.d；虚拟机的用户名是y，并且你已经将虚拟机的22端口映射到了Windows的22端口，那么你可以直接通过如下方式就可以ssh到虚拟机了

```
ssh -J x@a.b.c.d,y@127.0.0.1 z@e.f.g.h
```

## **参考**

* [https://www.cnblogs.com/sparkdev/p/10166061.html](https://link.zhihu.com/?target=https%3A//www.cnblogs.com/sparkdev/p/10166061.html)