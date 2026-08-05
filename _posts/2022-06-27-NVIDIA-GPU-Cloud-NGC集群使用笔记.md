---
layout: post
title: NVIDIA GPU Cloud (NGC)集群使用笔记
date: '2022-06-27'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/534424910
related_posts: false
toc:
  sidebar: left
---

## **1. 安装ngc命令**

NGC集群的使用需要用到ngc命令行，安装方法如下：

1. 下载NGC CLI

```
wget --content-disposition https://ngc.nvidia.com/downloads/ngccli_linux.zip && unzip ngccli_linux.zip && chmod u+x ngc-cli/ngc
```

1. 检查二进制文件md5 hash

```
find ngc-cli/ -type f -exec md5sum {} + | LC_ALL=C sort | md5sum -c ngc-cli.md5
```

1. 将ngc添加到path

```
echo "export PATH=\"\$PATH:$(pwd)/ngc-cli\"" >> ~/.bash_profile && source ~/.bash_profile
```

1. 配置ngc

```
ngc config set
```

如果需要卸载ngc，只需要执行如下命令：

```
dirname `which ngc` | xargs rm -r
```

## **2. NGC使用流程**

NGC运行的原理是基于docker，整个使用流程如下：

![](/assets/img/marsggbo/2022-06-27-NVIDIA-GPU-Cloud-NGC集群使用笔记/b4af3ecd.jpg)

1. 创建一个新的docker image，以pytorch为例，我们可以使用官方的pytorch image

```
docker pull nvcr.io/nvidia/pytorch:22.05-py3
```

注意要想pull [http://nvcr.io](https://link.zhihu.com/?target=http%3A//nvcr.io)的docker images，你需要首先都登陆docker，方法很简单，如下图示。注意，username就输入 **$oauthtoken** 即可，密码是你的token。

![](/assets/img/marsggbo/2022-06-27-NVIDIA-GPU-Cloud-NGC集群使用笔记/3eb4f44e.jpg)

2. 创建docker container

```
docker run --name hyperbox --gpus all  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it e34705793a75
```

* `--name test`：表示将创建的container命名为 `test`
* `--gpus all`: 表示使用GPU
* `-it`: 表示创建container后将进入交互模式
* e34705793a75: 是docker image的id

运行上面的命令后会进入container，你可以安装好所有需要的依赖库。安装好后执行如下命令即可退出container

```
exit
```

3. 将container打包成新的image

```
docker commit -a "author_name" -m "commit message" a460d64be5e3 nvcr.io/nvidian/onboarding/hyperbox:v1.1
```

* `a460d64be5e3`:是创建的container的id。可以通过`docker ps -a`查看container的id
* `nvcr.io/nvidian/onboarding/hyperbox:v1.1` : 这是给container打上了标签

+ `nvidian`是organization
+ `onboarding`是team
+ `hyperbox:v1.1` 是image的别名和版本号

4. 上传生成的image

等上面的命令执行结束后会生成新的image，执行如下命令即可上传image

```
docker push nvcr.io/nvidian/onboarding/hyperbox:v1.1
```

5. NGC配置运行

上传image后，我们在NGC网站界面便可选择指定的image了。

![](/assets/img/marsggbo/2022-06-27-NVIDIA-GPU-Cloud-NGC集群使用笔记/11722714.jpg)

除了image以外，还需要配置以下选项：

* `dataset`: NGC上有很多已经上传的dataset，用户可以把dataset挂载到指定位置`/mount/cifar10`

![](/assets/img/marsggbo/2022-06-27-NVIDIA-GPU-Cloud-NGC集群使用笔记/8c94e082.jpg)

* `workspace`：我们可以把代码存放到`workspace`，具体的操作细节会在下一节介绍。
* `result`: 实验的日志信息（如终端屏幕上打印出来的信息）都会保存到 `/result`目录下的一个log文件里，所以需要把代码存日志的路径改到`/result`路径下。

## **3. NGC基础概念：dataset result workspace**

## **3.1 `workspace`**

workspace简单理解就是云上的一个文件夹，这里面可以存很多需要的东西，比如代码，模型权重等等。

* 创建workspace，命名为`my_workspace`

```
ngc workspace create --name my_workspace
```

* 上传文件到workspace

```
ngc workspace upload --source ./local_path/source_code.py my_workspace
```

* workspace其他命令

![](/assets/img/marsggbo/2022-06-27-NVIDIA-GPU-Cloud-NGC集群使用笔记/ed811561.jpg)

## **3.2 `result`**

NGC 某个job结束后会把日志信息保存到`/result`，运行结束之后我们可以下载实验结果。

假设 job 的id是 3061231，运行如下命令即可下载实验结果

```
ngc result download 3061231
```

result其他命令如下：

![](/assets/img/marsggbo/2022-06-27-NVIDIA-GPU-Cloud-NGC集群使用笔记/e5cf692c.jpg)

## **3.3 `dataset`**

![](/assets/img/marsggbo/2022-06-27-NVIDIA-GPU-Cloud-NGC集群使用笔记/add3c584.jpg)

## **4. ngc运行命令示例**

以下命令也可以在NGC网页上自动生成

```
ngc batch run \
--name "Job-nv-us-west-2-837300" \
--preempt RUNONCE \
--min-timeslice 0s \
--ace nv-us-west-2 \
--instance dgx1v.32g.4.norm \
--total-runtime 500000s \
--image "nvidian/onboarding/hyperbox:v1.1" \
--commandline "cd /mount/workspace; python main.py" \
--result /result \
--org nvidian \
--datasetid 75237:/mount/cifar10 \
--workspace h6Vds94gT3-YyJ4365jZVg:/mount/workspace:RW
```

### **微信公众号：AutoML机器学习**

http://weixin.qq.com/r/HD8gOHzEmiHlrThb92oO (二维码自动识别)

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**   
**2022-06-27 13:59:51**