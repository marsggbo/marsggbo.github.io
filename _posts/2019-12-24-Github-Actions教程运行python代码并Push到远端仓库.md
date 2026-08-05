---
layout: post
title: Github Actions教程：运行python代码并Push到远端仓库
date: '2019-12-24'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/99080287
related_posts: false
toc:
  sidebar: left
---

> 原文: <http://zhuanlan.zhihu.com/p/99080287>

之前一篇文章介绍了AutoML论文聚合平台

[AutoML论文聚合平台](https://zhuanlan.zhihu.com/p/97961636)

因为频繁修改html文件很麻烦，所以这个平台是使用一个python脚本来生成。

具体生成的方法是python脚本会读取目录下的csv文件，将每一行数据解析成固定格式，然后生成html文件，最后需要将修改后的文件自动push到github。但是每次push之前都需要运行python文件，这很繁琐，所以后面使用Github Actions来实现了自动化部署。具体步骤逻辑如下：

1. 本地修改csv文件，然后push到github
2. push操作会触发实现设定好的action

1. 运行python脚本，生成新的html文件
2. 将修改后的文件再次push到远端仓库

action代码设置如下：

```yaml
name: Python application

on: [push]

jobs:
  build:

    runs-on: ubuntu-latest

    steps:
    - name: checkout actions
    - uses: actions/checkout@v1

    - name: Set up Python 3.7
      uses: actions/setup-python@v1
      with:
        python-version: 3.7
        
    - name: Update paper list
      run: |
        cd paper_infos
        python generate_tables.py
        
    - name: commit
      run: |
        git config --global user.email 1435679023@qq.com
        git config --global user.name marsggbo
        git add .
        git commit -m "update" -a
        
    - name: Push changes
      uses: ad-m/github-push-action@master
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
```

代码解释：

* 第一行`name`: 随便可以设置，就是你的action名字
* `on`: 触发条件，我这里设置的是`push`操作一旦发生就出发
* `jobs`: Github Actions的层级关系是这样的： workflow/jobs/steps/action。 注意将action和Github Actions中的Actions区分开来，二者是不同的概念，action就表示最低level的动作，Actions就是Github给我们提供的一个功能的名字而已。
* `steps`:和jobs类似。可以看到steps由若干个step组成，每个step都可以设置`name`
* `uses`:这个表示使用别人预先设置好的Actions，比如因为我代码中要用到python，所以就用了`actions/setup-python@v1`来设置python环境，不用我自己设置了。
* `run`: 表示具体运行什么命令行代码

+ 可以看到，我首先在名字为`Update paper list`里运行了python脚本
+ 之后对github文件夹做了commit
+ 最后使用别人的actions把更新后的代码再次push到github

* 最后一行`github_token`需要注意，这个弄了我好一会才明白，这个其实就相当于你的密码吧。这个设置方法是进入你在个人设置页面(即Settings，不是仓库里的Settings)，选择`Developer settings`>`Personal access tokens`>`Generate new token`,设置名字为`GITHUB_TOKEN`,然后勾选`repo`,`admin:repo_hook`,`workflow`等选项，最后点击`Generate token`即可。

![](/assets/img/marsggbo/2019-12-24-Github-Actions教程运行python代码并Push到远端仓库/1a3ce11c.jpg)![](/assets/img/marsggbo/2019-12-24-Github-Actions教程运行python代码并Push到远端仓库/0425ef38.jpg)![](/assets/img/marsggbo/2019-12-24-Github-Actions教程运行python代码并Push到远端仓库/ca72a990.jpg)
> 具体代码可参见[marsggbo/automl\_a\_survey\_of\_state\_of\_the\_art](https://link.zhihu.com/?target=https%3A//github.com/marsggbo/automl_a_survey_of_state_of_the_art)

### **MARSGGBO♥原创** **如有意合作，欢迎私戳** **邮箱:marsggbo@foxmail.com** **2019-12-24 11:25:45**