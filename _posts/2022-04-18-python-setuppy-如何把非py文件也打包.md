---
layout: post
title: python setup.py 如何把非py文件也打包？
date: '2022-04-18'
tags: [techniques]
category: techniques
grammar_cjkRuby: true
zhihu_url: http://zhuanlan.zhihu.com/p/500773671
related_posts: false
toc:
  sidebar: left
---

假设我的项目目录大致如下

```
myapp/
├── myapp
│   ├── configs
│   │   ├── data
│   │   │   └── data.yaml
│   │   └── trainer
│   │       └── trainer.yaml
│   ├── __init__.py
│   ├── run.py
│   └── trainer
│       ├── __init__.py
│       └── train.py
└── setup.py
```

注意要想制作成包的目录下必须要添加`__init__.py`文件，这里可以看到父目录名是`myapp`，它包含了一个同名的子目录，当然你也可以把这个子目录改成`src`，但是我习惯取相同的名字。

`setup.py`如下

```
from setuptools import setup, find_packages

setup(
    name="myapp",  # you should change "src" to your project name
    version="1.0",
    description="This is my app.",
    author="marsggbo",
    # replace with your own github project link
    #install_requires=["torch>=1.4"],
    packages=find_packages(),
    include_package_data=True,
)
```

我们运行如下命令

```
python setup.py sdist bdist_wheel
```

你会看到在你的目录下新生成了`build`和`dist`两个新文件夹，被打包的源代码就在`build/lib`里

```
myapp/
├── build
│   ├── bdist.linux-x86_64
│   └── lib
│       └── myapp
│           ├── __init__.py
│           ├── run.py
│           └── trainer
│               ├── __init__.py
│               └── train.py
├── dist
│   ├── myapp-1.0-py3-none-any.whl
│   └── myapp-1.0.tar.gz
├── myapp
│   ├── configs
│   │   ├── data
│   │   │   └── data.yaml
│   │   └── trainer
│   │       └── trainer.yaml
│   ├── __init__.py
│   ├── run.py
│   └── trainer
│       ├── __init__.py
│       └── train.py
├── myapp.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   └── top_level.txt
└── setup.py
```

可以看到只有py文件被打包了，而yaml文件都被丢掉了，为了解决这个问题，你需要创建一个`MANIFEST.in`文件，然后输入以下内容

```
recursive-include myapp/configs *.yaml
```

这一行代码的作用是去递归地提取出 `myapp/configs`路径下的所有yaml文件，当然如果不需要递归的话，你可以把`recursive-include`改成`include`。

现在我们重新执行打包命令

```
python setup.py sdist bdist_wheel
```

这个时候所有yaml文件也成功打包好了

```
myapp/
├── build
│   ├── bdist.linux-x86_64
│   └── lib
│       └── myapp
│           ├── configs
│           │   ├── data
│           │   │   └── data.yaml
│           │   └── trainer
│           │       └── trainer.yaml
│           ├── __init__.py
│           ├── run.py
│           └── trainer
│               ├── __init__.py
│               └── train.py
├── dist
│   ├── myapp-1.0-py3-none-any.whl
│   └── myapp-1.0.tar.gz
├── MANIFEST.in
├── myapp
│   ├── configs
│   │   ├── data
│   │   │   └── data.yaml
│   │   └── trainer
│   │       └── trainer.yaml
│   ├── __init__.py
│   ├── run.py
│   └── trainer
│       ├── __init__.py
│       └── train.py
├── myapp.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── SOURCES.txt
│   └── top_level.txt
└── setup.py
```

### **微信公众号：AutoML机器学习**

http://weixin.qq.com/r/HD8gOHzEmiHlrThb92oO (二维码自动识别)

**MARSGGBO♥原创**  
**如有意合作或学术讨论欢迎私戳联系~**  
**邮箱:marsggbo@foxmail.com**