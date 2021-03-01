# Python自然语言处理第二版笔记

## 20210228安装nltk
1. nltk下载语料时发生11004 getaddrinfo failed错误。不知道为啥自己好了。之后修改nltk_data路径
```Python
from nltk import data
data.path.append(r"F:\PycharmProjects\NLP\NLP_book") 
```
https://github.com/nltk/nltk_data手动下载慢得一批，于是通过更改hosts。
在 C:\Windows\System32\drivers\etc 路径下找到 hosts 文件
打开查询IP地址的网址：https://www.ipaddress.com/，输入raw.githubusercontent.com得到IP地址
在最后添加 199.232.68.133 raw.githubusercontent.com IP地址，并保存
重新运行 nltk.download()

2. 接下来是语言计算的相关内容，包括词语出现的concordance，similar和common_similar词语，以及统计词频
# Python自然语言处理第二版 读书笔记

## 第一章

```Python
list(bigarms([需要处理的string list])) 寻找双联词的bi-grams函数
# 其他的属性
w for w in text if w.endswith('str')
# 还有w.istitle(), w.isdigit()
w for w in text if 'str' in w
# 也可以使用 if not/and/or

# 词语计数: 非重复\忽略大小写\忽略非字母的标点符号和数字
len(set(word.lower() for word in text1 if word.isalpha()))

# 课后作业
w.isupper()和not w.islower()的区别在于not lower还包括了数字/标点等
```

