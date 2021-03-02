# Python自然语言处理第二版笔记

## 20210228安装nltk
1. nltk下载语料时发生11004 getaddrinfo failed错误。不知道为啥自己好了。之后修改nltk_data路径
```Python
from nltk import data
data.path.append(r"F:\PycharmProjects\NLP\NLP_book") 
```
https://github.com/nltk/nltk_data 手动下载慢得一批，于是通过更改hosts。
在 C:\Windows\System32\drivers\etc 路径下找到 hosts 文件
打开查询IP地址的网址：https://www.ipaddress.com/, 输入raw.githubusercontent.com得到IP地址
在最后添加 199.232.68.133 raw.githubusercontent.com IP地址，并保存
重新运行 nltk.download()

2. 接下来是语言计算的相关内容，包括词语出现的concordance，similar和common_similar词语，以及统计词频

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



## 非原创内容整理
```
<u>常用概念</u>: 

词级别: 分词(Seg), 词性标注(POS), 命名实体识别(NER), 未登录词识别, 词向量(word2vec), 词义消歧

句子级别: 情感分析, 关系提取, 意图识别, 依存句法分析(paser), 角色标注, 浅层语义分析, 指代消解

篇章级别: 信息抽取, 本体提取, 事件抽取, 主题提取, 文档聚类, 舆情分析, 篇章理解, 自动文摘
```
```
<u>工具与语料库</u>:

中科院计算所NLPIR http://ictclas.nlpir.org/nlpir/ (收费?)

ansj分词器 https://github.com/NLPchina/ansj_seg

哈工大的LTP https://github.com/HIT-SCIR/ltp 主页上给过调用接口, 每秒请求的次数有限制

清华大学THULAC https://github.com/thunlp/THULAC 

> 已有Java、Python和C++版本, 代码开源. 接口文档很详细，简单易上手. 

斯坦福分词器 https://nlp.stanford.edu/software/segmenter.shtml 可直接用训练好的模型, 也提供训练接口

Hanlp分词器 https://github.com/hankcs/ 原始模型用的训练语料是人民日报语料

> 基于词典，对各种实体词汇做了HMM，也提供了CRF模型。工程实现也不错，性能不是瓶颈。代码有相对完备的注释，文档也比较全，各种算法原理实现有对应blog，研究和二次开发较方便。
>
> CRF的字标注法是中文分词比赛上成绩最好的方案，尤其胜在新词识别上，而Language Model在词典全，词频统计正确的情况下分词效果也很好，同CRF相比，胜在分词效果稳定，易于调整，一旦发现分词错误，可以通过添加新词修正分词效果。

HanLP结巴分词 https://github.com/yanyiwu/cppjieba

> 基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的有向无环图 (DAG)；采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合；对于未登录词，采用了基于汉字成词能力的 HMM 模型，使用了 Viterbi 算法

KCWS分词器(字嵌入+Bi-LSTM+CRF) https://github.com/koth/kcws 人民日报语料, 序列标注方法

ZPar https://github.com/frcchang/zpar/releases 包括分词、词性标注和Parser，支持多语言

IKAnalyzer https://github.com/wks/ik-analyzer

https://github.com/ysc/cws_evaluation 曾对多款分词器速度和效果进行过对比，可供参考。
```

<u>公开的分词测试数据集</u>: 

SIGHAN Bakeoff 2005 MSR,560KB  http://sighan.cs.uchicago.edu/bakeoff2005/

SIGHAN Bakeoff 2005 PKU, 510KB  http://sighan.cs.uchicago.edu/bakeoff2005/

人民日报 2014, 65MB  https://pan.baidu.com/s/1hq3KKXe

<u>Python NLP的八个工具</u>: 

NLTK: NLTK是使用Python处理语言数据的领先平台。它为像WordNet这样的词汇资源提供了简便易用的界面。它还具有为文本分类(classification)、文本标记(tokenization)、词干提取(stemming)、词性标记(tagging)、语义分析(parsing)和语义推理(semantic reasoning)准备的文本处理库。

Pattern: Pattern具有用于词性标注(part-of-speech taggers)、n-gram搜索、情感分析和WordNet的一系列工具。它还支持矢量空间建模、聚类分析以及支持向量机。

TextBlob: TextBlob是处理文本数据的一个Python库。它为深入挖掘常规自然语言处理提供简单易用的API，例如词性标注(part-of-speech tagging)、名词短语提取(noun phrase extraction)、情感分析、文本分类、机器翻译等等。

Gensim: Gensim是一个用于主题建模、文档索引以及使用大规模语料数据的相似性检索。相比于RAM，它能处理更多的输入数据。作者称它是“根据纯文本进行非监督性建模最健壮、最有效的、最让人放心的软件”。

PyNLPl: PyNLPl:Python Natural Language Processing Library（发音为：pineapple）是一个用于自然语言处理的Python库。它由一系列的相互独立或相互松散独立的模块构成，用于处理常规或不太常规的NLP任务。PyNLPl可用于n-gram计算、频率列表和分布、语言建模。除此之外，还有更加复杂的数据模型，例如优先级队列；还有搜索引擎，例如波束搜索。

spaCy: spaCy是一个商业化开源软件，是使用Python和Cython进行工业级自然语言处理的软件。它是目前最快的、水平最高的自然语言处理工具。

Polyglot: Polyglot是一个支持海量多语言的自然语言处理工具。它支持多达165种语言的文本标记，196种语言的语言检测，40种语言的命名实体识别，16种语言的词性标注，136种语言的情感分析，137种语言的字根嵌入，135种语言的形态分析以及69种语言的音译。

MontyLingua: MontyLingua是一个免费的、常识丰富的、端对端的英语自然语言理解软件。用户只需要将原始英文文本输入MontyLingua，就能输出文本的语义解释。该软件完美适用于信息提取、需求处理以及问答。从给定的英语文本，它能提取主语/动词/形容词对象元组、名词短语和动词短语，并提取人的名字、地点、事件、日期和时间，以及其他语义信息。
