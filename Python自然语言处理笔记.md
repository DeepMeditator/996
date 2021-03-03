# Python自然语言处理第二版笔记

## 安装nltk
nltk下载语料时发生11004 getaddrinfo failed错误。后来自己好了。之后修改nltk_data路径
```Python
from nltk import data
data.path.append(r"F:\PycharmProjects\NLP\NLP_book") 
```
https://github.com/nltk/nltk_data 手动下载慢得一批，于是通过更改hosts。
在 C:\Windows\System32\drivers\etc 路径下找到 hosts 文件
打开查询IP地址的网址：https://www.ipaddress.com/, 输入raw.githubusercontent.com得到IP地址
在最后添加 199.232.68.133 raw.githubusercontent.com IP地址，并保存
重新运行 nltk.download()

## 第一章

```Python
w for w in text if w.endswith('str')  # 还有w.istitle(), w.isdigit()
w for w in text if 'str' in w  # 也可以使用 if not/and/or

# 词语计数: 非重复\忽略大小写\忽略非字母的标点符号和数字
len(set(word.lower() for word in text1 if word.isalpha()))
# w.isupper()和not w.islower()的区别在于not lower还包括了数字/标点等
```

### 1. nltk.text.Text

```python
Text.concordance(word)  # 返回word的上下文(高频搭配)
Text.similar(word)  # 返回与word具有相似上下文的单词
Text.common_contexts([word1, word2, ...])  # 返回word1, word2...上下文的交集
Text.dispersion_plot([word1, word2, ...])  # 画图表示语料中word1, word2...出现的位置(语料的第几个词)
```

### 2. nltk.probability.FreqDist

```python
fdist=FreqDist(samples)  # 生成samples的频率分布（samples可以是nltk.text.Text, 空格分割的字符串, 列表或者其他）
fdist[word]  # 返回word在语料中出现的次数
fdist.freq(word)  # 返回word在样本中出现的频率百分比
fdist.plot()  # 绘制每个word的频率分布图
fdist.plot(cumulative=True)  # 绘制每个word的频率分布累积图
```

### 3. nltk.util.bigrams

```python
list(bigrams(["more", "is", "said", "than", "done"]))  # 这些词中的2维固定搭配
# output [('more', 'is'), ('is', 'said'), ('said', 'than'), ('than', 'done')]
Text.collocations(num=n, window_size=m)  # Text中n个(?)m维固定搭配
```

### 4. nltk.corpus.reader

```python
from nltk.corpus import gutenberg        #古腾堡语料库
from nltk.corpus import webtext          #网络语料库
from nltk.corpus import nps_chat         #聊天文本
from nltk.corpus import brown            #布朗语料库
from nltk.corpus import reuters          #路透社语料库
from nltk.corpus import inaugural        #就职演说语料库

gutenberg.fileids()  # 语料中包含的文件名(file_id)
# ['austen-emma.txt', 'austen-persuasion.txt', 'austen-sense.txt'...]
brown.categories()  # 语料中的文体
# ['editorial', 'fiction', 'romance', 'science_fiction'...]
webtext.raw()  # 逐字母输出文件的内容, 包括空格
brown.words()  # 返回包含语料中的所有词的列表
gutenberg.words(["austen-emma.txt"])  # 返回austen-emma.txt中所有词的列表
brown.words(categories="news")  # 返回categories=news的所有word的列表
brown.sents()  # 返回语料中所有句子组成的列表, 也有words的剩下两种方法

# 例: 调用gutenberg语料库的语句, 并查看语料基本情况
from nltk.corpus import gutenberg
emma = gutenberg.words('austen-emma.txt')
# 统计平均词长, 平均句长, 每词出现的平均次数, 包括空格
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
    print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)  
```

### 5. nltk.probability.ConditionalFreqDist 条件频率分布

```Python
cfd=ConditionalFreqDist(pairs)  # 创建cdf
cfd.conditions()  # 返回字母序的分类tag
cfd[condition]  # 返回指定condition的频次分布(FreqDist)
cfd[codition][sample]  # 指定cond下指定内容sample的频次


# 例: brown语料库中news和romance两类文体中每个词(包含标点)的词频统计
from nltk.corpus import brown
cfd = nltk.ConditionalFreqDist(
    (genre, word)  # conditionalfreqdist以一个配对的列表作为输入
    for genre in brown.categories()
    for word in brown.words(categories=genre))

cfd = nltk.ConditionalFreqDist(genre_word)
cfd
# Output: <ConditionalFreqDist with 2 conditions>
cfd.conditions()
# Output: ['romance', 'news']
cfd["news"]
# Output: FreqDist({'the': 5580, ',': 5188, '.': 4030, 'of': 2849, 'and': 2146, 'to': 2116, 'a': 1993, 'in': 1893, 'for': 943, 'The': 806, ...})
cfd["romance"]
# Output: FreqDist({',': 3899, '.': 3736, 'the': 2758, 'and': 1776, 'to': 1502, 'a': 1335, 'of': 1186, '``': 1045, "''": 1044, 'was': 993, ...})
cfd["romance"]["love"]
# Output: 32


cfd.tabulate(samples, conditions)  # 表格
cfd.plot(samples, conditions)  # 绘图

# 一坨例子
from nltk.corpus import udhr
languages = ["Chickasaw", "English", "German_Deutsch"]
cfd = nltk.ConditionalFreqDist(
    (lang, len(word))
    for lang in languages
    for word in udhr.words(lang+"-Latin1"))

cfd.tabulate()
'''Output:
				1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 2  23
     Chickasaw 411 99 41 68 91 89 77 70 49 33 16 28 45 10 6 4 5 3 2 1 1 1
       English 185 340 358 114 169 117 157 118 80 63 50 12 11 6 1 0 0 0 0 0 0 0
German_Deutsch 171 92 351 103 177 119 97 103 62 58 53 32 27 29 15 14 3 7 5 2 1 0
'''
cfd.tabulate(conditions=["English", "German_Deutsch"], samples=range(10), cumulative=True)
'''Output:
                 0    1    2    3    4    5    6    7    8    9
       English    0  185  525  883  997 1166 1283 1440 1558 1638
German_Deutsch    0  171  263  614  717  894 1013 1110 1213 1275
'''
```

```python
# 各种词汇列表
from nltk.corpus import words  # 236736个英语词汇
from nltk.corpus import stopwords  # stopwords表
from nltk.corpus import names  # 英语姓名
from nltk.corpus import swadesh  # 多语言单词对照词汇
```
---

## 非原创内容整理

### 顶刊上的另类数据与股票收益研究-川总写量化

Cohen, L., C. Malloy, and Q. Nguyen (2020). Lazy prices. ***Journal of Finance*** 75(3), 1371 – 1415.

Lopez-Lira, A. (2020). Risk factors that matter: Textual analysis of risk disclosures for the cross-section of returns. Available at: https://ssrn.com/abstract=3313663.

随着机器学习算法的普及，对文本数据的研究早已成为了学术界的“必争之地”。近年来，通过分析上市公司财报中的文本信息来研究股票收益率的研究也屡见不鲜，其中最有代表性的一篇当属发表在 JF 上的 Lazy Prices（Cohen, Malloy and Nguyen (2020)）。

该文分析了美股上市公司季报和年报中的文本措辞变化是否和股票收益率有关。正如其标题揭示的那样，该文发现改动越少的公司未来的预期收益越高。通过做多改动少的公司、做空改动多的公司，该投资组合可以获得超过 20% 的年化超额收益。这篇文章的精彩之处在于对背后机制的讨论。

Cohen, Malloy and Nguyen (2020) 发现财报中措辞变动背后的原因通常是以下几种：more negative sentiment、higher uncertainty、more litigiousness 以及 CEO/CFO changes。这些原因往往意味着公司的运营面临更高的风险和不确定性。

<img src="https://mmbiz.qpic.cn/mmbiz_png/MQwkyU5EcvZZlDRBCCPueNjV495U6wdjGOMpSbribiadLwBtDjBUEFWhic8VRg1FTib4N8ZmUHTHDATgd43hPQgia8g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1" alt="图片" style="zoom:50%;" />

除此之外，该文更进一步揭示了财报中的哪些 sections 发生的措辞变化最为关键，为后续进一步的深入研究奠定了很好的基础。其中一个值得多唠叨两句的 section 是美股年报中的 Item 1A：Risk Factors。

之所以单提它，是因为它让我想起了另一篇利用 Natural Language Processing 研究财报的文章，而该文研究的对象恰好就是年报中的 Risk Factors 一节（Lopez-Lira (2020)）。顺便一提，Lopez-Lira (2020) 尚未被发表，还是一篇 working paper。

Lopez-Lira (2020) 使用 Latent Dirichlet Allocation（LDA）从 Risk Factors 一节提取出 25 个 risk topics。通过进一步分析发现其中有一些可以代表不同公司面临的系统性风险，且这些系统性风险因子（risk topics）中有一些是被定价的；基于这些因子构造的多因子模型的定价能力不亚于传统的 Fama-French 三/五因子模型。感兴趣的小伙伴不妨找来一读。

---
### <u>常用概念</u>: 

词级别: 分词(Seg), 词性标注(POS), 命名实体识别(NER), 未登录词识别, 词向量(word2vec), 词义消歧

句子级别: 情感分析, 关系提取, 意图识别, 依存句法分析(paser), 角色标注, 浅层语义分析, 指代消解

篇章级别: 信息抽取, 本体提取, 事件抽取, 主题提取, 文档聚类, 舆情分析, 篇章理解, 自动文摘

### <u>工具与语料库</u>:

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

### <u>公开的分词测试数据集</u>: 

SIGHAN Bakeoff 2005 MSR,560KB  http://sighan.cs.uchicago.edu/bakeoff2005/

SIGHAN Bakeoff 2005 PKU, 510KB  http://sighan.cs.uchicago.edu/bakeoff2005/

人民日报 2014, 65MB  https://pan.baidu.com/s/1hq3KKXe

### <u>Python NLP的八个工具</u>: 

NLTK: NLTK是使用Python处理语言数据的领先平台。它为像WordNet这样的词汇资源提供了简便易用的界面。它还具有为文本分类(classification)、文本标记(tokenization)、词干提取(stemming)、词性标记(tagging)、语义分析(parsing)和语义推理(semantic reasoning)准备的文本处理库。

Pattern: Pattern具有用于词性标注(part-of-speech taggers)、n-gram搜索、情感分析和WordNet的一系列工具。它还支持矢量空间建模、聚类分析以及支持向量机。

TextBlob: TextBlob是处理文本数据的一个Python库。它为深入挖掘常规自然语言处理提供简单易用的API，例如词性标注(part-of-speech tagging)、名词短语提取(noun phrase extraction)、情感分析、文本分类、机器翻译等等。

Gensim: Gensim是一个用于主题建模、文档索引以及使用大规模语料数据的相似性检索。相比于RAM，它能处理更多的输入数据。作者称它是“根据纯文本进行非监督性建模最健壮、最有效的、最让人放心的软件”。

PyNLPl: PyNLPl:Python Natural Language Processing Library（发音为：pineapple）是一个用于自然语言处理的Python库。它由一系列的相互独立或相互松散独立的模块构成，用于处理常规或不太常规的NLP任务。PyNLPl可用于n-gram计算、频率列表和分布、语言建模。除此之外，还有更加复杂的数据模型，例如优先级队列；还有搜索引擎，例如波束搜索。

spaCy: spaCy是一个商业化开源软件，是使用Python和Cython进行工业级自然语言处理的软件。它是目前最快的、水平最高的自然语言处理工具。

Polyglot: Polyglot是一个支持海量多语言的自然语言处理工具。它支持多达165种语言的文本标记，196种语言的语言检测，40种语言的命名实体识别，16种语言的词性标注，136种语言的情感分析，137种语言的字根嵌入，135种语言的形态分析以及69种语言的音译。

MontyLingua: MontyLingua是一个免费的、常识丰富的、端对端的英语自然语言理解软件。用户只需要将原始英文文本输入MontyLingua，就能输出文本的语义解释。该软件完美适用于信息提取、需求处理以及问答。从给定的英语文本，它能提取主语/动词/形容词对象元组、名词短语和动词短语，并提取人的名字、地点、事件、日期和时间，以及其他语义信息。
