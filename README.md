# soldner2019uphill

## Project
基于论文Soldner, Felix, Justin Chun-ting Ho, Mykola Makhortykh, Isabelle WJ van der Vegt,Maximilian Mozes, and Bennett Kleinberg. 2019. “Uphill from Here: Sentiment Patterns 
in Videos from Left-and Right-Wing YouTube News Channels.” In Proceedings of the Third Workshop on Natural Language Processing and Computational Social Science, 84–93. 的复现尝试。
先将输入句子按照长度分成等分的时间序列，再分别计算序列特征的情感值，对其进行kmeans聚类，进行轨迹分析。



## Install
请确保您有以下包，若没有也可输入以下进行安装
    pip install pandas  
    pip install jieba  
    pip install numpy  
    pip install csv  
    pip install matplotlib  
    pip install sklearn  
    pip install math  


## Usage
1. 请下载整个项目，将dict文件夹解压缩
2. 打开score_sentiment.py进行china-japan.csv文件的情感值计算，得到sentiment_score.csv
3. 打开kmeans.py进行聚类，并得到聚类结果图



## Problems
* 目前聚类结果的分类别折线图在总体情感值上有差别，但折线形状较为相似，可能是样本量太小、数据处理不当和聚类粗糙的问题，并没有得到论文中明显的类别区分
* 不熟悉R语言，所以在参考作者所使用的R包中可能存在理解错误
