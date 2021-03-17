# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 21:13:21 2021

@author: 黄一一
"""
# -*- coding: utf-8 -*-

import pandas as pd
import jieba
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
from numpy import array as matrix, arange,zeros,transpose,matmul,ones,multiply
import math
from sklearn import preprocessing
from matplotlib import pyplot as plt

cluster_lower_ = 3#取前两个词和后两个词一起组成这个词的cluster
cluster_upper_ = 3
bins=25#序列数量
stopwords = set()#停用词集合
sen_dict = {}#情感词字典
degree_dic = defaultdict()#程度词字典
not_word_list=[]#否定词字典

'''获取词典'''
def init():
    #停用词
    f = open("dict/stop_words.txt", "r",encoding="utf-8")
    s= f.readlines()
    for word in s:
        stopwords.add(word.strip())
    f.close()
    
    # 读取情感字典文件(使用BosonNLP)
    sen_file = open('dict/BosonNLP_sentiment_score.txt', 'r+', encoding='utf-8')
    # 获取字典文件内容
    sen_list = sen_file.readlines()
    # 读取字典文件每一行内容，将其转换为字典对象，key为情感词，value为对应的分值
    for s in sen_list:
        # 每一行内容根据空格分割，索引0是情感词，索引01是情感分值
        sen_dict[s.split(' ')[0]] = s.split(' ')[1]
    
    # 读取否定词文件
    not_word_file = open('dict/negator.txt', 'r+', encoding='utf-8')
    # 由于否定词只有词，没有分值，使用list
    not_word_list = not_word_file.readlines()
    
    # 读取程度副词文件
    degree_file = open('dict/degree.txt', 'r+', encoding='utf-8')
    degree_list = degree_file.readlines()
    # 程度副词与情感词处理方式一样，转为程度副词字典对象，key为程度副词，value为对应的程度值
    for d in degree_list:
        degree_dic[d.split(',')[0]] = d.split(',')[1]
    sen_file.close()
    degree_file.close()
    not_word_file.close()
    
    
'''去除停用词'''
def seg_word(sentence):
    #使用jieba分词
    sentence=sentence.replace(u'\u3000',u'') #去掉\u3000字符
    seg_list = jieba.cut(sentence)
    seg_result = []
    for w in seg_list:
        seg_result.append(w)
    # 去除停用词
    seg_word=list(filter(lambda x: x not in stopwords, seg_result))
    return seg_word

#将分词后的列表转为字典，key为单词，value为单词在列表中的索引，索引相当于词语在文档中出现的位置
def list_to_dict(word_list):
    data = {}
    for x in range(0, len(word_list)):
        data[word_list[x]] = x
    return data

'''模型构建'''
#定位情感词，否定词和程度副词
def classify_words(word_dict):
    # 分类结果，词语的index作为key,词语的分值作为value，否定词分值设为-1
    sen_word = dict()
    not_word = dict()
    degree_word = dict()
 
    # 分类
    for word in word_dict.keys():
        if word in sen_dict.keys() and word not in not_word_list and word not in degree_dic.keys():
            # 找出分词结果中在情感字典中的词
            sen_word[word_dict[word]] = sen_dict[word]
        elif word in not_word_list and word not in degree_dic.keys():
            # 分词结果中在否定词列表中的词
            not_word[word_dict[word]] = -1
        elif word in degree_dic.keys():
            # 分词结果中在程度副词中的词
            degree_word[word_dict[word]] = degree_dic[word]
    # 将分类结果返回
    return sen_word, not_word, degree_word

def prod(li):
    score=1
    for i in range(len(li)):
        if(li[i]!=0):
            score*=li[i]
    return score

#计算分数
def score_sentiment(sen_word, not_word, degree_word, seg_result):
    # 权重初始化为1
    w_result=[0]*len(seg_result)
    w_result_mod=[0]*len(seg_result)
    # 情感词下标初始化
    sentiment_index = -1
    # 情感词的位置下标集合,利用sort排序
    sentiment_index_list = list(sen_word.keys())
    sentiment_index_list.sort()
    
    #得到所有的weight
    for i in range(0, len(seg_result)):
        W = 1
        if i in sen_word.keys():
            W= W * float(sen_word[i])
        elif i in not_word.keys():
            W*=-1
        elif i in degree_word.keys():
            W *= float(degree_word[i])
        else:
            W=0
        w_result[i]=W
        
    # 遍历分词结果(遍历分词结果是为了定位两个情感词之间的程度副词和否定词)
    for i in range(0, len(seg_result)):
        if w_result[i]!=0:
            cluster_boundary_lower= i-cluster_lower_
            cluster_boundary_upper= i+cluster_upper_+1
            if i-cluster_lower_ <=0 and i+cluster_upper_ > len(seg_result):
                #取前后三个词，范围超出上下界
                cluster_boundary_lower= 0
                cluster_boundary_upper= len(seg_result)
            elif i-cluster_lower_<0:
                #下界超出，上界不超
                cluster_boundary_lower= 0
            elif i+cluster_upper_ >= len(seg_result):
                #上界超出，下界不超
                cluster_boundary_upper= len(seg_result)

            a=w_result[cluster_boundary_lower:cluster_boundary_upper]
            w_result_mod[i]=prod(a)#记录下每一个以i为cluster的情感值
    return w_result_mod

'''计算每句话的分数'''
def sentiment_score(sententce):
    # 对文档分词
    seg_list = seg_word(sententce)
    # 将分词结果列表转为dic，然后找出情感词、否定词、程度副词
    sen_word, not_word, degree_word = classify_words(list_to_dict(seg_list))
    # 计算得分
    scored = score_sentiment(sen_word, not_word, degree_word, seg_list)
    return scored

#尝试模拟get_dct_transform失败...
def dct(a):
    row=a.shape[0]
    col=a.shape[1]
    A=zeros((row,col))#生成0矩阵
    for i in range(row):
        for j in range(col):
            if(i == 0):
                x=math.sqrt(1/col)
            else:
                x=math.sqrt(2/col)
            A[i][j]=x*math.cos(math.pi*(j+0.5)*i/col)#与维数相关
    A_T=A.transpose()#矩阵转置
    Y1=multiply(A,a)#矩阵叉乘
    Y= multiply(Y1,A_T)
    return Y

'''主程序'''
data=pd.read_csv("d_china-japan.csv",encoding='utf-8')
col=[x for x in range(1,bins+1)]
ma=np.zeros((data.shape[0],bins))
#计算情感值
init()#构建字典
for i in range(data.shape[0]):
    text=data.iloc[i,0]
    scored=sentiment_score(text)#返回情感list
    l=int(len(scored)/bins)
    for j in range(bins):
        score_cut=scored[j*l:(j+1)*l]#分割文本
        score = np.sum(score_cut)#将一个分割中的情感值相加得到这个序列的情感值
        ma[i,j]=score#保存情感值
#逐行标准化      
for i in range(ma.shape[0]):
    row= ma[i,:]
    value = row.reshape(len(row), 1)
    scaler = StandardScaler().fit(value)
    st_score = scaler.transform(value)#进行标准化
    st_score=st_score.reshape(len(row))
    ma[i,:]=st_score

'''mm = MinMaxScaler(feature_range=(-1,1))#进行归一化，缩放到(-1,1)之间
ma=mm.fit_transform(ma)'''
np.savetxt("sentiment_score.csv",ma,delimiter=",")#保存到文件中'''


#新闻的分布直方图
'''for k in range(data.shape[0]):
    length.append(len(data.iloc[k,0]))
plt.figure() #初始化一张图
plt.hist(length)  #直方图关键操作
plt.grid(alpha=0.5,linestyle='-.') #网格线，更好看 
plt.xlabel('Life Cycle /Month')  
plt.ylabel('Number of Events')  
plt.title(r'Life cycle frequency distribution histogram of events in New York') 
plt.show()'''

#单条新闻的序列分析图
'''col=[x for x in range(1,bins+1)]
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter
data2=pd.read_csv("sentiment_score.csv",encoding='utf-8')
score=data2.iloc[-7,:]
plt.plot(col,score)
x_smooth = np.linspace(1, bins, 1000)#list没有min()功能调用
y_sf = savgol_filter(score, 11, 3, mode= 'nearest')
y_smooth = make_interp_spline(col,y_sf )(x_smooth)
plt.plot(x_smooth, y_smooth,"b")'''


