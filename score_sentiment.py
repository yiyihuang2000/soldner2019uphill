# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 19:26:36 2021

@author: 黄一一"""
# -*- coding: utf-8 -*-

import pandas as pd
import jieba
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy import array as matrix, arange,zeros,transpose,matmul,ones,multiply
import math

cluster_lower_ = 2#取前两个词和后两个词一起组成这个词的cluster
cluster_upper_ = 2
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

#计算分数
def score_sentiment(sen_word, not_word, degree_word, seg_result):
    # 权重初始化为1
    W = 1
    score = 0
    # 情感词下标初始化
    sentiment_index = -1
    pre_cluster_boundary_upper=-1
    # 情感词的位置下标集合,利用sort排序
    sentiment_index_list = list(sen_word.keys())
    sentiment_index_list.sort()
    # 遍历分词结果(遍历分词结果是为了定位两个情感词之间的程度副词和否定词)
    for i in range(0, len(seg_result)):
        # 如果是情感词（根据下标是否在情感词分类结果中判断）
        if i in sen_word.keys():
            # 权重*情感词得分
            score += W * float(sen_word[i])
            # 情感词下标加1，获取下一个情感词的位置
            sentiment_index += 1
            if sentiment_index < len(sentiment_index_list):
                #如果下个值超出了下标，就使用最后一个值作为下个值
                if(sentiment_index + 1<len(sentiment_index_list)):
                    next_index= sentiment_index_list[sentiment_index + 1]
                else:
                    next_index=len(seg_result)
                
                #判断是否超出数组界限
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
                
                #将范围限制在pre<=lower<=upper<=next中间，避免重复计算
                if i+cluster_upper_>=next_index:
                    cluster_boundary_upper=next_index
                if i-cluster_boundary_lower<=pre_cluster_boundary_upper:
                    cluster_boundary_lower=pre_cluster_boundary_upper
                pre_cluster_boundary_upper=cluster_boundary_upper
                
                # 判断当前的情感词的cluster里面是否存在程度副词或否定词
                for j in range(cluster_boundary_lower,cluster_boundary_upper):
                    # 更新权重，如果有否定词，乘-1
                    if j in not_word.keys():
                        W *= -1
                    elif j in degree_word.keys():
                        # 更新权重，如果有程度副词，分值乘以程度副词的程度分值
                        W *= float(degree_word[j])
                i=next_index # 定位到下一个情感词
    return score

'''计算每句话的分数'''
def sentiment_score(sententce):
    # 对文档分词
    seg_list = seg_word(sententce)
    # 将分词结果列表转为dic，然后找出情感词、否定词、程度副词
    sen_word, not_word, degree_word = classify_words(list_to_dict(seg_list))
    # 计算得分
    score = score_sentiment(sen_word, not_word, degree_word, seg_list)
    return score


'''主程序'''
data=pd.read_csv("china-japan.csv",encoding='utf-8')
col=[x for x in range(1,bins+1)]
data=pd.concat([data, pd.DataFrame(columns=col)])#增加bins列，分割原文本
init()#构建字典
for i in range(data.shape[0]):
    text=data.iloc[i,0]
    for j in range(bins):
        l=int(len(text)/bins)
        scut=text[j*l:(j+1)*l]#分割文本
        score = sentiment_score(scut)#给分割后文本计算情感值
        data.iloc[i,j+1]=score#保存情感值
data=data.drop(labels="新闻",axis=1)#去掉原文本，方便计算均值和标准差
mm = MinMaxScaler(feature_range=(-1,1))#进行归一化，缩放到(-1,1)之间
data=mm.fit_transform(data)
np.savetxt("sentiment_score.csv",data,delimiter=",")#保存到文件中
