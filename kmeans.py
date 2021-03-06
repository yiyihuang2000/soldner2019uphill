# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 19:25:54 2021

@author: 黄一一
"""
import pandas as pd
import jieba
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

bins=25#序列数量
cluster=4#聚类cluster的数量

#读取数据
data=pd.read_csv("sentiment_score.csv",encoding='utf-8')

'''使用kmeans聚类'''
#探索最好的cluster
'''scores = []
ch_scores=[]
for i in range(2,50):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(data)
    scores.append(metrics.silhouette_score(data, kmeans.labels_ , metric="euclidean"))
    ch_scores.append(metrics.calinski_harabasz_score(data, kmeans.labels_))
#画silhouette图
plt.figure()
plt.plot(range(2,50), scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('silhouette_score')
plt.show()
#画ch值的图
plt.figure()
plt.plot(range(2,50), ch_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Calinski-Harabaz_score')
plt.show()'''

#kmeans
kmeans = KMeans(n_clusters=cluster, random_state=0).fit(data)
label=pd.Series(kmeans.labels_)
data["label"]=label#保存分类

#画出同类的图
data=pd.DataFrame(data,dtype=np.float)#转变数据类型
#得到平均值和标准差
mean=data.groupby("label").mean()
std=data.groupby("label").std()
low=mean-std
high=mean+std
#画图
col=[x for x in range(1,bins+1)]
if cluster%2==0:
    row=cluster/2
else:
    row=cluster/2+1
for i in range(cluster):
    plt.subplot(row,2,(i+1))
    plt.plot([0,bins],[0,0])#画0的水平线
    plt.plot(col,mean.iloc[i,:],"b")
    plt.plot(col,low.iloc[i,:],"r")
    plt.plot(col,high.iloc[i,:],"r")
    plt.xticks(range(0,bins+1,5))
    plt.yticks(np.linspace(-1,1,5))
    
