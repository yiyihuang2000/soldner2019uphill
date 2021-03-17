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
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

bins=25#序列数量
cluster=3#聚类cluster的数量

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
std.fillna(0,inplace=True)
low=mean-std
high=mean+std
#画图
col=[x for x in range(1,bins+1)]
if cluster%2==0:
    row=cluster/2
else:
    row=int(cluster/2)+1

'''平滑方法'''
#插值法
'''x_smooth = np.linspace(1, bins, 1000)#list没有min()功能调用
for i in range(cluster):
    plt.subplot(row,2,(i+1))
    plt.plot([0,bins],[0,0])#画0的水平线
    mean_smooth = make_interp_spline(col, mean.iloc[i,:])(x_smooth)
    high_smooth= make_interp_spline(col, high.iloc[i,:])(x_smooth)
    low_smooth= make_interp_spline(col, low.iloc[i,:])(x_smooth)
    plt.plot(x_smooth, mean_smooth,"b")
    plt.plot(x_smooth,low_smooth,"r")
    plt.plot(x_smooth,high_smooth,"r")
    plt.xticks(range(0,bins+1,5))
    plt.yticks(np.linspace(-1,1,5))
    plt.xlabel("standardize narrative time")
    plt.ylabel("sentiment")
    plt.show()'''

#Savitzky-Golay 滤波器+插值法
#name=["mood swing","middle up","uphill from here","mood swing"]
name=['mood swing','uphill from here','end on a low note']
x_smooth = np.linspace(1, bins, 2000)#list没有min()功能调用
for i in range(cluster):
    plt.subplot(row,2,(i+1))
    plt.plot([0,bins],[0,0])#画0的水平线
    mean_sf = savgol_filter(mean.iloc[i,:], 9, 3, mode= 'nearest')
    high_sf= savgol_filter(high.iloc[i,:], 9, 3, mode= 'nearest')
    low_sf= savgol_filter(low.iloc[i,:], 9,3, mode= 'nearest')
    mean_smooth = make_interp_spline(col,mean_sf )(x_smooth)
    high_smooth= make_interp_spline(col, high_sf)(x_smooth)
    low_smooth= make_interp_spline(col, low_sf)(x_smooth)
    plt.plot(x_smooth, mean_smooth,"b")
    plt.plot(x_smooth,low_smooth,"r")
    plt.plot(x_smooth,high_smooth,"r")
    #plt.xticks(range(0,bins+1,3))
    plt.yticks(np.linspace(-1,1,9))
    plt.xlabel("standardize narrative time")
    plt.ylabel("sentiment")
    plt.title("Cluster%s:%s"%(i+1,name[i]))
    plt.show()

#moving_average+插值
'''def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

x_smooth = np.linspace(1, bins, 1000)#list没有min()功能调用
for i in range(cluster):
    plt.subplot(row,2,(i+1))
    plt.plot([0,bins],[0,0])#画0的水平线
    mean_avg = moving_average(mean.iloc[i,:], 4)
    high_avg= moving_average(high.iloc[i,:], 4)
    low_avg= moving_average(low.iloc[i,:], 4)
    mean_smooth = make_interp_spline(col,mean_avg )(x_smooth)
    high_smooth= make_interp_spline(col, high_avg)(x_smooth)
    low_smooth= make_interp_spline(col, low_avg)(x_smooth)
    plt.plot(x_smooth, mean_smooth,"b")
    plt.plot(x_smooth,low_smooth,"r")
    plt.plot(x_smooth,high_smooth,"r")
    plt.xticks(range(0,bins+1,5))
    plt.yticks(np.linspace(-1,1,5))
    plt.xlabel("standardize narrative time")
    plt.ylabel("sentiment")
    plt.show()'''