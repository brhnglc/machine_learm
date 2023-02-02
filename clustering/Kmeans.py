# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 02:40:59 2021

@author: brhng
"""

import pandas as pd
import matplotlib.pyplot as plt
veriler = pd.read_csv('C:\\Users\\brhng\\Desktop\\musteriler.csv')

x = veriler.iloc[:,3:].values

from sklearn.cluster import KMeans
km = KMeans(n_clusters=4,init="k-means++")
#n_cluster = k yi yani kaç tane küme oluşturulucagı
#init = versiyon seçimi kmeans++,kmeans
y_pred=km.fit_predict(x)
plt.figure()
plt.title(km.n_iter_)
#plt.scatter(veriler.iloc[:,3].values,veriler.iloc[:,4].values,color="pink")
plt.scatter(x[y_pred==0,0], x[y_pred==0,1], c="red")
plt.scatter(x[y_pred==1,0], x[y_pred==1,1], c="green")
plt.scatter(x[y_pred==2,0], x[y_pred==2,1], c="blue")
plt.scatter(x[y_pred==3,0], x[y_pred==3,1], c="yellow")
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color="black",marker="D",s=50)
#marker şekli s boyutu belirtir

#burada k kaç ise o kadar scatter eklenmeli
print(km.cluster_centers_)#küme merkezlerinin konumu
plt.show()

sonuclar = []
for i in range(1,11):
    km = KMeans(n_clusters=i,init="k-means++",random_state=123)
    #burada 123 sabit random olmasın diye
    km.fit(x)
    sonuclar.append(km.inertia_)#inertia başarı değeri

plt.plot(range(1,11),sonuclar,marker="x")


plt.show()    
plt.hist(veriler["Yas"])

