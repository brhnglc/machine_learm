# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:51:07 2021

@author: brhng
"""
import pandas as pd
import matplotlib.pyplot as plt
veriler = pd.read_csv('C:\\Users\\brhng\\Desktop\\musteriler.csv')

x = veriler.iloc[:,3:].values
from sklearn.cluster import AgglomerativeClustering
ahc = AgglomerativeClustering(n_clusters=4,linkage="ward",affinity="euclidean")
#n_clusters = küme sayısı
#linkage = kümeler arası mesafe ölçütü
#affinity = mesafe ölçütü (ward ise linkage euclidean(öklid) kullanımı zorunlu)

y_pred=ahc.fit_predict(x)
b = ahc.children_

plt.scatter(x[:,0],x[:,1])
plt.title("Cluster")
plt.show()

#y_pred == 0 oldugu değerlin hepsi 
plt.scatter(x[y_pred==0,0], x[y_pred==0,1], c="red")
plt.scatter(x[y_pred==1,0], x[y_pred==1,1], c="green")
plt.scatter(x[y_pred==2,0], x[y_pred==2,1], c="blue")
plt.scatter(x[y_pred==3,0], x[y_pred==3,1], c="yellow")

plt.show()

import scipy.cluster.hierarchy as sch
dm= sch.dendrogram(sch.linkage(x,method="ward"))
#uzun sürüyor datadan dolayı oluşturması
#bu veriden bakılırsa 2 ve 4 almak iyi 3 çok kısa cluster değeri

""" cizim için varyasyon
plt.title("Clustered")
for i in range(0,200):
    if y_pred[i]== 0:  
        plt.scatter(x[i,0],x[i,1],color="blue")
    if y_pred[i]== 1:  
        plt.scatter(x[i,0],x[i,1],color="red")
    if y_pred[i]== 2:  
        plt.scatter(x[i,0],x[i,1],color="black")    
"""