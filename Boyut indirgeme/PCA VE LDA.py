# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:31:11 2021

@author: brhng
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('C:\\Users\\brhng\\Desktop\\wine.csv')
x = veriler.iloc[:,:-1].values
y = veriler.iloc[:,-1].values


#train_test_split
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.20, random_state=0)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#PCA sınıfları ignorlayarak çalışır
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
#n_components = kaç feature a indirmek istedigimiz  columa indirmemiz

X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)


#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components =2)
#n_components = kaç feature a indirmek istedigimiz  columa indirmemiz
X_train3 = lda.fit_transform(X_train,y_train)# işte pca ile ayrımı y trainide alıyor denetilimli
X_test3 = lda.transform(X_test)




#LogisticRegression
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
#random stateleri 0 yapıyoruz ki bir daha kullanınca deneyimizde 
#farklı durumlar olmasın biz sadece verileren setin degişimine baglı bir gözlem istiyoruz
logr.fit(X_train,y_train)
y_pred = logr.predict(X_test)


logr2 = LogisticRegression(random_state=0)
logr2.fit(X_train2,y_train)
y_pred_pca = logr2.predict(X_test2)

logr3 = LogisticRegression(random_state=0)
logr3.fit(X_train3,y_train)
y_pred_lda = logr3.predict(X_test3)

from sklearn.metrics import plot_confusion_matrix
a= plot_confusion_matrix(logr,X_test,y_test).ax_.set_title("Normal Veri")
b= plot_confusion_matrix(logr2,X_test2,y_test).ax_.set_title("PCA ile veri") 
c= plot_confusion_matrix(logr3,X_test3,y_test).ax_.set_title("LDA ile veri") 










