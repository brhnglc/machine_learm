# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 16:50:10 2021

@author: brhng
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


veriler = pd.read_csv('C:\\Users\\brhng\\Desktop\\odev_tenis.csv')

he =preprocessing.OneHotEncoder()
le =preprocessing.LabelEncoder()
le =preprocessing.LabelEncoder()

play= veriler.iloc[:,-1].values
play= le.fit_transform(play)


hava= veriler.iloc[:,3:4].values
hava =le.fit_transform(hava)


outlook= veriler.iloc[:,0:1].values
outlook =he.fit_transform(outlook).toarray()


sonuc = pd.DataFrame(data = play,index=range(14),columns=["play"])
sonuc2 = pd.DataFrame(data=hava,index=range(14),columns=["windy"])
sonuc3 = pd.DataFrame(data=outlook,index=range(14),columns=["o","r","s"])
sonuc4 = pd.DataFrame(data=veriler.iloc[:,1:3].values,index=range(14),columns=["tempature","humidity"])

s= pd.concat([sonuc2,sonuc3],axis=1)
s= pd.concat([s,sonuc4],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(s,sonuc,test_size=0.33,random_state=0)
 
    
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

import statsmodels.api as sm

X=np.append(arr = np.ones((14,1)).astype(int),values=s,axis=1) #1 ekliyor xverilerinin başına
X_l = s.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonuc,X_l).fit() # tahmin ve gerçek predict(x),x verdi googlle 

print(model.summary())