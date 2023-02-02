# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 20:59:11 2021

@author: brhng
"""

import matplotlib.pyplot as plt
import pandas as pd



veriler = pd.read_csv('C:\\Users\\brhng\\Desktop\\maaslar.csv')
x=veriler.iloc[:,1:2].values#values demessen dataframede kalıyor
y=veriler.iloc[:,2:].values

from sklearn.ensemble import RandomForestRegressor
#regressor tahmin için
#ensemble bir den fazla kişi gibi yapılar

rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)
#n_esimator= kaç ağaç oluşturulucak

rf_reg.fit(x,y.ravel())
print(rf_reg.predict([[6.6]]))

from sklearn.metrics import r2_score #R-square hesaplama
print(r2_score(y,rf_reg.predict(x)))


plt.scatter(x,y)
plt.plot(x,rf_reg.predict(x))
#decison tree aksine bellirli değerler dışında içinde bulundurdugu agaçların değerlerini ortalaması
#ya da öyle değerler ile farklı sonuçlar döndürür


from sklearn.model_selection import cross_val_score
print("MEAN")
cvs1 = cross_val_score(rf_reg,x,y.ravel(),cv=4)
print('knn:',cvs1.mean())