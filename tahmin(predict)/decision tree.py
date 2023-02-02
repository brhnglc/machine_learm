# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 20:33:09 2021
3.872
@author: brhng
"""

import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('C:\\Users\\brhng\\Desktop\\maaslar.csv')
x=veriler.iloc[:,1:2].values 
y=veriler.iloc[:,2:].values

from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(x,y) #x den y yi ögren

plt.scatter(x,y)
plt.plot(x,r_dt.predict(x))



#aralıklar hep aynı değeri veriyor
#overfitting ile ezberleme ve zaman kaybı yaşanabilir
#kutulama yapıyor