# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:29:53 2021

@author: brhng
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

veriler = pd.read_csv('C:\\Users\\brhng\\Desktop\\veriler.csv')


from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
veriler.iloc[:,-1:] = lb.fit_transform(veriler.iloc[:,-1:].values)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
z = ohe.fit_transform(veriler.iloc[:,[0]].values).toarray()



z = pd.DataFrame(data= z,index=range(22),columns=["tempature","humidity","humiditsy"])

x = pd.concat([veriler.iloc[:,1:-1],z],axis=1)
y = veriler.iloc[:,[-1]]



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(x)    
Y = sc.fit_transform(y)    

from sklearn.svm import SVR
svr_reg = SVR(kernel ="rbf") #kernel burada functionlar RBF(Radial basis function)
svr_reg.fit(X,Y)
y_pred = svr_reg.predict(X)

import statsmodels.api as sm

X_l = X[:,[0,1,2,3,4]]
X_l = np.array(X_l,dtype=float)
model = sm.OLS(y_pred,X_l).fit() # tahmin ve gerçek predict(x),x verdi googlle 

print(model.summary())

    





