"""
Created on Sat Jun  5 19:31:59 2021

@author: brhng
"""
"""Support Vector Machine for regression

Scale edilmek zorunda ve böylelikle marjinal verilere karşı dayanıklı hale geliyor

sınıflandırmada=> en geniş otoban seçme
tahmin=> otoban içine en çok nokta alma min margin ile

margin with(pay ,kenar aralıgı) = otoban yol genişligi

"""


import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('C:\\Users\\brhng\\Desktop\\maaslar.csv')
x=veriler.iloc[:,1:2].values
y=veriler.iloc[:,2:].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_ = sc.fit_transform(x)    
y_ = sc.fit_transform(y)    
    

from sklearn.svm import SVR
svr_reg = SVR(kernel ="rbf") #kernel burada functionlar RBF(Radial basis function)
svr_reg.fit(x_,y_)
y_pred = svr_reg.predict(x_)



plt.scatter(x_,y_)
plt.plot(x_,y_pred)    

