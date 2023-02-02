# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 01:39:04 2021

@author: brhng
"""

import pandas as pd 
url = "https://www.bilkav.com/satislar.csv"
veriler = pd.read_csv(url)

x=veriler.iloc[:,[0]].values
y=veriler.iloc[:,[-1]].values


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x_train,y_train)

import pickle 

#dump boşaltmak demek dumperli kamyon

#wb write binary demek

pickle.dump(lr,open("kayit1","wb"))#lr modelini kaydettik


model_loaded  = pickle.load(open("kayit1","rb"))

print(model_loaded.predict([[20]]))  #sadece pickle import etmen yeterli oluyor acıcagın dosyada