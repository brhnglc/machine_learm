import tensorflow as tf
import theano
import pandas as pd
import numpy as np

veriler = pd.read_csv("C:\\Users\\brhng\\Desktop\\Churn_Modelling.csv")


x = veriler.iloc[:,3:-1].values
y = veriler.iloc[:,-1].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
x[:,1] = le.fit_transform(x[:,1])
x[:,2] = le.fit_transform(x[:,2])


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ohe = ColumnTransformer([("ohe",OneHotEncoder(dtype=float),[1])],remainder="passthrough")
 
X=ohe.fit_transform(x)
x=X[:,1:]


from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.20, random_state=0)

#Scale
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

import keras

from keras.models import Sequential
from keras.layers import Dense

classifier =Sequential() #kerasa yapay sinir ağı oluştur kullanıcaz diyoruz
classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu",use_bias=True,input_dim=11))#1.gizli katman 6 nöronlu
#units nörön  sayisi burası sanatsal bir seçim giriş ve çıkış değerlerini ortalaması alınabilir 
#init initialization weigths uniform yap sen bir şeys(Layer weight initializers)
#activation = aktivasyon functionu
#use_bias bias kullanıp kullanmama  
#input dim = giriş değerleri yani feature sayisi öznitelik şeklis



classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu",use_bias=True))#2.gizli katman 6 nöronlu
#giriş yok ondan ,input_dim=11 yazmıyoruz


classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid",use_bias=True))#çıkış katmanı ondan 1 unit

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#"adam"stochastic gradient discent in bir varyasonu
#loss error function sanırsam 
#metric ödül nedir diyor sanırsam

classifier.fit(X_train,y_train,epochs=50)
#epochs = çağ
y_pred= classifier.predict(X_test)

y_pred= (y_pred  > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("accuracy:")
print(((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][1]+cm[1][0]))*100)


from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train,y_train)
y_pred_xgb = xgb.predict(X_test)


cm2 = confusion_matrix(y_test, y_pred_xgb)
print("xgboost:")
print(cm2)
print("accuracy:")
print(((cm2[0][0]+cm2[1][1])/(cm2[0][0]+cm2[0][1]+cm2[1][1]+cm2[1][0]))*100)
