# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 18:43:39 2021

@author: brhng

#set küme yapar yani tek bir kere her elemandan
#yorum.split()  print([ps.stem(lst[j]) for lst[j] if not lst[j] in set(stopwords.words("english"))])


"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import re
import nltk    
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords


stpwrd=nltk.download("stopwords")
ps = PorterStemmer()

file = open("Restaurant_Reviews.txt")
a = file.readlines()
yorumlar = []
y = []
for i in range(1,len(a)):
    if(a[i].find("0") != -1): 
        yorumlar.append(a[i].replace(',0', ''))
        y.append(0)
    if(a[i].find("1") != -1): 
        yorumlar.append(a[i].replace(',1', ''))
        y.append(1)
        
derlem = []
for i in range(0,len(yorumlar)):        
    yorumlar[i] = re.sub("[^a-zA-Z]"," ",yorumlar[i])        
    yorumlar[i] = yorumlar[i].lower()
    lst = yorumlar[i].split()
    lst2 =""
    for j in range(0,len(lst)):
        if not (lst[j] in set(stopwords.words("english"))):
            lst2 = lst2+" "+ps.stem(lst[j]) 
    derlem.append(lst2)  
#feautre extraction bag of words(BOW) version #her bir kelime feature olarak ayarlanıyor sınıflandırma için
from sklearn.feature_extraction.text import CountVectorizer
countter = CountVectorizer(max_features=2000)#en çok alınıcak kelime
x=countter.fit_transform(derlem).toarray()  

       
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)



#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=5,criterion = "gini")
rfc.fit(x_train,y_train)
y_pred = rfc.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) #sırası önemli değil
print(cm) #sol yukardan sağ aşagıya olan diagon doğruluk tersi yanlışlık y_test,y_pred için
print("accuracy:")
print(((cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][1]+cm[1][0]))*100)

