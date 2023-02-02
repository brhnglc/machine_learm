import numpy as np  
import pandas as pd
from sklearn import preprocessing

veriler = pd.read_csv("C:\\Users\\brhng\\Desktop\\eksikveriler.csv")
ulke=veriler.iloc[:,0:1].values


le =preprocessing.LabelEncoder()
ulke[:,0] =le.fit_transform(veriler.iloc[:,0])
#label encodera her biri için 0 1 2 gibi degerler vermesi söyledik

ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
# 0 1 2 gibi değeri her biri için yeni column acıp 0 1 ileyazıyor
#listeyi verip liste olarak geri alıyorus


#katogerikten numerical a geçiş işlemleri idi   



veriler2  = veriler.apply=(preprocessing.LabelEncoder().fit_trnsform)
# tüm verilere labelencoder uygular sonra 
# bunun üstünden almak istediklerini alırsan daha hızlı olur
# dikkat normal sayılarda labelencoder yüzünden bozulucak


