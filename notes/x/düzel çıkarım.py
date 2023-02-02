
""" Kelimeler
impute
plot = harita üzerinden işratelemek,haritasını çıkarmak
koralizasyon
"""
""" pyhton basics
# print('') print("") iki kullanimda kabul
x = 4   int
l = [1,2,3] list(array)


realitve path dizin vermeden direk dosyanin oldugu yere bakmasi sadece
abs path dizin vermek

class human: 
    #tab yaptiktan sonraki hepsi scope a dahil olur
    #: blogun basladigini belirtir

burhan = human()

def name(self,a):  # self =this gibi
    return a=a+5;
    #fonksiyon olusturu
    
    

boy = veriler[["boy"]]
boykilo = veriler[["boy","kilo"]]
"""    



veriler.corr() #verilerin kendi arasındaki bağlılıklarını gösteririr

    #lib
import numpy as np  #numericalpyhton
import pandas as pd #veri tutmak icin
import matplot.lib.pyplot as plt #cizim icin


pd.read_csv(path file) 
    #csv dosyası okur

yas = veriler.iloc[:,1:4].values
    # satırın hepsi : demek
    #1:4 1 den 4 e tüm columnlar demek
    #values diyerek dataframden np.array e çeviriyoruz


sonuc = pd.DataFrame(data = ulke,index=range(22),columns=["fr","tr","us"])
sonuc2 = pd.DataFrame(data=yas,index=range(22),columns=["boy","kilo","yas"])
s= pd.concat([sonuc,sonuc2],axis=1)
    #birleştirmek
    #axis birleşitirirken baz alınıcak yer yan yanamı alt alta mı


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)
    #x girdileri  y sonuclar test_size=0.33 bölünme yüzdesi %33test geri kalan train
    
    
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)    
    #ölçeklendiriyor  tüm column değrlerini belli bir aralıga ölçeklendiriyor   
    
    

     
x_train =x_train.sort_index()#indexlerine göre sıralama

plt.plot(x_train,y_train)#doğru çizimi
plt.scatter(x.values,y.values,color="red")#nokta çizimi
plt.show() #ile bitirilip yeni bir cizime başlanabilir


import statsmodels.api as sm
X=np.append(arr = np.ones((14,1)).astype(int),values=s,axis=1) #1 ekliyor xverilerinin başına
X_l = x.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(y,X_l).fit() #predict(x),x

     