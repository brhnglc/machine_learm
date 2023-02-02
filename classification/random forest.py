import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#fit eğitime
#transform uygulama


veriler = pd.read_csv('C:\\Users\\brhng\\Desktop\\veriler.csv')


x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken


from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
#ensemble bir den fazla kişi gibi yapılar majority vote(çoğunluk oyu) şekli
rfc = RandomForestClassifier(n_estimators=10,criterion = "entropy")
#criterion gini veya entropy olabilir log2 olup olmaması formülde onu belirliyor
#n_esimator= kaç ağaç oluşturulucak
rfc.fit(x_train,y_train)

y_pred=rfc.predict(x_test)



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) #sırası önemli değil
print(cm) #sol yukardan sağ aşagıya olan diagon doğruluk tersi yanlışlık y_test,y_pred için

y_proba = rfc.predict_proba(x_test) #sınıflandırma olasılıgı


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(rfc,x_test,y_test) #mode,xtest,ytest
#sol yukardan sağ aşagıya olan diagon doğruluk tersi yanlışlık 

#ROC,TPR,FPR değerlri
from sklearn.metrics import roc_curve
fpr,tpr,thold = roc_curve(y_test,y_proba[:,0],pos_label="e")#e positif seçiliyor
"""
print(fpr)
print(tpr)
print(thold)
"""











