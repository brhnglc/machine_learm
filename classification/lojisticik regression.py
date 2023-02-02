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

from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(x_train,y_train)
y_pred =logr.predict(x_test)
#hoca scale edilmiş veriyi kullandı o da olabilir ama bu daha iyi çalıştı

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred) #sırası önemli değil
print(cm) #sol yukardan sağ aşagıya olan diagon doğruluk tersi yanlışlık y_test,y_pred için














