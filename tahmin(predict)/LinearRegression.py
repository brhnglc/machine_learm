"""Linear Regression(simple or multipe)
birden daha fazla feature alma durumu multipe
regresyon değişkenler arası bağlantıyı matekmatiksel olarak yazmak demek

simple linear regression = b0+b1x+e
multiple linear regressin = b0+b1x+b2x2+...+bixi+e
poly linear regressiin = b0+b1+x1+b2x^2+....+bix^i+e

b'lerin hesaplanması süreci ögrenme sürecidir

"""


import matplotlib.pyplot as plt
import pandas as pd

#preprocressing
veriler = pd.read_csv('C:\\Users\\brhng\\Desktop\\maaslar.csv')
x=veriler.iloc[:,[1]].values
y=veriler.iloc[:,[-1]].values



#simple and multiple LinearRegression eğer bir feature verirsen simple daha fazlasi multiple
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y)
y_pred = lr.predict(x)
print("simple linear regression",lr.score(x, y))
print("mx+b")
print("(egim)m=",lr.coef_)
print("(y'yi kesitigi nokta')b=",lr.intercept_)


plt.scatter(x,y)
plt.plot(x,y_pred,color="green")



#Poly LinearRegression
#featureları poly linear regression için hazırlıyor
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform([[2,3]])
print(x_poly)

lr_poly = LinearRegression()
lr_poly.fit(x_poly,y) 
y_pred_poly = lr_poly.predict(x_poly)

plt.plot(x,y_pred_poly,color="red")





"""
#fit eğitime
#transform uygulama
#values demessen dataframede 
#cizgi çizme 
"""