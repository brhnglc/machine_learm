import numpy as np  
import pandas as pd

    
veriler = pd.read_csv("C:\\Users\\brhng\\Desktop\\eksikveriler.csv")

yas= veriler.iloc[:,1:4].values
print(yas)
yas2 =yas.copy()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan,strategy ="mean" ) 
yas[:,1:4] = imputer.fit_transform(yas[:,1:4])



#eksik olan nan değerler yerine mean ortalamalarını aldık