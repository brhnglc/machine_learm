# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 01:36:36 2021

@author: brhng
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
veriler = pd.read_csv("C:\\Users\\brhng\\Desktop\\Ads_CTR_Optimisation.csv")

import math
import random

n=10000# bu veri colomn sayısı 
d=10#bu örnekte reklam sayısı
toplam = 0
oduller = [0] * d
tiklamalar =[0]*d
secilenler =[]
for i in range(0,n):
    ad=0#seçilen ilan
    max_ucb = 0
    for j in range(0,d):
        if tiklamalar[j]>0:
            ortalama = oduller[j]/tiklamalar[j]
            delta = math.sqrt(3/2*math.log(n)/tiklamalar[j])
            ucb = ortalama + delta
        else:
            ucb=n*10
        if max_ucb < ucb:
            max_ucb = ucb
            ad=j
            
    secilenler.append(ad)
    tiklamalar[ad] = tiklamalar[ad]+1
    odul = veriler.values[i,ad]
    oduller[ad] = oduller[ad]+odul
    toplam = toplam + odul
    
    
    
plt.hist(secilenler)
plt.show()