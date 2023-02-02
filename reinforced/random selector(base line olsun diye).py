# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 00:25:46 2021

@author: brhng
"""
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
veriler = pd.read_csv("C:\\Users\\brhng\\Desktop\\Ads_CTR_Optimisation.csv")

import random

n=10000# bu veri colomn sayısı 
d=10#bu örnekte reklam sayısı
toplam = 0
secilenler = []
for i in range(0,n):
    ad = random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[i,ad]
    toplam = toplam + odul

plt.hist(secilenler)#histogram oluşturur
plt.show()
