# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:10:58 2021

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
tiklamalar =[0]*d
secilenler =[]
birler = [0]*d
sifirlar = [0]*d

for i in range(0,n):
    ad=0#seçilen ilan
    max_th = 0
    for j in range(0,d):
        rasbeta = random.betavariate(birler[j]+1,sifirlar[j]+1)
        if rasbeta > max_th:
            max_th = rasbeta
            ad =j
            
    secilenler.append(ad)
    odul = veriler.values[i,ad]
    if odul == 1:
        birler[ad] = birler[ad]+1
    else:
        sifirlar[ad] = sifirlar[ad]+1
        
        
    toplam = toplam + odul
    
    
    
plt.hist(secilenler)
plt.show()