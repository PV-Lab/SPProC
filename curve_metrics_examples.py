# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 17:13:14 2019

@author: fovie
"""

import similaritymeasures
import numpy as np

x = np.linspace(0,100,100)
y = np.sqrt(x)
y2 = np.linspace(2,2,100)

num_data = np.zeros((100, 2))
num_data[:, 0] = x
num_data[:, 1] = y

exp_data = np.zeros((100,2))
exp_data[:, 0] = x
exp_data[:, 1] = y2

area = similaritymeasures.area_between_two_curves(num_data, exp_data)


###%%
from scipy.integrate import simps
import pandas as pd
import numpy as np

x = np.linspace(0,100,100)
t = pd.Series(x)
y = abs(-np.sqrt(x)+2)

res = simps(y,t)




a = pd.Series(y)
b = pd.Series(np.linspace(0,0,100))

res = (a==b)
loc = pd.Series(res[res == True].index)

init = np.insert(loc.values,0,0)
fin = np.append(init,len(a))

results = []

for i in range(0,len(fin)-1):
    print(i)
    if i < (len(fin) - 1):
        temp_y = a.loc[fin[i]:fin[i+1]]
        temp_x = t.loc[fin[i]:fin[i+1]]
        temp_r = abs(simps(temp_y, temp_x))
        results.append(temp_r)    








