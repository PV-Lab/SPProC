#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 19:19:40 2019

@author: armi
"""

import pandas as pd

r=pd.read_csv('sample_r_cal.csv', header=None)
g = pd.read_csv('sample_g_cal.csv', header=None)
b= pd.read_csv('sample_b_cal.csv', header=None)
s= pd.read_csv('Samples.csv', header=None)
t= pd.read_csv('times.csv', header=None)

rn = r.iloc[[5,6,7,9,10,11,13,14,15,17,18,19,20,21,22,23,24],:]
gn = g.iloc[[5,6,7,9,10,11,13,14,15,17,18,19,20,21,22,23,24],:]
bn = b.iloc[[5,6,7,9,10,11,13,14,15,17,18,19,20,21,22,23,24],:]
tn = t.iloc[[5,6,7,9,10,11,13,14,15,17,18,19,20,21,22,23,24],:]
sn = s.iloc[[5,6,7,9,10,11,13,14,15,17,18,19,20,21,22,23,24],:]

rn.to_csv('../sample_r_cal.csv', header=None, index=None)
gn.to_csv('../sample_g_cal.csv', header=None, index=None)
bn.to_csv('../sample_b_cal.csv', header=None, index=None)
tn.to_csv('../times.csv', header=None, index=None)
sn.to_csv('../Samples.csv', index=None)