#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:21:28 2019

Create a Gibbs energy surface as a function of material composition using
Gaussian process regression and a few values calculated by DFT

@author: armi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import GPy

def GP_model():
    file_CsFA = './phasestability/CsFA/fulldata/CsFA_T300_below.csv' # 300K = 26.85C
    file_FAMA = './phasestability/FAMA/fulldata/FAMA_T300_below.csv' # 300K = 26.85C
    file_CsMA = './phasestability/CsMA/fulldata/CsMA_T300_below.csv' # 300K = 26.85C
    file_CsFA_2 = './phasestability/CsFA/fulldata/CsFA_T300_above.csv' # 300K = 26.85C
    file_FAMA_2 = './phasestability/FAMA/fulldata/FAMA_T300_above.csv' # 300K = 26.85C
    file_CsMA_2 = './phasestability/CsMA/fulldata/CsMA_T300_above.csv' # 300K = 26.85C


    data_CsFA = pd.read_csv(file_CsFA)
    data_FAMA = pd.read_csv(file_FAMA)
    data_CsMA = pd.read_csv(file_CsMA)
    data_CsFA_2 = pd.read_csv(file_CsFA_2)
    data_FAMA_2 = pd.read_csv(file_FAMA_2)
    data_CsMA_2 = pd.read_csv(file_CsMA_2)
    data_all = pd.concat([data_CsFA_2, data_FAMA_2, data_CsMA_2])#data_CsFA, data_FAMA, data_CsMA, data_CsFA_2, data_FAMA_2, data_CsMA_2]) # Above version is more meaningful for us!
    # TO DO: Newest Bayesian Opt version works for any order of elements. Need
    # to update also DFT to do that one at some point.
    variables = ['Cs', 'MA', 'FA']
    
    # sample inputs and outputs
    X = data_all[variables] # This is 3D input
    Y = data_all[['dGmix (ev/f.u.)']] # Negative value: stable phase. Uncertainty = 0.025 
    X = X.iloc[:,:].values # Optimization did not succeed without type conversion.
    Y = Y.iloc[:,:].values
    # RBF kernel
    kernel = GPy.kern.RBF(input_dim=3, lengthscale=0.03, variance=0.025)
    # Logistic kernel --> No!
    #kernel = GPy.kern.LogisticBasisFuncKernel(input_dim=1, centers=[0, 0.5, 1], active_dims=[0], variance = 0.05) * GPy.kern.LogisticBasisFuncKernel(input_dim=1, centers=[0, 0.5, 1], active_dims=[1], variance = 0.05) * GPy.kern.LogisticBasisFuncKernel(input_dim=1, centers=[0, 0.5, 1], active_dims=[2], variance = 0.05)
    model = GPy.models.GPRegression(X,Y,kernel)
    
    # optimize and plot
    model.optimize(messages=True,max_f_eval = 100)
    #print(model)
    
    #GP.predict(X) (return mean), and __pass it to a sigmoid (0,1)__ (return), GP.raw_predict
    
    return model
    
    # This code should return the whole GP model for Gibbs.
    
    # Then, write another function here that will take composition X and model GP
    # as an input, calculate the predicted mean value of Gibbs using the model, pass
    # it to a sigmoid (0,1) to transform it to a "propability" and give that one as
    # an output.
    
    
    # X should be a numpy array containing the suggested composition(s) in the
    # same order than listed in the variables.
def mean_and_propability(x, model):#, variables):
    #if variables != ['Cs', 'FA', 'MA']:
    #    raise ValueError('The compositions in x do not seem to be in the same order than the model expects.')
    print(x)
    mean = model.predict_noiseless(x) # Manual: "This is most likely what you want to use for your predictions."
    mean = mean[0] # TO DO: issue here with dimensions?
    conf_interval = model.predict_quantiles(np.array(x)) # 95% confidence interval by default. TO DO: Do we want to use this for something?

    propability = 1/(1+np.exp(mean/0.025)) # Inverted because the negative Gibbs energies are the ones that are stable.
    print(propability)
    return mean, propability, conf_interval



model = GP_model()
model.plot(visible_dims=[0,2])
model.plot(visible_dims=[0,1])
model.plot(visible_dims=[1,2])
x1 = np.linspace(0,1,20)
x2 = np.ones(x1.shape) - x1
x3 = np.zeros(x1.shape)
x_CsMA = np.column_stack([x1,x2,x3]) #[[0.5,0,0.5], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.25,0.5,0.25], [0.5,0.25,0.25], [0.25,0.25,0.5]])
mean_CsMA, P_CsMA, conf_interval = mean_and_propability(x_CsMA, model)
x_CsFA = np.column_stack([x2,x3,x1]) #[[0.5,0,0.5], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.25,0.5,0.25], [0.5,0.25,0.25], [0.25,0.25,0.5]])
mean_CsFA, P_CsFA, conf_interval = mean_and_propability(x_CsFA, model)
x_MAFA = np.column_stack([x3,x1,x2]) #[[0.5,0,0.5], [0.5, 0.5, 0], [0, 0.5, 0.5], [0.25,0.5,0.25], [0.5,0.25,0.25], [0.25,0.25,0.5]])
mean_MAFA, P_MAFA, conf_interval = mean_and_propability(x_MAFA, model)

plt.show()
mpl.rcParams.update({'font.size': 22})
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
ax.set(xlabel='% of compound', ylabel='dGmix (ev/f.u.)',
       title='Modelled Gibbs energy')
ax2.set(xlabel='% compound', ylabel='P(is stable)',
       title='Modelled probability distribution')
ax.grid()
ax2.grid()
ax.plot(x1, mean_CsMA, label='Mean CsMA')
ax.plot(x1, mean_CsFA, label = 'Mean CsFA')
ax.plot(x1, mean_MAFA, label ='Mean MAFA')
ax.legend()

ax2.plot(x1, P_CsMA, '--', label='P CsMA')
ax2.plot(x1, P_CsFA, '--', label='P CsFA')
ax2.plot(x1, P_MAFA, '--', label='P MAFA')
ax2.legend()

x = np.linspace(-2,2,200)
y1 = 1/(1+np.exp(x/0.2))
y2 = 1/(1+np.exp(x/0.025))

fig3, ax3 = plt.subplots()
ax3.set(xlabel='x i.e. Gibbs energy', ylabel='inverted sigmoid i.e. P')
ax3.grid()
ax3.plot(x, y1, label = 'scale 0.2')
ax3.plot(x, y2, label = 'scale 0.025')
ax3.legend()

fig4, ax4 = plt.subplots()
ax4.set(xlabel='x', ylabel='inverted sigmoid')
ax4.grid()
ax4.plot(x, y1, label = 'scale 0.2')
ax4.plot(x, y2, label = 'scale 0.025')
ax4.set_xlim(-0.2, 0.2)


plt.show()
