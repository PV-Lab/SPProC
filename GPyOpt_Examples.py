# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:42:20 2019

Implementation of some examples of GpyOpt library


@author: Felipe Oviedo
"""

#%% Simple example of optmization of analytic function
import GPy
import GPyOpt
from numpy.random import seed
import matplotlib


def myf(x):
    return (2*x)**2

bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1)}]

max_iter = 15

myProblem = GPyOpt.methods.BayesianOptimization(myf,bounds)
myProblem.run_optimization(max_iter)


myProblem.x_opt
myProblem.fx_opt

#%% Simple example for 2D function

# create the object function
f_true = GPyOpt.objective_examples.experiments2d.sixhumpcamel()
f_sim = GPyOpt.objective_examples.experiments2d.sixhumpcamel(sd = 0.1)
bounds =[{'name': 'var_1', 'type': 'continuous', 'domain': f_true.bounds[0]},
         {'name': 'var_2', 'type': 'continuous', 'domain': f_true.bounds[1]}]
f_true.plot()

# Creates three identical objects that we will later use to compare the optimization strategies 
myBopt2D = GPyOpt.methods.BayesianOptimization(f_sim.f,
                                              domain=bounds,
                                              model_type = 'GP',
                                              acquisition_type='EI',  
                                              normalize_Y = True,
                                              acquisition_weight = 2)    

# runs the optimization for the three methods
max_iter = 40  # maximum time 40 iterations
max_time = 60  # maximum time 60 seconds

myBopt2D.run_optimization(max_iter,max_time,verbosity=False)     

myBopt2D.plot_acquisition() 

myBopt2D.plot_convergence()

#%% Evaluating external objective function

import GPyOpt
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt

func = GPyOpt.objective_examples.experiments1d.forrester() 

domain =[{'name': 'var1', 'type': 'continuous', 'domain': (0,1)}]


X_init = np.array([[0.0],[0.5],[1.0]])
Y_init = func.f(X_init)

iter_count = 1
current_iter = 0

#Rows are sample, columns are variable or objective evaluations
X_step = X_init
Y_step = Y_init

while current_iter < iter_count:
    bo_step = GPyOpt.methods.BayesianOptimization(f = None, domain = domain, X = X_step, Y = Y_step)
    x_next = bo_step.suggest_next_locations()
    y_next = func.f(x_next)
    
    X_step = np.vstack((X_step, x_next))
    Y_step = np.vstack((Y_step, y_next))
    
    current_iter += 1
    
x = np.arange(0.0, 1.0, 0.01)
y = func.f(x)

plt.figure()
plt.plot(x, y)
for i, (xs, ys) in enumerate(zip(X_step, Y_step)):
    plt.plot(xs, ys, 'rD', markersize=10 + 20 * (i+1)/len(X_step))
    
#To allow even more control over the execution, this API allows to specify points that should be ignored (say the objetive is known to fail in certain locations), as well as
# points that are already pending evaluation (say in case the user is running several candidates in parallel). Here is how one can provide this information.
    
#pending_X = np.array([[0.75]])
#ignored_X = np.array([[0.15], [0.85]])
#bo = GPyOpt.methods.BayesianOptimization(f = None, domain = domain, X = X_step, Y = Y_step, de_duplication = True)
#bo.suggest_next_locations(pending_X = pending_X, ignored_X = ignored_X)

#%% Batch Bayesian Optimization

import GPyOpt
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt

# --- Objective function
objective_true  = GPyOpt.objective_examples.experiments2d.branin()                 # true function
objective_noisy = GPyOpt.objective_examples.experiments2d.branin(sd = 0.1)         # noisy version
bounds = objective_noisy.bounds   


domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0]}, ## use default bounds
          {'name': 'var_2', 'type': 'continuous', 'domain': bounds[1]}]


objective_true.plot()

batch_size = 20 #Batch size 10
num_cores = 1 #We don't care about parallel run

seed(123)

BO_demo_parallel = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,  
                                            domain = domain,                  
                                            acquisition_type = 'EI',              
                                            normalize_Y = True,
                                            initial_design_numdata = 20,
                                            evaluator_type = 'local_penalization',
                                            batch_size = batch_size,
                                            acquisition_jitter = 0)    

max_iter = 1                                       
BO_demo_parallel.run_optimization(max_iter)

aa = BO_demo_parallel.suggest_next_locations()

BO_demo_parallel.plot_acquisition()

BO_demo_parallel.plot_convergence()

BO_demo_parallel.save_evaluations('fo_LP.txt')

BO_demo_parallel.save_report('fo_report_LP.txt')

#%%
#Constrained BO

func = GPyOpt.objective_examples.experiments2d.sixhumpcamel()
space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1)},
        {'name': 'var_2', 'type': 'continuous', 'domain': (-1.5,1.5)}]

constraints = [{'name': 'constr_1', 'constraint': '-x[:,1] -.5 + abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'},
              {'name': 'constr_2', 'constraint': 'x[:,1] +.5 - abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'}]

feasible_region = GPyOpt.Design_space(space = space, constraints = constraints)



