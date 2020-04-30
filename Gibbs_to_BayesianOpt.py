#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 12:12:53 2019

@author: armi
"""

from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.acquisitions.EI import AcquisitionEI
from numpy.random import beta
import numpy as np
import Gibbs
import GPyOpt
#import GPy
import matplotlib.pyplot as plt

class modified_EI(AcquisitionBase):
    
    analytical_gradient_prediction = True
    
    def __init__(self, model, space, optimizer=None, cost_withGradients=None, par_a=1, par_b=1, num_samples= 10):
        super(modified_EI, self).__init__(model, space, optimizer)
        
        self.par_a = par_a
        self.par_b = par_b
        self.num_samples = num_samples
        self.samples = beta(self.par_a,self.par_b,self.num_samples)
        self.EI = AcquisitionEI(model, space, optimizer, cost_withGradients)
        self.constraint_model = Gibbs.GP_model()
    
    def acquisition_function(self,x):
        mean, prob, conf_interval = Gibbs.mean_and_propability(x, self.constraint_model)
        acqu_x = self.EI.acquisition_function(x)*prob
        
        return acqu_x
    
    def acquisition_function_withGradients(self,x):
        
        mean, prob, conf_interval = Gibbs.mean_and_propability(x, self.constraint_model)
        acqu_x, acqu_x_grad = self.EI.acquisition_function_withGradients(x)
        acqu_x = acqu_x*prob
        acqu_x_grad = acqu_x_grad*prob
        return acqu_x, acqu_x_grad


# This code will calculate P(stable composition) using the previous file
# (think about stable<0 --> is sigmoid doing that?), take the original acq function
# and its gradient, and multiply everything with P(stable).
        
# The next step (Felipe ): transform our Bayesian model to Bayesiean modularized model.


# --- Function to optimize
func  = GPyOpt.objective_examples.experimentsNd.ackley(3)
#func.plot()

objective = GPyOpt.core.task.SingleObjective(func.f)

space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-5,10)},
                                    {'name': 'var_2', 'type': 'continuous', 'domain': (1,15)},
                                    {'name': 'var_3', 'type': 'continuous', 'domain': (1,15)}])

model = GPyOpt.models.GPModel(optimize_restarts=5,verbose=False)

aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(space)

initial_design = GPyOpt.experiment_design.initial_design('random', space, 5)

acquisition = modified_EI(model, space, optimizer=aquisition_optimizer, par_a=1, par_b=10, num_samples=5)
#xx = plt.hist(acquisition.samples,bins=50)

evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

bo = GPyOpt.methods.ModularBayesianOptimization(model, space, objective, acquisition, evaluator, initial_design)

max_iter  = 20                                            
bo.run_optimization(max_iter = max_iter)

test = bo.suggest_next_locations()

bo.plot_acquisition()
bo.plot_convergence()

print('Next locations:\n', test)
