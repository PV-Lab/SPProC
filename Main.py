"""
SPProC: Sequential learning with Physical Probabilistic Constraints
@authors: 	Armi Tiihonen, Felipe Oviedo, Shreyaa Raghavan, Zhe Liu
MIT Photovoltaics Laboratory
"""

#Libraries: seaborn, scipy, pandas, Python 3.XX and GPyOpt are required

import pandas as pd
import numpy as np
import os
import matplotlib
import seaborn as sns
import GPyOpt
from scipy.integrate import simps

import ternary
import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri
from RGB_data import RGB_data

from plotting_v2 import triangleplot
from plotting_v2 import plotBO

def tolerance_factor(suggestion_df = None, tolerance_factor_bound = None):
    if (suggestion_df is not None and tolerance_factor_bound is None):
        # Calculate the tolerance factor for the given dataframe of suggestions.
        tolerance_factor_series = (220+(167*suggestion_df.CsPbI +
                                        217*suggestion_df.MAPbI +
                                        253*suggestion_df.FAPbI))/(1.4142*(119+220))
        # Works for materials = ['CsPbI', 'MAPbI', 'FAPbI'] only!
        result = tolerance_factor_series
    elif (suggestion_df is None and tolerance_factor_bound is not None):
        tolerance_factor_constraint = str(tolerance_factor_bound) + ' - ((220+(167*x[:,0] + 217*x[:,1] + 253*x[:,2]))/(1.4142*(119+220)))' # This function is designed for these elements in this
        # specific order only: materials = ['CsPbI', 'MAPbI', 'FAPbI']
        result = tolerance_factor_constraint
    else:
        raise ValueError('This function is not intended to be used for this kind of inputs.')

    return result
    
#%%
###############################################################################

# These variables are related to data collection and computing the figure of merit.
# Scroll down for changing variables related to the Bayesian Optimization.
        
# Give path to the folders that contain camera CSV data. The data will be
# analyzed in this order. Include '/' or '\' from the end of the string.
original_folder = os.getcwd()

folders = [r'/20190606-R1-JT/BMP/RGB/Calibrated/',
           r'/20190622-R1-JT/BMP/RGB/Calibrated/',
           r'/20190711-R1-JT/BMP/RGB/Calibrated/',
           r'/20190723-R1-JT/BMP/RGB/Calibrated/',
           r'/20190809-R1-JT/BMP/RGB/Calibrated/']

for n in range(len(folders)):
    folders[n] = original_folder + folders[n]     

# Give True if the data is calibrated, and False if the data is raw.
is_calibrated = True
# Give True if the data is RGB, and False if the data is LAB.
is_rgb = True

# Give the materials the compositions of which are being optimized. Use the
# same format than in the 'Samples.csv' of the aging tests.
materials = ['CsPbI', 'MAPbI', 'FAPbI']
# Note: Current implementation of tolerance factor function works only for
# these materials. Current implementation of solubility/DFT works only for
# these materials in this order.

# Give the cutoff (in minutes) for analyzing the data. The firs n minutes will
# be utilized and the rest of the data is dropped off.
cutoff = 7000

# We need to drop the empty sample slots from the analysis. This is done by
# by searching for an indicator string from 'Samples.csv', column 'Sample'. The
# default value is '---'.
indicator_for_empty = '---' # Or 'NA'

# Choose how the possible duplicate samples (same compositions several times)
# are treated. Write one of these options: 'first' (choose the first one of the
# duplicates), 'last' (choose the last one of the duplicates), 'mean'
# (calculate the mean of the duplicates, remove the original samples, and
# replace by the mean value), or 'full' (pass all the data to the BO function
# as such - the BO function takes into account every sample). This will treat
# only duplicates in each round, not between the rounds. Default value is 'full'.
duplicate_operation = 'full'

###############################################################################

# Collect the data and compute the figure of merit.

rounds = len(folders)

df_compositions = [None for j in range(rounds)]
mean_RGB = [None for j in range(rounds)]
red = [None for j in range(rounds)]
blue = [None for j in range(rounds)]
green = [None for j in range(rounds)]
times = [None for j in range(rounds)]
merit_area = [None for j in range(rounds)]
merit_diff = [None for j in range(rounds)]
merit_inv_moment = [None for j in range(rounds)]
degradation_input = [None for j in range(rounds)]
compositions_input = [None for j in range(rounds)]

for k in range(rounds):
    df_compositions[k] = pd.read_csv(folders[k] + 'Samples.csv')
    #Import RGB data and sample details.
    mean_RGB[k] = RGB_data(folders[k], df_compositions[k].get("Sample"), cutoff, is_calibrated, is_rgb)
    #Get dataframes with sample as level
    red[k], blue[k], green[k], times[k]= mean_RGB[k].preprocess()
    
    #Compute figure-of-merit, returns dataframe with figure of merit
    merit_area[k] = mean_RGB[k].compute_degradation(method = 'area')
    merit_diff[k] = mean_RGB[k].compute_degradation(method = 'diff_area')
    merit_inv_moment[k] = mean_RGB[k].compute_degradation(method = 'inverted_moment')
    
    #Pick the one that looks best, in this case will do merit_diff_abs
    #Drop the empty slots and prepare the data to be fed into GpyOpt
    print('Round ' + str(k) + ':')
    print('These slots are empty:')
    print(merit_diff[k][merit_diff[k]['Sample'].astype(str).str.contains(indicator_for_empty)])
    degradation_input[k] = merit_diff[k][~merit_diff[k]['Sample'].astype(str).str.contains(indicator_for_empty)]
    compositions_input[k] = df_compositions[k][~df_compositions[k]['Sample'].astype(str).str.contains(indicator_for_empty)]
    
    #Creating dataframe to report comparison between methods
    merit_diff[k] = merit_diff[k][~merit_diff[k]['Sample'].astype(str).str.contains(indicator_for_empty)]
    merit_inv_moment[k] = merit_inv_moment[k][~merit_inv_moment[k]['Sample'].astype(str).str.contains(indicator_for_empty)]
    merit_area[k] = merit_area[k][~merit_area[k]['Sample'].astype(str).str.contains(indicator_for_empty)]

os.chdir(original_folder)

###############################################################################
# These variables are related to the Bayesian optimization.
num_cores = 1 # Not a parallel run
composition_total = [0.995, 1] # The sum of the amount of each material will be
# limited between these values. If you need everything to sum up to 100% within
# 1%-units accuracy, choose [0.995, 1] (default value). If the code runs too
# long with this value, choose [0.99,1] or even wider range. The code currently
# works only for exactly three materials.
tolerance_factor_bound = 0.80 # Tolerance factor will be limited above this value.
tolerance_factor_function = tolerance_factor(suggestion_df = None, 
                                             tolerance_factor_bound = tolerance_factor_bound)
jitter = 0.01 # The level of exploration.


###############################################################################
#%%
# BEGIN BAYESIAN OPTIMIZATION

# Define the variables and the domain for each
# One can define here also material- or round-specific parameters.
bounds = [None for j in range(len(materials))]

for j in range(len(materials)):
    bounds[j] = {'name': materials[j], 'type': 'continuous', 'domain': (0,1)}

X_rounds = [None for j in range(rounds)]
Y_rounds = [None for j in range(rounds)]
X_step = [np.empty((0,len(materials))) for j in range(rounds)]
Y_step = [np.empty((0,1)) for j in range(rounds)] # Change (0,1) to (0,2) if multiobjective
batch_size = [None for j in range(rounds)]
constraints = [None for j in range(rounds)]

for k in range(rounds):
    batch_size[k] = len(degradation_input[0])
    # The batch size i.e. the number of suggestions the algorithm gives is the
    # same as the number of samples that were degraded in the first round.
    constraints[k] = [{'name': 'constr_1', 'constraint': 'x[:,0] +x[:,1] + x[:,2] - ' + str(composition_total[1])},
                   {'name': 'constr_2', 'constraint': str(composition_total[0])+'-x[:,0] -x[:,1] - x[:,2] '},
                   {'name': 'constr_3', 'constraint': tolerance_factor_function}]
    
    # These lines perform the selected operations to the duplicate samples
    # (choose first, choose last, choose the average, do nothing).
    df = compositions_input[k].copy()
    df['Merit'] = degradation_input[k]['Merit'].values
    if duplicate_operation == 'first':
        df = df.drop_duplicates(subset=materials, keep='first').reset_index()
    elif duplicate_operation == 'last':
        df = df.drop_duplicates(subset=materials, keep='last').reset_index()
    elif duplicate_operation == 'full':
        df = df
    elif duplicate_operation == 'mean':
        df = df.groupby(materials).mean().reset_index()
    else:
        raise Exception('The given value for treating duplicate samples is not valid. Give a valid value.')
    # X is a vector of compositions, Y is a vector of merit values.
    X_rounds[k] = df[materials].values
    # Reshaping is done to go from (n,) to (n,1), which is required by GPyOpt.
    Y_rounds[k] = np.reshape(df['Merit'].values, (df['Merit'].values.shape[0], 1))
    
    # For each BayesianOp round, we include only the data that has been
    # collected by that time.
    for j in range(rounds):
        if j >= k:
            X_step[j] = np.append(X_step[j], X_rounds[k], axis=0)
            Y_step[j] = np.append(Y_step[j], Y_rounds[k], axis=0)
    
# Do the Bayesian Optimization.
x_next = [None for k in range(rounds)]
suggestion_df = [None for k in range(rounds)]
BO_batch = [None for k in range(rounds)]
# These files contain DFT data that is integrated into the optimization loop as
# a soft constraint.
DFT_files = ['/phasestability/CsFA/fulldata/CsFA_T300_above.csv', 
             '/phasestability/FAMA/fulldata/FAMA_T300_above.csv', 
             '/phasestability/CsMA/fulldata/CsMA_T300_above.csv'
             ]
for n in range(len(DFT_files)):
    DFT_files[n] = original_folder + DFT_files[n]

for i in range(rounds):
    print('X_step and Y_step for round ' + str(i) + ':', X_step[i], Y_step[i])
    #Define Bayesian Opt object
    #Instantiate BO object, f=None as we are using tabular data, no analytical function
    BO_batch[i] = GPyOpt.methods.BayesianOptimization(f=None,  
                                            domain = bounds,
                                            constraints = constraints[i],
                                            acquisition_type = 'EI_DFT',
                                            files = DFT_files,
                                            normalize_Y = True,
                                            X = X_step[i],
                                            Y = Y_step[i],
                                            evaluator_type = 'local_penalization',
                                            batch_size = batch_size[i],
                                            acquisition_jitter = jitter)    
    #Suggest next points (samples to prepare).
    x_next[i] = BO_batch[i].suggest_next_locations()
    suggestion_df[i] = pd.DataFrame(x_next[i], columns = materials)
    suggestion_df[i]['Total'] = suggestion_df[i].sum(axis = 1)
    suggestion_df[i]['Tolerance Factor'] = tolerance_factor(
            suggestion_df = suggestion_df[i],
            tolerance_factor_bound = None)

# Plot and save the results.
plotBO(rounds, suggestion_df, compositions_input, degradation_input, BO_batch, materials, X_rounds, x_next, Y_step, X_step)    

print('Results are saved into folder ./Results.')

# Save the model as an backup
# dbfile = open('Backup-model-{date:%Y%m%d%H%M%S}'.format(date=datetime.datetime.now()), 'ab') 
# pickle.dump([BO_batch, suggestion_df, x_next, X_rounds, Y_rounds], dbfile)                      
# dbfile.close()
