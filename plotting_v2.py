#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Armi Tiihonen, Felipe Oviedo, Shreyaa Raghavan, Zhe Liu
MIT Photovoltaics Laboratory
"""

import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.tri as tri

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'Results/')

if not os.path.isdir(results_dir):
    os.makedirs(results_dir)


def triangleplot(surf_points, surf_data, norm, surf_axis_scale = 1, cmap = 'RdBu_r',
                 cbar_label = '', saveas = None, surf_levels = None,
                 scatter_points=None, scatter_color = None, cbar_spacing = None,
                 cbar_ticks = None):


    mpl.rcParams.update({'font.size': 8})
    mpl.rcParams.update({'font.sans-serif': 'Arial', 'font.family': 'sans-serif'})
    
    
    b=surf_points[:,0]
    c=surf_points[:,1]
    a=surf_points[:,2]
    # values stored in the last column
    v = np.squeeze(surf_data)/surf_axis_scale#[:,-1]/surf_axis_scale
    # translate the data to cartesian
    x = 0.5 * ( 2.*b+c ) / ( a+b+c )
    y = 0.5*np.sqrt(3) * c / (a+b+c)
    # create a triangulation
    T = tri.Triangulation(x,y)
    
    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.set_figheight(3.6/2.54)
    fig.set_figwidth(5/2.54)
    plt.subplots_adjust(left=0.05, right=0.85, bottom=0.14, top=0.91)

    if scatter_points is not None:
        #Triangulation for the points suggested by BO.
        b_p = scatter_points[:,0]
        c_p = scatter_points[:,1]
        a_p = scatter_points[:,2]
        x_p = 0.5 * ( 2.*b_p+c_p ) / ( a_p+b_p+c_p)
        y_p = 0.5*np.sqrt(3) * c_p / (a_p+b_p+c_p)
        
        im3 = ax.scatter(x_p, y_p, s=8, c=scatter_color, cmap=cmap, edgecolors='black', linewidths=.5, alpha=1, zorder=2, norm=norm)
    # plot the contour
    if surf_levels is None:
    #    im=ax.tricontourf(x,y,T.triangles,v, cmap=cmap, levels=surf_levels)
    #else:
        nlevels = 8
        minvalue = 0
        if norm.vmin < minvalue:
            minvalue = norm.vmin
        if norm.vmin > (minvalue + ((norm.vmax-minvalue)/nlevels)):
            minvalue = norm.vmin
        surf_levels = np.arange(minvalue, norm.vmax, (norm.vmax-minvalue)/nlevels)
        surf_levels = np.round(surf_levels, -int(np.floor(np.log10(norm.vmax))-1))#2))
        surf_levels = np.append(surf_levels, 2*surf_levels[-1] - surf_levels[-2])
    im=ax.tricontourf(x,y,T.triangles,v, cmap=cmap, levels=surf_levels)
    
    myformatter=matplotlib.ticker.ScalarFormatter()
    myformatter.set_powerlimits((0,2))
    if cbar_spacing is not None:
        cbar=plt.colorbar(im, ax=ax, spacing=cbar_spacing, ticks=cbar_ticks)
    else:
        cbar=plt.colorbar(im, ax=ax, format=myformatter, spacing=surf_levels, ticks=surf_levels)
    cbar.set_label(cbar_label)#, labelpad = -0.5)
    plt.axis('off')
    plt.text(0.35,-0.1,'Cs (%)')
    plt.text(0.10,0.54,'FA (%)', rotation=61)
    plt.text(0.71,0.51,'MA (%)', rotation=-61)
    plt.text(-0.0, -0.1, '0')
    plt.text(0.87, -0.1, '100')
    plt.text(-0.07, 0.13, '100', rotation=61)
    plt.text(0.39, 0.83, '0', rotation=61)
    plt.text(0.96, 0.05, '0', rotation=-61)
    plt.text(0.52, 0.82, '100', rotation=-61)
    
    # create the grid
    corners = np.array([[0, 0], [1, 0], [0.5,  np.sqrt(3)*0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    # creating the grid
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=0)
    #plotting the mesh
    im2=ax.triplot(trimesh,'k-', linewidth=0.5)
    
    
    if saveas:
        fig.savefig(results_dir + saveas + '.pdf', transparent = True)
        fig.savefig(results_dir + saveas + '.svg', transparent = True)
        fig.savefig(results_dir + saveas + '.png', dpi=300)
    plt.show()
    return fig, ax

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    ## TO DO: This is a modified version of an open access function. Deal with ack.
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), #128, endpoint=False), 
        np.linspace(midpoint, 0.3, 80, endpoint=False),
        np.linspace(0.3, 1.0, 49, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

def plotBO(rounds, suggestion_df, compositions_input, degradation_input, BO_batch, materials, X_rounds, x_next, Y_step, X_step):
    #%%    
    ### This grid is used for sampling+plotting the posterior mean and std_dv + acq function.
    a = np.arange(0.0,1.0, 0.005)
    xt, yt, zt = np.meshgrid(a,a,a, sparse=False)
    points = np.transpose([xt.ravel(), yt.ravel(), zt.ravel()])
    points = points[abs(np.sum(points, axis=1)-1)<0.005]
    
    posterior_mean = [None for k in range(rounds)]
    posterior_std = [None for k in range(rounds)]
    acq_normalized = [None for k in range(rounds)]
    
    original_folder = os.getcwd()
    os.chdir(original_folder)
    
    ###############################################################################
    # SAVE RESULTS TO CSV FILES
    ###############################################################################
    for i in range(rounds):
        suggestion_df[i].to_csv(results_dir + 'Bayesian_suggestion_round_'+str(i)+'.csv', float_format='%.3f')
        inputs = compositions_input[i]
        inputs['Ic (px*min)'] = degradation_input[i]['Merit']
        inputs=inputs.sort_values('Ic (px*min)')
        inputs=inputs.drop(columns=['Unnamed: 0'])
        inputs.to_csv(results_dir + 'Model_inputs_round_'+str(i)+'.csv', float_format='%.3f')
        # : Here the posterior mean and std_dv+acquisition function are calculated and saved to csvs.
        posterior_mean[i],posterior_std[i] = BO_batch[i].model.predict(points) # MAKING THE PREDICTION
        # Scaling the normalized data back to the original units.
        posterior_mean[i] = posterior_mean[i]*Y_step[i].std()+Y_step[i].mean()
        posterior_std[i] = posterior_std[i]*Y_step[i].std()
        # Scaling the acquisition function to btw 0 and 1.
        acq=BO_batch[i].acquisition.acquisition_function(points)
        acq_normalized[i] = (-acq - min(-acq))/(max(-acq - min(-acq)))
    
        inputs2 = pd.DataFrame(points, columns=materials)
        inputs2['Ic (px*min)']=posterior_mean[i]
        inputs2['Std of Ic (px*min)']=posterior_std[i]
        inputs2['EIC']=acq_normalized[i]
        inputs2.to_csv(results_dir + 'Model_round_' + str(i) + '.csv', float_format='%.3f')
        
    
    
    
    ###############################################################################
    # PLOT MERIT
    ###############################################################################
    # Let's plot the resulting suggestions. This plot works for 3 materials only.
    # Colorbar scale:
    #scale = 1
    
    # Min and max values for each contour plot are determined and normalization
    # of the color range is calculated.
    axis_scale = 60 # Units from px*min to px*hour.
    lims = [[np.min(posterior_mean)/axis_scale, np.max(posterior_mean)/axis_scale],
            [np.min(posterior_std)/axis_scale, np.max(posterior_std)/axis_scale],
            [np.min(acq_normalized), np.max(acq_normalized)]] # For mean, std, and acquisition.
    
    norm = matplotlib.colors.Normalize(vmin=lims[0][0], vmax=lims[0][1])    
    for i in range(rounds):
        triangleplot(points, posterior_mean[i]/axis_scale, norm, cmap = 'RdBu_r',
                 cbar_label = r'$I_{c}(\theta)$ (px$\cdot$h)', saveas = 'Modelled-Ic-no-grid-round'+str(i)) #A14


    norm = matplotlib.colors.Normalize(vmin=lims[1][0], vmax=lims[1][1])
    for i in range(rounds):
        triangleplot(points, posterior_std[i]/axis_scale, norm, cmap = 'RdBu_r',
                 cbar_label = r'Std $I_{c}(\theta)$ (px$\cdot$h)', saveas = 'St-Dev-of-modelled-Ic-round'+str(i)) #A14

    norm = matplotlib.colors.Normalize(vmin=lims[2][0], vmax=lims[2][1])
    # Shift the colormap (removes the red background that appears with the std colormap)
    orig_cmap = mpl.cm.RdBu
    shifted_cmap = shiftedColorMap(orig_cmap, start=0, midpoint=0.000005, stop=1, name='shifted')
    # Colors of the samples in each round.     
    newPal = {0:'k', 1:'k', 2:'k', 3:'k', 4: 'k'}#{0:'#8b0000', 1:'#7fcdbb', 2:'#9acd32', 3:'#ff4500', 4: 'k'} #A14
    for i in range(rounds):
        print('Round: ', i) #A14
        if i==0:
            # In the first round there is an even distribution.
            test_data = np.concatenate((points, acq_normalized[i]), axis=1)
            test_data = test_data[test_data[:,3] != 0]
            test_data[:,3] = np.ones(test_data[:,3].shape)*0.3
        else:
            test_data = np.concatenate((points, acq_normalized[i-1]), axis=1)
            test_data = test_data[test_data[:,3] != 0]
        triangleplot(test_data[:,0:3], test_data[:,3], norm,
                     surf_axis_scale = 1.0, cmap = 'shifted',
                     cbar_label = r'$EIC(\theta, \beta_{DFT})$', #A14
                     saveas = 'EIC-with-single-round-samples-round'+str(i), #A14
                     surf_levels = (0,0.009,0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8,1),
                     scatter_points=X_rounds[i], scatter_color = newPal[i],
                     cbar_spacing = 'proportional',
                     cbar_ticks = (0,0.2,0.4,0.6,0.8,1))
    for i in range(rounds):
        print('Round: ', i) #A14
        if i==0:
            # In the first round there is a even distribution.
            test_data = np.concatenate((points, acq_normalized[i]), axis=1)
            test_data = test_data[test_data[:,3] != 0]
            test_data[:,3] = np.ones(test_data[:,3].shape)*0.3
        else:
            test_data = np.concatenate((points, acq_normalized[i-1]), axis=1)
            test_data = test_data[test_data[:,3] != 0]
        triangleplot(test_data[:,0:3], test_data[:,3], norm,
                     surf_axis_scale = 1.0, cmap = 'shifted',
                     cbar_label = r'$EIC(\theta, \beta_{DFT})$', #A14
                     saveas = 'EIC-round'+str(i), #A14
                     surf_levels = (0,0.009,0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8,1),
                     cbar_spacing = 'proportional',
                     cbar_ticks = (0,0.2,0.4,0.6,0.8,1))
    #A14
    norm = matplotlib.colors.Normalize(vmin=lims[0][0], vmax=lims[0][1])
    sample_points = np.empty((0,3))
    for i in range(rounds):
        triangleplot(points, posterior_mean[i]/axis_scale, norm,
                     cmap = 'RdBu_r',
                     cbar_label = r'$I_{c}(\theta)$ (px$\cdot$h)', #A14
                     saveas = 'Modelled-Ic-with-samples-round'+str(i), #A14
                     scatter_points=X_step[i],
                     scatter_color = np.ravel(Y_step[i]/axis_scale),
                     cbar_spacing = None, cbar_ticks = None)

