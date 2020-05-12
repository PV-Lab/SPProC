#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 16:51:39 2019

@author: Shreyaa
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
        # create a triangulation out of these points
        T_p = tri.Triangulation(x_p,y_p)

        im3 = ax.scatter(x_p, y_p, s=8, c=scatter_color, cmap=cmap, edgecolors='black', linewidths=.5, alpha=1, zorder=2, norm=norm)
    # plot the contour
    if surf_levels is not None:
        im=ax.tricontourf(x,y,T.triangles,v, cmap=cmap, levels=surf_levels)
    else:
        im=ax.tricontourf(x,y,T.triangles,v, cmap=cmap)
    im.set_norm(norm)
    # create the grid
    corners = np.array([[0, 0], [1, 0], [0.5,  np.sqrt(3)*0.5]])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    # creating the grid
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=0)
    #plotting the mesh
    im2=ax.triplot(trimesh,'k-', linewidth=0.5)
    
    myformatter=matplotlib.ticker.ScalarFormatter()
    myformatter.set_powerlimits((0,2))
    if cbar_spacing is not None:
        cbar=plt.colorbar(im, ax=ax, spacing='proportional', ticks=(0,0.2,0.4,0.6,0.8,1))
    else:
        cbar=plt.colorbar(im, ax=ax, format=myformatter)
    cbar.set_label(cbar_label)#, fontsize=14)
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
    if saveas:
        fig.savefig(results_dir + saveas + '.pdf')
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
    
    for i in range(rounds):
        suggestion_df[i].to_csv('Bayesian_suggestion_round_'+str(i)+'.csv', float_format='%.3f')
        inputs = compositions_input[i]
        inputs['Merit'] = degradation_input[i]['Merit']
        inputs=inputs.sort_values('Merit')
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
        inputs2['Posterior mean (a.u.)']=posterior_mean[i]
        inputs2['Posterior std (a.u.)']=posterior_std[i]
        inputs2['Aqcuisition (a.u.)']=acq_normalized[i]
        inputs2.to_csv(results_dir + 'Model_round_' + str(i) + '.csv', float_format='%.3f')
        
    
    
    
    ###############################################################################
    # PLOT MERIT
    ###############################################################################
    # Let's plot the resulting suggestions. This plot works for 3 materials only.
    # Colorbar scale:
    scale = 1
    
    # Min and max values for each contour plot are determined and normalization
    # of the color range is calculated.
    axis_scale = 100000
    lims = [[np.min(posterior_mean)/axis_scale, np.max(posterior_mean)/axis_scale],
            [np.min(posterior_std)/axis_scale, np.max(posterior_std)/axis_scale],
            [np.min(acq_normalized), np.max(acq_normalized)]] # For mean, std, and acquisition.
    norm = matplotlib.colors.Normalize(vmin=lims[0][0], vmax=lims[0][1])
    
    for i in range(rounds):
        triangleplot(points, posterior_mean[i], norm, surf_axis_scale = axis_scale, cmap = 'RdBu_r',
                 cbar_label = 'Posterior mean (a.u.)', saveas = 'Posterior-mean-no-grid-round'+str(i))


    norm = matplotlib.colors.Normalize(vmin=lims[1][0], vmax=lims[1][1])
    for i in range(rounds):
        triangleplot(points, posterior_std[i], norm, surf_axis_scale = axis_scale, cmap = 'RdBu_r',
                 cbar_label = 'Post. Std.Dev. (a.u.)', saveas = 'Posterior-StdDev-round'+str(i))

    norm = matplotlib.colors.Normalize(vmin=lims[2][0], vmax=lims[2][1])
    # Shift the colormap (removes the red background that appears with the std colormap)
    orig_cmap = mpl.cm.RdBu
    shifted_cmap = shiftedColorMap(orig_cmap, start=0, midpoint=0.000005, stop=1, name='shifted')
    # Colors of the samples in each round.     
    newPal = {0:'#8b0000', 1:'#7fcdbb', 2:'#9acd32', 3:'#ff4500', 4: 'k'}
    for i in range(rounds):
        print('i is ', i)
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
                     cbar_label = 'Acquisition Prob. (a.u.)',
                     saveas = 'Acquisition-round'+str(i)+'-with-colored-samples-from-single-round',
                     surf_levels = (0,0.009,0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8,1),
                     scatter_points=X_rounds[i], scatter_color = newPal[i],
                     cbar_spacing = 'proportional',
                     cbar_ticks = (0,0.2,0.4,0.6,0.8,1))
    
    ### Here the posterior mean is plotted together with the samples.
    norm = matplotlib.colors.Normalize(vmin=lims[0][0], vmax=lims[0][1])
    sample_points = np.empty((0,3))
    for i in range(rounds):
        triangleplot(points, posterior_mean[i], norm,
                     surf_axis_scale = axis_scale, cmap = 'RdBu_r',
                     cbar_label = 'Posterior mean (a.u.)',
                     saveas = 'Posterior-mean-with-colored-samples-round'+str(i),
                     scatter_points=X_step[i],
                     scatter_color = np.ravel(Y_step[i]/axis_scale),
                     cbar_spacing = None, cbar_ticks = None)

