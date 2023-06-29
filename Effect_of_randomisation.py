# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 13:06:43 2023

@author: Sylvain Barde, University of Kent
"""

import time
import numpy as np
from workshopFunctions import AR, smm, npsmle, autocorr, formatTableText

from scipy.stats import norm, skew
from numpy.random import default_rng
from matplotlib import pyplot as plt

#%% --------------------------------------------------------------------------
# Declare any extra functions

def genMoments(data):
    """
    Generate the moments

    Parameters
    ----------
    data : numpy 1D array of floats
        Input data..

    Returns
    -------
    moments : numpy 1D array of floats
        Vector containing the moments of interest

    """
    data = data.flatten() # in case the data isn't already flat

    # Append moments below. This will need to change for each problem
    acf = autocorr(data)
    moments = np.array(np.mean(data))
    moments = np.append(moments,np.var(data))
    moments = np.append(moments,skew(data))
    moments = np.append(moments,acf[1])
    moments = np.append(moments,acf[2])
    
    return moments


#%% --------------------------------------------------------------------------
# Simulate synthetic 'empirical' data
betaVec = np.asarray([0.7,-0.2])    # Pick some AR parameters
T = 500                             # pick simulation length
paramNames = ['Beta_1', 'Beta_2']

ARsim = AR(seed = 3)
ARsimData = ARsim.simulate(T+2,betaVec)
y = ARsimData[2:T+2]

# Plot the data
fig = plt.figure(figsize=(16,8))
fig.set_facecolor('white')
ax = plt.axes()
ax.plot(np.arange(T), y)
ax.set_xlim(0,T)
ax.set_xlabel(r'Steps', fontdict = {'fontsize': 20},position=(1, 0))
ax.set_ylabel(r'Observations', fontdict = {'fontsize': 20},position=(0, 0.5))
ax.set_title('Synthetic AR(2) data', fontdict = {'fontsize': 20});
#%% --------------------------------------------------------------------------
# Generate optimisation surfaces in the neighbourhood of the optimum
gridRes = 25
N = 500
paramRange = np.linspace(-0.2,0.2,gridRes)
beta1Grid, beta2Grid = np.meshgrid(
                            betaVec[0] + paramRange,
                            betaVec[1] + paramRange)
surfaces = np.zeros([gridRes,gridRes,4])

epsRng = default_rng(seed = 400)
epsilon = norm.ppf(epsRng.random((N,1)))

# Generate an instance of each of the metrics (NPSMLE and SMM)
ARnpsmle = npsmle(y,paramNames)
ARsmm = smm(y,genMoments, bootstrap = True) # 'automatic' SMM, with bootstrap

# Iterate over the grid. This is the costly bit...
print('Iterating over gridpoints - nested loops, I know...')
for i in range(gridRes):
    t_start = time.time()
    for j in range(gridRes):
        paramVec = np.asarray([beta1Grid[i,j],beta2Grid[i,j]])
        
        surfaces[i,j,0] = -sum(ARnpsmle.logLike(ARsim.step,
                               2,
                               paramVec,
                               epsilon)/T)
        
        epsilon2 = norm.ppf(epsRng.random((N,1)))
        surfaces[i,j,1] = -sum(ARnpsmle.logLike(ARsim.step,
                               2,
                               paramVec,
                               epsilon2)/T)
        
        surfaces[i,j,2] = ARsmm.dist(ARsim.simulate,paramVec, N=N)
        ARsimFuzzy = AR(seed = gridRes*i + j)
        
        surfaces[i,j,3] = ARsmm.dist(ARsimFuzzy.simulate,paramVec, N=N)
        
    print(' Row {:2d} time: {:10.4f} secs.'.format(i+1,time.time() - t_start))
#%% --------------------------------------------------------------------------
# Plot the results

# Establish the objective function values  at the optimal point
npsmleTrue = -sum(ARnpsmle.logLike(ARsim.step,
                               2,
                               betaVec,
                               epsilon)/T)

smmTrue = ARsmm.dist(ARsim.simulate,betaVec, N=N)

# Plot the surfaces
elev = 60
azim = -45

labels = [r'$-\mathcal{L}$', r'$d_{SMM}$']
truths = [npsmleTrue,smmTrue]
titles = ['Likelihood','SMM']
shocks = ['constant','randomised']

for i in range (4):
    j = int(np.floor(i/2))
    k = i % 2
    fig = plt.figure(figsize=(16,8))
    fig.set_facecolor('white')
    ax = plt.axes(projection='3d')
    ax.plot_surface(beta1Grid, beta2Grid, surfaces[:,:,i], cmap='viridis')
    ax.plot3D([betaVec[0]], [betaVec[1]], [truths[j]], 
              marker = 'o', color = 'r', zorder = 5)
    ax.view_init(elev, azim)        
    ax.set_xlim(betaVec[0]-0.2,betaVec[0]+0.2)
    ax.set_ylim(betaVec[1]-0.2,betaVec[1]+0.2)
    ax.set_xlabel(r'$\beta_1$', fontdict = {'fontsize': 20},position=(1, 0))
    ax.set_ylabel(r'$\beta_2$', fontdict = {'fontsize': 20},position=(0, 1))
    ax.set_zlabel(labels[j], fontdict = {'fontsize': 20}, position=(0, 1))
    ax.set_title('{:s} surface, {:s} shocks'.format(titles[j],shocks[k]),
                 fontdict = {'fontsize': 20})
    