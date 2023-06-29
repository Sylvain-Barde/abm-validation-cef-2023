# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 10:10:01 2023

@author: Sylvain Barde, University of Kent
"""

import time
import numpy as np

from scipy.stats import norm, skew
from numpy.random import default_rng
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from workshopFunctions import AR, smm, npsmle, autocorr, formatTableText
from statsmodels.iolib.table import SimpleTable

#%% --------------------------------------------------------------------------
# Declare any extra functions

def genMoments(data):
    """
    Generate the moments

    Parameters
    ----------
    data : numpy 1D array of floats
        Input data.

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
# Run NPSMLE on Synthetic data

# Pick parameters for the MC setting
N = 1000            # Number of MC replications
tolerance = 1e-3    # Tolerance on BFGS. Be generous. These are MC methods!

# Run the estimationm itself
print(' Running NPSMLE on synthetic AR(2) data')
t_start = time.time()

init = np.asarray([0,0])      # initialise a null model
epsRng = default_rng(seed = 400)
epsilon = norm.ppf(epsRng.random((N,1)))

ARnpsmle = npsmle(y,paramNames)
negLogLik = lambda a : -sum(ARnpsmle.logLike(ARsim.step,2,a,epsilon)/T)
smleEstimation = minimize(negLogLik, 
                          init,
                          method='BFGS', 
                          callback=ARnpsmle.callback,
                          options={'disp':True,
                                    'gtol':tolerance})
timeNpsmle = time.time() - t_start
print('Total time: {:10.4f} secs.'.format(timeNpsmle))
for name, value in zip(paramNames, smleEstimation.x):
    print('{:s} estimate: {:10.4f}'.format(name, value))

#%% --------------------------------------------------------------------------
# Run SMM on Synthetic data

print('{:s}\n Running SMM on synthetic AR(2) data, W = identity\n'.format(
    u'\u2500' * 72))
t_start = time.time()
ARsmm1 = smm(y,genMoments, paramNames)     # 'default' SMM, identity matrix

smmObjective = lambda a : ARsmm1.dist(ARsim.simulate,a)
smmmEstimation1 = minimize(smmObjective, 
                          init,
                          method='BFGS', 
                          callback = ARsmm1.callback,
                          options={'disp':True,
                                    'gtol':tolerance})

timeSmm1 = time.time() - t_start
print('Total time: {:10.4f} secs.'.format(timeSmm1))
for name, value in zip(paramNames, smmmEstimation1.x):
    print('{:s} estimate: {:10.4f}'.format(name, value))

# Let's generate an ensemble of simulated data and associated moments:
momentEnsemble = np.zeros([len(ARsmm1.empMoments),N])
for rep in range(N):
    simData = ARsim.simulate(T,smmmEstimation1.x,rep)
    momentEnsemble[:,rep] = genMoments(simData)

# The new W is the inverse of the variance-covariance of the moments
momentCovariance = np.cov(momentEnsemble)
newW = np.linalg.inv(momentCovariance)

print('{:s}\n Running SMM on synthetic AR(2) data, W = two-stage\n'.format(
    u'\u2500' * 72))
t_start = time.time()
ARsmm2 = smm(y,genMoments, paramNames, W = newW) # User provided W.

smmObjective = lambda a : ARsmm2.dist(ARsim.simulate,a)
smmmEstimation2 = minimize(smmObjective, 
                          init,
                          method='BFGS', 
                          callback = ARsmm2.callback,
                          options={'disp':True,
                                    'gtol':tolerance})

timeSmm2 = time.time() - t_start
print('Total time: {:10.4f} secs.'.format(timeSmm2))
for name, value in zip(paramNames, smmmEstimation2.x):
    print('{:s} estimate: {:10.4f}'.format(name, value))

print('{:s}\n Running SMM on synthetic AR(2) data, W = bootstrap\n'.format(
    u'\u2500' * 72))
t_start = time.time()
ARsmm3 = smm(y,genMoments, paramNames, bootstrap = True) # Bootstrapped W

smmObjective = lambda a : ARsmm3.dist(ARsim.simulate,a)
smmmEstimation3 = minimize(smmObjective, 
                          init,
                          method='BFGS', 
                          callback = ARsmm3.callback,                         
                          options={'disp':True,
                                    'gtol':tolerance})

timeSmm3 = time.time() - t_start
print('Total time: {:10.4f} secs.'.format(timeSmm3))
for name, value in zip(paramNames, smmmEstimation3.x):
    print('{:s} estimate: {:10.4f}'.format(name, value))

#%% --------------------------------------------------------------------------
# Show tables to compare performance
# Gather parameter values 
parameterValues = np.concatenate(
                            (betaVec[None,:],
                             smleEstimation.x[None,:],
                             smmmEstimation1.x[None,:],
                             smmmEstimation2.x[None,:],
                             smmmEstimation3.x[None,:]),
                                  axis=0)
# Gather compute times 
times = np.asarray([0,timeNpsmle,timeSmm1,timeSmm1+timeSmm2,timeSmm3])

# Print table
tableValues = np.concatenate((parameterValues,times[:,None]),axis = 1)
table = SimpleTable(
        formatTableText(tableValues,'{:8.3f}'),
        stubs=['True','NPSMLE', 'SMM W = I', 'SMM 2-step W', 'SMM btstrp W'],
        headers=paramNames + ['Time'],
        title='Estimation performance',
    )

print(table)
print('\n')

# W matrix comparison tables
# Print the two-step matrix
table = SimpleTable(
        formatTableText(newW,'{:5.3f}'),
        title='Two-step W matrix',
    )
print(table)
print('\n')

# Print the bootstrap matrix
table = SimpleTable(
        formatTableText(ARsmm3.W,'{:5.3f}'),
        title='Bootstrapped W matrix',
    )
print(table)
