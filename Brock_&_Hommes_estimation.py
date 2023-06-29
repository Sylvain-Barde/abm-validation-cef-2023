# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 17:27:09 2023

@author: Sylvain Barde, University of Kent
"""

import time
import numpy as np
import pandas as pd

from scipy.stats import norm, skew, kurtosis
from numpy.random import default_rng
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from workshopFunctions import brockHommes, smm, npsmle, autocorr, formatTableText
from statsmodels.iolib.table import SimpleTable

#%% --------------------------------------------------------------------------
# Declare any extra functions

def hill(y,alpha):
    """
    Calculate the hill index of the data

    Parameters
    ----------
    y : numpy 1D array of floats
        input data.
    alpha : float
        Quantile for the size of the right hand side tail.

    Returns
    -------
    iota : float64
        hill index of the input data.

    """
    ysort = np.sort(y)                                # sort the returns
    n = len(y)
    k = int(np.floor(n*(1-alpha)))
    iota = 1/(np.mean(np.log(ysort[k:]) - np.log(ysort[k])))   # get the index
    return iota

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

    # Append moments below. Note the extra moments compared to the AR(2) case
    acf = autocorr(data)
    absData = abs(data)
    absAcf = autocorr(absData)
    
    moments = np.array(np.mean(absData))
    moments = np.append(moments,skew(data))
    moments = np.append(moments,kurtosis(data))

    moments = np.append(moments,acf[1])
    moments = np.append(moments,absAcf[1])
    moments = np.append(moments,(absAcf[4]+absAcf[5]+absAcf[6])/3)
    moments = np.append(moments,(absAcf[9]+absAcf[10]+absAcf[11])/3)
    moments = np.append(moments,(absAcf[24]+absAcf[25]+absAcf[26])/3)
    moments = np.append(moments,(absAcf[49]+absAcf[50]+absAcf[51])/3)
    moments = np.append(moments,(absAcf[99]+absAcf[100]+absAcf[101])/3)
    
    moments = np.append(moments,hill(absData,0.05))
    
    return moments
    
#%% --------------------------------------------------------------------------
# Simulate sythetic data
beta = 6
sigma = 1
b = 0
g = 1.2
betaVec = np.asarray([beta,sigma,b,g])
paramNames = ['Beta', 'Sigma','b_2','g_2']

seed = 400
T = 500
N = 1000
tolerance = 1e-3 # Be generous. These are MC methods!

BHsim = brockHommes(seed = 3, r = 0.1)
BHsimData = BHsim.simulate(T+3,betaVec)
y = BHsimData[3:T+3]

# Plot the data
fig = plt.figure(figsize=(16,8))
fig.set_facecolor('white')
ax = plt.axes()
ax.plot(np.arange(T), y)
ax.set_xlim(0,T)
ax.set_xlabel(r'Steps', fontdict = {'fontsize': 20},position=(1, 0))
ax.set_ylabel(r'Observations', fontdict = {'fontsize': 20},position=(0, 0.5))
ax.set_title('Synthetic Brock & Hommes data', fontdict = {'fontsize': 20});
#%% --------------------------------------------------------------------------
# Run NPSMLE and SMM on Synthetic data for parameter recovery test
print(' Running NPSMLE on synthetic Brock & Hommes data')
t_start = time.time()

init = np.asarray([2,2,0,0])
epsRng = default_rng(seed = seed)
epsilon = norm.ppf(epsRng.random((N,1)))

BHnpsmle = npsmle(y,paramNames)
negLogLik = lambda a : -sum(BHnpsmle.logLike(BHsim.step,3,a,epsilon)/T)
smleEstimation = minimize(negLogLik, 
                          init,
                          method='BFGS', 
                          callback=BHnpsmle.callback,   
                          options={'disp':True,
                                   'gtol':tolerance})
timeNpsmle = time.time() - t_start
print('Total time: {:10.4f} secs.'.format(timeNpsmle))
for name, value in zip(paramNames, smleEstimation.x):
    print('{:s} estimate: {:10.4f}'.format(name, value))


print('{:s}\n Running SMM on synthetic Brock & Hommes, W = bootstrap\n'.format(
    u'\u2500' * 72))
t_start = time.time()
BHsmm = smm(y,genMoments, paramNames, bootstrap = True) # Bootstrapped W

smmObjective = lambda a : BHsmm.dist(BHsim.simulate,a,N=500)
smmmEstimation = minimize(smmObjective, 
                          init,
                          method='BFGS', 
                          callback = BHsmm.callback,     
                          tol = 1e-2,     # add to avoid slow convergence
                          options={'disp':True,
                                    'gtol':tolerance})

timeSmm = time.time() - t_start
print('Total time: {:10.4f} secs.'.format(timeSmm))
#%% --------------------------------------------------------------------------
# Show tables to compare performance
# Gather parameter values 
parameterValues = np.concatenate(
                            (betaVec[None,:],
                             smleEstimation.x[None,:],
                             smmmEstimation.x[None,:]),
                                  axis=0)
# Gather compute times 
times = np.asarray([0,timeNpsmle,timeSmm])

# Print table
tableValues = np.concatenate((parameterValues,times[:,None]),axis = 1)
table = SimpleTable(
        formatTableText(tableValues,'{:8.3f}'),
        stubs=['True','NPSMLE', 'SMM btstrp W'],
        headers=paramNames + ['Time'],
        title='Estimation performance, parameter recovery',
    )

print(table)
print('\n')

#%% --------------------------------------------------------------------------
# We're going to load the S&P500 index data
dataPath = 'data/SP500.csv'
rawData = pd.read_csv(dataPath, index_col=[0])
rawData.index = pd.to_datetime(rawData.index)

# Let's pick the same start/end dates as Kukacka & Barunik (2017)
startDate = '1994-02-23'
endDate = '2013-12-31'
selection = (rawData.index >= startDate) & (rawData.index <= endDate)
data = rawData.loc[selection]
numRawObs = len(data)
y = np.flip(data.to_numpy()) # We need to 'numpify' the data and flip it

# Fundamental price - estimated with 61 day centered window
w = 30
numObs = numRawObs-2*w-1
yFun = np.zeros([numObs,1])
for i in range(w,numRawObs-w-1):
    yFun[i-w,:] = np.mean(y[i-w:i+w+1])
    
yDiff = y[w:numRawObs-w-1] - yFun
    
# Plot the S&P data and fundamental price, as well as deviations
fig = plt.figure(figsize=(16,8))
fig.set_facecolor('white')
ax = plt.axes()
ax.plot(np.arange(numObs), y[w:numRawObs-w-1])
ax.plot(np.arange(numObs), yFun, color = 'r')
ax.set_xlim(0,numObs)
ax.set_xlabel(r'Obs', fontdict = {'fontsize': 20},position=(1, 0))
ax.set_ylabel(r'SP500', fontdict = {'fontsize': 20},position=(0, 0.5))
ax.set_title('S&P 500 and MA-{:d} fundamental estimate'.format(w), 
             fontdict = {'fontsize': 20});

fig = plt.figure(figsize=(16,8))
fig.set_facecolor('white')
ax = plt.axes()
ax.plot(np.arange(numObs), yDiff)
ax.set_xlim(0,numObs)
ax.set_xlabel(r'Obs', fontdict = {'fontsize': 20},position=(1, 0))
ax.set_ylabel(r'SP500', fontdict = {'fontsize': 20},position=(0, 0.5))
ax.set_title('S&P 500 deviations from MA-{:d} fundamental estimate'.format(w), 
             fontdict = {'fontsize': 20});

#%% --------------------------------------------------------------------------
# Let's run an empirical estimation now

# Create a new Brock & Hommes simualtion with the Kukacka & Barunik (2017)
# risk-free rate (around 2.5% annualised)
BHsimSP500 = brockHommes(seed = 3, r = 0.0001)

#Estimate with NPSMLE
print(' Running NPSMLE on S&P 500 data')
t_start = time.time()

init = np.asarray([0,np.std(yDiff),0,0]) # Initialise a null/ pure noise model
epsRng = default_rng(seed = seed)
epsilon = norm.ppf(epsRng.random((500,1)))
BHnpsmleSP500 = npsmle(yDiff,paramNames)
negLogLik = lambda a : -sum(BHnpsmleSP500.logLike(BHsimSP500.step,3,a,epsilon)/numObs)
smleEstimation = minimize(negLogLik, 
                          init,
                          method='BFGS', 
                          callback=BHnpsmleSP500.callback,   
                          options={'disp':True,
                                   'gtol':tolerance})
timeNpsmle = time.time() - t_start
print('Total time: {:10.4f} secs.'.format(timeNpsmle))
for name, value in zip(paramNames, smleEstimation.x):
    print('{:s} estimate: {:10.4f}'.format(name, value))
    
#Estimate with SMM
print('{:s}\n Running SMM on S&P 500 data, W = bootstrap\n'.format(
    u'\u2500' * 72))
t_start = time.time()
BHsmmSP500 = smm(yDiff,genMoments, paramNames, bootstrap = True) # Bootstrapped W

smmObjective = lambda a : BHsmm.dist(BHsimSP500.simulate,a,N=500)
smmmEstimation = minimize(smmObjective, 
                          init,
                          method='BFGS', 
                          callback = BHsmmSP500.callback,     
                          tol = 1e-2,     # add to avoid slow convergence
                          options={'disp':True,
                                    'gtol':tolerance})

timeSmm = time.time() - t_start
print('Total time: {:10.4f} secs.'.format(timeSmm))

#%% --------------------------------------------------------------------------
# Show tables to compare performance on S&P 500 data
# Gather parameter values 
parameterValues = np.concatenate(
                            (np.asarray([[0.015,0.653, 0.009, 1.567]]),
                             smleEstimation.x[None,:],
                             smmmEstimation.x[None,:]),
                                  axis=0)
# Gather compute times 
times = np.asarray([0,timeNpsmle,timeSmm])

# Print table
tableValues = np.concatenate((parameterValues,times[:,None]),axis = 1)
table = SimpleTable(
        formatTableText(tableValues,'{:8.3f}'),
        stubs=['K & B (2017)','NPSMLE', 'SMM btstrp W'],
        headers=paramNames + ['Time'],
        title='Estimation performance, SP500 estimation',
    )

print(table)
print('\n')
