# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 08:57:15 2023

@author: sb636
"""

import time
import sobol

import numpy as np
import sampyl as smp
from begrs import begrs
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from workshopFunctions import VAR, formatTableText
from statsmodels.iolib.table import SimpleTable

#%% --------------------------------------------------------------------------
# Declare any extra functions

def getSobolSamples(numSamples, parameterSupport, skips):
    """
    Draw sobol samples out of a parameter space

    Parameters
    ----------
    numSamples : INT
        Number of samples to draw from the sequence.
    parameterSupport : numpy 2D array of floats
        Bounds for the parameter values.
    skips : INT
        Number of initial draws to skip (i.e. start deeper in the sequence).

    Returns
    -------
    sobol_samples : numpy 2D array of floats
        Draws from the Sobol sequence.

    """
    params = np.transpose(parameterSupport)
    sobol_samples = params[0,:] + sobol.sample(
                        dimension = parameterSupport.shape[0], 
                        n_points = numSamples, 
                        skip = skips
                        )*(params[1,:]-params[0,:])
    
    return sobol_samples

#%%---------------------------------------------------------------------------
# Pick some fixed parameters
N = 200                     # 200 empirical observations
burn = 50                   # burn-in period for simulation
C = np.asmatrix([[0],[0]])  # zero mean
S = np.asmatrix([[1,0.5],   # Variance/covariance of innovations
                 [0.5,1]])

# Generate sobol samples for parameter space
numSamples = 40
parameterRange = np.array([
        (-0.5, 0.5),       # a1 param
        (-0.5, 0.5)])      # a2 param
samples = getSobolSamples(numSamples, parameterRange, skips = 500)

# Generate training data from samples
numParams = 2
xTrain = np.zeros([N,numParams,numSamples])
for i, sample in enumerate(samples):
    A = np.asmatrix([[0.6,sample[0]],
                     [sample[1],0.6]])
    VARsimTrain = VAR(C,A,S)
    xTrain[:,:,i] = VARsimTrain.simulate(N,burn).transpose()

# Generate 'empirical' series
trueParams = np.asarray([-0.25,0.15])
A = np.asmatrix([[0.6,trueParams[0]],
                 [trueParams[1],0.6]])

VARsim = VAR(C,A,S)
xEmp = VARsim.simulate(N,burn)
Lemp = VARsim.likelihood(xEmp)

# Plot the data
fig = plt.figure(figsize=(16,8))
fig.set_facecolor('white')
ax = plt.axes()
ax.plot(np.arange(N), xEmp[0,:])
ax.plot(np.arange(N), xEmp[1,:], color = 'r')
ax.set_xlim(0,N)
ax.set_xlabel(r'Steps', fontdict = {'fontsize': 20},position=(1, 0))
ax.set_ylabel(r'Observations', fontdict = {'fontsize': 20},position=(0, 0.5))
ax.set_title('Synthetic VAR data', fontdict = {'fontsize': 20});
#%%---------------------------------------------------------------------------
# Train the surrogate
# Set the hyper parameters
num_latents = 2         # We can set 2, small bivariate example
num_inducing_pts = 20   # Subset of 'non-sparse' inputs - is 20 enough?
numiter = 75            # Number of epoch iterations
batchSize = 10000       # Size of training minibatches - Can do all here
learning_rate = 0.1     # Learning rate (0.1 is good here)

# Create a begrs estimation object, train the surrogate on simulated data
t_start = time.time()
begrsEst = begrs()
begrsEst.setTrainingData(xTrain, samples, parameterRange)
begrsEst.train(num_latents, num_inducing_pts, batchSize, numiter, 
                learning_rate)
print(' Training time: {:10.4f} secs.'.format(time.time() - t_start))

#%%---------------------------------------------------------------------------
# First, we need to declare the log posterior to be optimised:
def logP(sample):
    # Example of posterior function, using the BEGRS soft flat prior
    # Any prior can be used, but needs to produce both the log prior and its
    # gradient.
    
    prior = begrsEst.softLogPrior(sample) 
    logL = begrsEst.logP(sample)

    return (prior[0] + logL[0], prior[1] + logL[1])

# Estimate the parameters from the empirical data
# Set testing data
begrsEst.setTestingData(xEmp.transpose())

# Find posterior mode
negLogLik = lambda *args: tuple( -i for i in logP(*args))
print('Finding MAP vector')
t_start = time.time()

init = np.zeros(numParams)
bounds = numParams*[(-3**0.5,3**0.5)]
sampleMAP = minimize(negLogLik, init, method='L-BFGS-B',
                      bounds = bounds, jac = True)
print('BEGRS maximisation time: {:10.4f} secs.'.format(time.time() - t_start))
mapEst = begrsEst.uncenter(sampleMAP.x)

# Let's see what the maximum likelihood estimate is for comparison
negLogLikTrue = lambda vec:  -VARsim.likelihood(xEmp,
                                                A=np.asmatrix([[0.6,vec[0]],
                                                               [vec[1],0.6]]
                                                              )
                                                )
print('Finding Maximum likelihood vector')
t_start = time.time()
maximumLikelihoodVAR = minimize(negLogLikTrue, 
                          init,
                          method='BFGS', 
                          options={'disp':True,
                                   'gtol':1e-4})
print('Max. Likelihood time: {:10.4f} secs.'.format(time.time() - t_start))

# Show tables to compare performance
# Gather parameter values 
parameterValues = np.concatenate(
                            (trueParams[None,:],
                             mapEst[None,:],
                             maximumLikelihoodVAR.x[None,:]),
                                  axis=0)
# Print table
table = SimpleTable(
        formatTableText(parameterValues,'{:8.3f}'),
        stubs=['True','BEGRS', 'MLE'],
        headers=['theta 1','theta 2'],
        title='Estimation performance',
    )

print(table)
print('\n')

#%%---------------------------------------------------------------------------
# Generate Likelihood surfaces for visualisation
res = 40
paramBase = np.linspace(-0.5,0.5,res)
theta1, theta2 = np.meshgrid(paramBase, paramBase)
logLikSurf = np.zeros([res,res])
logLikSurfGP = np.zeros([res,res])
dTheta_1 = np.zeros([res,res])
dTheta_2 = np.zeros([res,res])

print(' Getting surrogate likelihood surface in sample space')
t_total = time.time()
for i in range(res):
    t_start = time.time()

    for j in range(res):

        theta = begrsEst.center(        # Don't forget to center!!
                            np.asarray(
                                [theta1[i,j], theta2[i,j]]
                                    )
                                )
        
        logLikInfo = logP(theta)
        logLikSurfGP[i,j] = logLikInfo[0]
        dTheta_1[i,j] = logLikInfo[1][0]
        dTheta_2[i,j] = logLikInfo[1][1]
    print(' Row {:2d} time: {:10.4f} secs.'.format(i+1,time.time() - t_start))

print(' Likelihood GP surf:    {:10.4f} secs.'.format(time.time() - t_total))

print('\n Getting true likelihood surface in sample space')
t_total = time.time()
for i in range(res):
    t_start = time.time()
    for j in range(res):
        A = np.asmatrix([[0.6,theta1[i,j]],
                     [theta2[i,j],0.6]])
        VARsimSurf = VAR(C,A,S)
        logLikSurf[i,j] = VARsimSurf.likelihood(xEmp)
    print(' Row {:2d} time: {:10.4f} secs.'.format(i+1,time.time() - t_start))

print(' Likelihood true surf: {:10.4f} secs.'.format(time.time() - t_total))

# Plot surfaces
fig = plt.figure(figsize=(20,10))
fig.set_facecolor('white')
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.plot_surface(theta1, theta2, logLikSurf, cmap='viridis')
ax.plot3D([maximumLikelihoodVAR.x[0]], [maximumLikelihoodVAR.x[1]], 
          [-maximumLikelihoodVAR.fun], marker = 'o', color = 'r', zorder = 5)
ax.set_xlabel(r'$param 1$', fontdict = {'fontsize': 20},position=(1, 0))
ax.set_ylabel(r'$param 2$', fontdict = {'fontsize': 20},position=(0, 1))
ax.set_title('True likelihood surface', fontdict = {'fontsize': 20})
    

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(theta1, theta2, logLikSurfGP, cmap='viridis')
ax.plot3D([mapEst[0]], [mapEst[1]], [-sampleMAP.fun], 
              marker = 'o', color = 'r', zorder = 5)
ax.set_xlabel(r'$param 1$', fontdict = {'fontsize': 20},position=(1, 0))
ax.set_ylabel(r'$param 2$', fontdict = {'fontsize': 20},position=(0, 1))
ax.set_title('Surrogate likelihood surface', fontdict = {'fontsize': 20})

fig = plt.figure(figsize=(10,10))
fig.set_facecolor('white')
ax = fig.add_subplot(1, 1, 1)
ax.quiver(theta1, theta2, dTheta_1, dTheta_2)
ax.plot(mapEst[0], mapEst[1], marker = 'o', color = 'r')
ax.set_title('Surrogate likelihood gradient', fontdict = {'fontsize': 20})

#%%----------------------------------------------------------------------------
# Let's go full Bayesian. After all, it's 'BEGRS', not 'MLEGRS'...
print(' NUTS estimation on Synthetic data')
start = smp.state.State.fromfunc(logP)
start.update({'sample': sampleMAP.x})

scale = 1*np.ones([len(sampleMAP.x)])
E_cutoff = -2*logP(sampleMAP.x)[0]

nuts = smp.NUTS(logP, start, scale = {'sample': scale},
                grad_logp = True, 
                step_size = 0.01, 
                Emax = E_cutoff) 
chain = nuts.sample(5100, burn=100)     # 5000 samples should be enough
print(' HMC time: {:10.4f} mins.'.format((time.time() - t_start)/60))

# Uncenter the samples
flat_samples = np.zeros_like(chain.sample)
for i, sampleCntrd in enumerate(chain.sample):
    flat_samples[i,:] = begrsEst.uncenter(sampleCntrd)

# Get expected values
theta_mean = np.mean(flat_samples, axis = 0)

# Generate plots - most of this is just 'prettifying code' to make a nice plot
fontSize = 20
for i in range(2):
    d = flat_samples[:,i]    
    x_range = parameterRange[i,1] - parameterRange[i,0]
    xlim_left = parameterRange[i,0] - x_range*0.025
    xlim_right = parameterRange[i,1] + x_range*0.025
    
    fig = plt.figure(figsize=(16,8))
    fig.set_facecolor('white')
    ax = plt.axes()
    posteriorDist = ax.hist(x=d, bins='fd', density = True, edgecolor = 'black', 
                  color = 'gray', alpha=0.4)
    y_max = 1.25*max(posteriorDist[0])
    ax.plot([trueParams[i],trueParams[i]], [0, y_max], linestyle = (0, (5, 10)), 
            linewidth=1, color = 'r', alpha=0.6, 
            label = 'truth')
    ax.plot([theta_mean[i],theta_mean[i]], [0, y_max], 'r',  linewidth=1, 
            alpha=0.6, label = 'mean')
    ax.set_xlabel(r'$\theta_{:d}$'.format(i+1), 
                  fontdict = {'fontsize': fontSize})
    ax.set_ylabel(r'$p(\theta_{:d})$'.format(i+1), rotation=0, labelpad=100,
                  fontdict = {'fontsize': fontSize})
    ax.xaxis.set_label_coords(.925, -.06)
    ax.yaxis.set_label_coords(-.06, .925)
    ax.axes.yaxis.set_ticks([])
    ax.legend(loc='best', frameon=False, prop={'size':fontSize})

    ax.set_ylim(top = y_max, bottom = 0)
    ax.set_xlim(left = xlim_left,right = xlim_right)
    ax.plot(xlim_right, 0, ">k", clip_on=False)
    ax.plot(xlim_left, y_max, "^k", clip_on=False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.tick_params(axis='x', labelsize=fontSize)    
    ax.tick_params(axis='y', labelsize=fontSize)