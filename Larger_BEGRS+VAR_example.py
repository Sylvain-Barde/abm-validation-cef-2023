# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 15:07:58 2023

@author: sb636
"""

import time
import sobol
import math

import numpy as np
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

def logP(sample):
    """
    Example of posterior function, using the BEGRS soft flat prior
    Any prior can be used, but needs to produce both the log prior and its
    gradient.    

    Parameters
    ----------
    sample : numpy 1D array of floats
        input parameter values.

    Returns
    -------
    float
        log posterior at the sample point.
    numpy 1D array of floats
        gradient of the log posterior at the input parameter values.

    """
     
    prior = begrsEst.softLogPrior(sample) 
    logL = begrsEst.logP(sample)

    return (prior[0] + logL[0], prior[1] + logL[1])
#%%---------------------------------------------------------------------------
# Generate low-discrepancy training samples
# Specify parameters
numVars = 4
numParams = numVars**2
numSamples = 1000
skip = 500
bufferRatio = 1.5
parameter_range = np.array([(-0.7, 0.7)]*numParams)
N = 200
burn = 50
numIter = 1000

# Draw from a Sobol sequence, checking VAR stability
samples_raw = getSobolSamples(int(numSamples*bufferRatio), 
                                parameter_range,skip)

samplesEigen = []
samples = []
MaxL_norms = []

for count, sample in enumerate(samples_raw):
    A = np.asmatrix(
            np.reshape(sample, (numVars,numVars))
        )

    L,U = np.linalg.eig(A)
    L_norm = np.absolute(L) 
    
    if max(L_norm) < 1:
        samplesEigen.append(L)
        samples.append(sample)
        MaxL_norms.append(max(L_norm))
    
    if len(samples) == numSamples:
        samplesEigen = np.asarray(samplesEigen)
        samples = np.asarray(samples)
        break

print('Training data:')
print(' {:d} draws required for {:d} samples'.format(count+1,numSamples))

# Simulate VAR training data
Scorr = 0.25            # Set correlation to a moderate level
C = np.asmatrix(np.zeros((numVars,1)))
S = (Scorr*np.asmatrix(np.ones(numVars)) 
     + (1-Scorr)*np.asmatrix(np.identity(numVars)))

trainData = np.zeros([N,numVars,numSamples])
for i,sample in enumerate(samples):
     A = np.asmatrix(
            np.reshape(sample, (numVars,numVars))
        )
     VARsim = VAR(C,A,S)
     trainData[:,:,i] = VARsim.simulate(N,burn,rep=i).transpose()
     
# Simulate 'emprirical' testing samples
samplesRawEmp = getSobolSamples(10, 
                                parameter_range,
                                skip + int(numSamples*bufferRatio))

for count, sample in enumerate(samplesRawEmp):
    A = np.asmatrix(
            np.reshape(sample, (numVars,numVars))
        )

    L,U = np.linalg.eig(A)
    L_norm = np.absolute(L) 
    
    if max(L_norm) < 1:
        sampleEmp = sample
        break
    
print('Testing data:')
print(' Number of draws used: {:d}'.format(count+1))

# Simulate VAR testing data 
A = np.asmatrix(
       np.reshape(sample, (numVars,numVars))
   )

numTest = 100
testData = np.zeros([N,numVars,numTest])

for i in range(numTest):
     VARsim = VAR(C,A,S)

     testData[:,:,i] = VARsim.simulate(N,burn, rep = i).transpose()
#%%----------------------------------------------------------------------------
# Some visualisations
# Plot the first dataset
fig = plt.figure(figsize=(16,8))
fig.set_facecolor('white')
ax = plt.axes()
ax.plot(np.arange(N), testData[:,0,0], color = 'b')
ax.plot(np.arange(N), testData[:,1,0], color = 'r')
ax.plot(np.arange(N), testData[:,2,0], color = 'g')
ax.plot(np.arange(N), testData[:,3,0], color = 'm')
ax.set_xlim(0,N)
ax.set_xlabel(r'Steps', fontdict = {'fontsize': 20},position=(1, 0))
ax.set_ylabel(r'Observations', fontdict = {'fontsize': 20},position=(0, 0.5))
ax.set_title('Synthetic VAR data', fontdict = {'fontsize': 20});

# Plot the first dataset - just the first 50 observations
fig = plt.figure(figsize=(16,8))
fig.set_facecolor('white')
ax = plt.axes()
ax.plot(np.arange(50), testData[0:50,0,0], color = 'b')
ax.plot(np.arange(50), testData[0:50,1,0], color = 'r')
ax.plot(np.arange(50), testData[0:50,2,0], color = 'g')
ax.plot(np.arange(50), testData[0:50,3,0], color = 'm')
ax.set_xlim(0,50)
ax.set_xlabel(r'Steps', fontdict = {'fontsize': 20},position=(1, 0))
ax.set_ylabel(r'Observations', fontdict = {'fontsize': 20},position=(0, 0.5))
ax.set_title('Synthetic VAR data', fontdict = {'fontsize': 20});

# Plot eigenvalues to ilustrate coverage of the dynamics
fontSize = 40
ax_min = -1.2
ax_max = 1.2

pi_vec = np.arange(0,2*math.pi,(2*math.pi)/1000)
eigs = samplesEigen.flatten()

fig = plt.figure(figsize=(12,12))
fig.set_facecolor('white')
ax = fig.add_subplot(1, 1, 1)
ax.plot(np.cos(pi_vec),np.sin(pi_vec),'k', linewidth=1)
ax.scatter(np.real(eigs), np.imag(eigs), color = [0.8, 0.8, 0.8], 
           label = 'Training data')
ax.scatter(np.real(L), np.imag(L), c='k',
           label = 'Testing data')
ax.legend(loc='upper center', frameon=False, ncol = 2, columnspacing = 5, 
          prop={'size':fontSize})
ax.set_ylim(top = ax_max, bottom = ax_min)
ax.set_xlim(left = ax_min,right = ax_max)
ax.plot(ax_max, 0, ">k", clip_on=False)
ax.plot(0, ax_max, "^k", clip_on=False)
ax.set_aspect('equal', 'box')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2)
ax.spines['bottom'].set_position('zero')
ax.spines['left'].set_linewidth(2)
ax.spines['left'].set_position('zero')
ax.set_xticks([-1,1])
ax.set_xticklabels(['',''])
ax.set_yticks([-1,1])
ax.set_yticklabels(['',''])
ax.xaxis.set_tick_params(length = 10, width = 2)
ax.yaxis.set_tick_params(length = 10, width = 2)
ax.annotate('1', xy=(1.05, -0.1), xytext=(1.05, -0.15), 
            size = fontSize)
ax.annotate('-1', xy=(-1.1, -0.1), xytext=(-1.15, -0.15), 
            size = fontSize)
ax.annotate('$i$', xy=(-0.1, 1.05), xytext=(-0.1, 1.05), 
            size = fontSize)
ax.annotate('$-i$', xy=(-0.15, -1.1), xytext=(-0.2, -1.1), 
            size = fontSize);

#%%----------------------------------------------------------------------------
# Train the surrogate on the VAR data

# Set the hyper parameters
num_latents = 4         # 4 here, but in general can be identified by pca
num_inducing_pts = 1000 # Subset of 'non-sparse' inputs - 500 or 1000
numiter = 50            # Number of epoch iterations (50 init)
batchSize = 5000        # Size of training minibatches (mainly a speed issue)
learning_rate = 0.05    # Learning rate (0.05 is good here)

# Create a begrs estimation object, train on simulated data
t_start = time.time()
begrsEst = begrs()
begrsEst.setTrainingData(trainData, samples, parameter_range)
begrsEst.train(num_latents, num_inducing_pts, batchSize, numiter, 
                learning_rate)
print(' Training time: {:10.4f} secs.'.format(time.time() - t_start))

# Let's save it to make sure the time cost isn't wasted...
begrsEst.save('begrsVAR')

#%%----------------------------------------------------------------------------
# Estimate on the testing VAR data

negLogLik = lambda *args: tuple( -i for i in logP(*args))

init = np.zeros(numParams)
bounds = numParams*[(-3**0.5,3**0.5)]
estimatesBegrs = []
estimatesTrue = []
for empInd in range(numTest):

    # Get  'empirical'  data, set it as the BEGRS testing set
    xEmp = testData[:,:,empInd]
    print('Empirical setting: {:d}'.format(empInd))
    
    begrsEst.setTestingData(xEmp)
    
    # Find posterior mode
    print('Finding MAP vector')
    t_start = time.time()

    sampleMAP = minimize(negLogLik, init, method='L-BFGS-B',
                          bounds = bounds, jac = True)

    rawEst = sampleMAP.x
    estimatesBegrs.append(begrsEst.uncenter(rawEst))
    print('BEGRS estim. time: {:10.4f} secs.'.format(time.time() - t_start))
    
    # Let's also get the MLE estimate for the , for comparison
    negLogLikTrue = lambda vec: -VARsim.likelihood(xEmp.transpose(),
                                                   A=np.asmatrix(
                                                       np.reshape(vec, 
                                                                  (numVars,
                                                                   numVars))
                                                                   )
                                                    )
    print('Finding Maximum likelihood vector')
    t_start = time.time()
    maximumLikelihoodVAR = minimize(negLogLikTrue, 
                              init,
                              method='BFGS', 
                              options={'gtol':1e-4})
    print('Max. Likelihood time: {:10.4f} secs.'.format(time.time() - t_start))
    estimatesTrue.append(maximumLikelihoodVAR.x)

#%%---------------------------------------------------------------------------
# Diagnostics
# Let's pick a random test series, see how we do compared to the truth and MLE
ind = 53
paramStrs = [r'A_{1,1}',
             r'A_{1,2}',
             r'A_{1,3}',
             r'A_{1,4}',
             r'A_{2,1}',
             r'A_{2,2}',
             r'A_{2,3}',
             r'A_{2,4}',
             r'A_{3,1}',
             r'A_{3,2}',
             r'A_{3,3}',
             r'A_{3,4}',
             r'A_{4,1}',
             r'A_{4,2}',
             r'A_{4,3}',
             r'A_{4,4}']

parameterValuesSingle = np.concatenate(
                            (sampleEmp[:,None],
                             estimatesBegrs[ind][:,None],
                             estimatesTrue[ind][:,None],
                             ),
                                  axis=1)    

table = SimpleTable(
        formatTableText(parameterValuesSingle,'{:8.3f}'),
        stubs=paramStrs,
        headers=['True','BEGRS', 'MLE'],
        title='Estimation performance, series {:d}'.format(ind),
    )

print(table)
print('\n')

# Now let's look at the average performance over all the test series to see
# if BEGRS deviates statistically from MLE  
estimatesBegrs = np.asarray(estimatesBegrs)
estimatesTrue = np.asarray(estimatesTrue)
    
parameterValues = np.concatenate(
                            (sampleEmp[:,None],
                             np.mean(estimatesBegrs,axis=0)[:,None],
                             np.mean(estimatesTrue,axis=0)[:,None],
                             np.mean(estimatesBegrs - estimatesTrue,
                                     axis=0)[:,None],
                             np.std(estimatesBegrs - estimatesTrue,
                                    axis=0)[:,None]
                             ),
                                  axis=1)    

table = SimpleTable(
        formatTableText(parameterValues,'{:8.3f}'),
        stubs=paramStrs,
        headers=['True','BEGRS', 'MLE', 'mean Delta','std. Delta'],
        title='Estimation performance, average',
    )

print(table)
print('\n')

