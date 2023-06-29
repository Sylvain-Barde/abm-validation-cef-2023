# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 16:21:05 2023

Contains the classes and functions required to run the examples presented in 
the "Agent-based Models: Methodology, Calibration and Estimation" 
pre-conference workshop of the 2023 CEF conference, Nice.

Contents:
    Three model/simulation classes:
        - AR: Simple auto-regressive simulation class
        - brockHommes: Simulation class for the Brock & Hommes (1998) model
        - VAR: Simple 1-lag vector auto-regression simulation class
    
    Two estimation classes
        - npsmle: Non-parametric Simulated Maximum Likelihood estimate class
        - smm: Simulated method of moments class
        
    Two utilities/quality-of-life functions
        - autocorr: Calculate the autocorrelation function
        - formatTableText: Convert a 2D numpy array into a 2D list of 
          formatted strings 

@author: Sylvain Barde, University of Kent
"""


import time
import math
import numpy as np
from scipy.stats import norm
from numpy.random import default_rng
from sklearn.neighbors import KernelDensity
from arch.bootstrap import StationaryBootstrap

#-----------------------------------------------------------------------------
class AR:
    """
    Simple auto-regressive simulation class

        Attributes:
            seed (int): Set the seed for the simulations

        Methods:
            __init__:
                Initialises the class
            simulate:
                Simulate an AR process
            step: 
                Simulate a single time step for multiple shocks              

    """
    
    def __init__(self,seed = 0):
        """
        Initialises the class 

        Parameters
        ----------
        seed : INT, optional
            Seed for the RNG. The default is 0.

        Returns
        -------
        None.

        """
        self.seed = seed
        
    def simulate(self,T,params,rep=0):
        """
        Simulate an AR process

        Parameters
        ----------
        T : INT
            Number of timesteps requires for the simulation.
        params : numpy 1D array of floats
            Vector of autoregressive parameters.
        rep : INT, optional
            Offset for RNG seed. The default is 0.

        Returns
        -------
        y : numpy 1D array of floats
            Simulated autoregressive series.

        """
        numLags = len(params)
        y = np.zeros(T)
        rng = default_rng(self.seed+rep)
        shock = norm.ppf(rng.random((T)))
        for t in range(numLags,T):
            y[t] = self.step(y[t-numLags:t],params,shock[t])
        
        return y
        
    def step(self,yLag,params,shock):
        """
        Simulate a single time step for multiple shocks

        Parameters
        ----------
        yLag : numpy 1D array of floats
            Lagged values of the AR process.
        params : numpy 1D array of floats
            Vector of autoregressive parameters.
        shock : numpy 1D array of floats
            Vector of shocks.

        Returns
        -------
        y : numpy 1D array of floats
            Vector of on-step-ahead AR values.

        """
        y = np.flip(yLag[None,:]) @ params[:,None] + shock

        return y
#-----------------------------------------------------------------------------
class brockHommes:
    """
    Simulation class for the Brock & Hommes (1998) model

        Attributes:
            seed (int): Set the seed for the simulations
            R: risk free interest rate term

        Methods:
            __init__:
                Initialises the class 
            simulate:
                Simulate an AR process 
            step: 
                Simulate a single time step for multiple shocks              

    """
    
    def __init__(self,seed = 0,r=0.01):
        """
        Initialises the class         

        Parameters
        ----------
        seed : INT, optional
            Seed for the RNG. The default is 0.
        r : FLOAT, optional
            Risk-free rate of return. The default is 0.01.

        Returns
        -------
        None.

        """
        self.seed = seed
        self.R = 1 + r
        
    def simulate(self,T,params,rep=0):
        """
        Simulate the Brock & Hommes (1998) model

        Parameters
        ----------
        T : INT
            Number of timesteps requires for the simulation.
        params : numpy 1D array of floats
            Vector of autoregressive parameters.
        rep : INT, optional
            Offset for RNG seed. The default is 0.

        Returns
        -------
        y : numpy 1D array of floats
            Simulated series.

        """
        y = np.zeros(T)
        rng = default_rng(self.seed+rep)
        shock = norm.ppf(rng.random((T)))
        for t in range(3,T):
            y[t] = self.step(y[t-3:t],params,shock[t])
        
        return y
        
    def step(self,yLag,params,shock):
        """
        Simulate a single time step for multiple shocks

        Parameters
        ----------
        yLag : numpy 1D array of floats
            Lagged values of the AR process.
        params : numpy 1D array of floats
            Vector of autoregressive parameters.
        shock : numpy 1D array of floats
            Vector of shocks.

        Returns
        -------
        y : numpy 1D array of floats
            Vector of on-step-ahead values for the Borck & Hommes (1998) model.

        """
                
        # Unpack parameters
        beta = params[0]
        sigma = params[1]
        numStrategies = int(1 + (len(params)-2)/2)
        b = np.zeros([numStrategies,1])
        b[1:numStrategies,0] = params[2:1+numStrategies]
        g = np.zeros([numStrategies,1])
        g[1:numStrategies,0] = params[1+numStrategies:]
        
        # Calculate Brock & Hommes updating equations
        beliefs = g*yLag[-1] + b
        lagBeliefTerm = (g*yLag[-3] + b - self.R*yLag[-2])
        expBU = np.exp(
                    np.clip(        # Clip values to avoid +/- inf
                        beta*(yLag[-1] - self.R*yLag[-2])*lagBeliefTerm,
                        -400,400
                        )
                    )
        n = expBU/sum(expBU)
        y = (np.transpose(beliefs) @ n + sigma*shock)/self.R
        
        return y
#-----------------------------------------------------------------------------
class VAR:
    """
        Simple 1-lag vector auto-regression simulation class

        Attributes:
            seed (int): Set the seed for the simulations
            C: vector of constants for the VAR
            A: Matrix of parameters for the first lag
            S: variance-covariance matrix for innovations

        Methods:
            __init__:
                Initialises the class 
            simulate:
                Simulate a VAR(1) process 
            likelihood: 
                Calculate the likelihood given data and parameters

    """
    
    def __init__(self, C, A, S, seed = 0):
        """
        Initialises the class 

        Parameters
        ----------
        C : numpy 2D Matrix of floats
            vector of constants for the VAR.
        A : numpy 2D Matrix of floats
            Matrix of parameters for the first lag.
        S : numpy 2D Matrix of floats
            variance-covariance matrix for innovations.
        seed : INT, optional
            Seed for the RNG. The default is 0.

        Returns
        -------
        None.

        """
        self.C = C
        self.A = A
        self.S = S
        self.seed = seed
    
    def simulate(self, N, burn, rep=0):
        """
        Simulate a VAR(1) process 

        Parameters
        ----------
        T : INT
            Number of timesteps requires for the simulation.
        burn : INT
            Number of burn-in periods for the simulation.
        rep : INT, optional
            Offset for RNG seed. The default is 0.

        Returns
        -------
        numpy 1D array of floats
            Simulated series.

        """
        rng = default_rng(self.seed+rep)

        num_vars = len(self.C)
        L,U = np.linalg.eig(self.S)
        Sroot = np.asmatrix(U)*np.diag(L)**0.5
        
        # Correlated shocks for VAR
        e = Sroot*norm.ppf(rng.random([num_vars,N+burn]))   
        
        x_full = np.zeros([num_vars,N+burn])
        for i in range(1,N+burn):
            x_full[:,i:i+1] = self.C + self.A*x_full[:,i-1,None] + e[:,i]
        
        return x_full[:,burn:N+burn]
    
    def likelihood(self, x, C=None,A=None,S=None):
        """
        Calculate the likelihood given data and parameters

        Parameters
        ----------
        x : numpy 1D array of floats
            Empirical data to be used in the likelihood.
        C : numpy 2D Matrix of floats, optional
            Parameter value to evaluate likelihood. The default is None.
        A : numpy 2D Matrix of floats, optional
            Parameter value to evaluate likelihood. The default is None.
        S : numpy 2D Matrix of floats, optional
            Parameter value to evaluate likelihood. The default is None.

        Returns
        -------
        L : float64
            Likelihood of the parameters given empirical data.

        """
        if C is None:
            C = self.C
        if A is None:
            A = self.A
        if S is None:
            S = self.S
        
        N = x.shape[1]
        u = np.zeros(N)
        Sinv = np.linalg.inv(S)
        D = np.linalg.det(S)
    
        for i in range(1,N):
            u_base = x[:,i:i+1] - C - A*x[:,i-1:i]
            u[i] = np.transpose(u_base)*Sinv*u_base
            
        L = -0.5*(N-1)*(2*np.log(2*math.pi) + np.log(D)) - sum(u)/2
        return L

#-----------------------------------------------------------------------------
class npsmle:    
    """
    Non-parametric Simulated Maximum Likelihood estimate class
    Provides a non-parametric (Kernel-based) estimate of the likelihood
    function following Kristensen & Shin (2012), journal of Econometrics

        Attributes:
            iterCount (int): Number of times the callback has been called
            timer (float): timestamp used in callback method
            y (nd.array): vector of empirical data
            names (list): list of parameter names (for display)

        Methods:
            __init__:
                Initialises the class
            logLike:
                Calculates the NPSMLE for a parameter vector on a dataset
            callback: 
                Used by the optimiser in each iteration to provide feedback on
                the state of the optimisation

    """
    
    def __init__(self,empData, paramNames = []):
        """
        Initialises the class

        Parameters
        ----------
        empData : numpy 1D array of floats
            Empirical data to be used in the estimation.
        paramNames : list of strings, optional
            Parameter names for the callback function. The default is [].

        Returns
        -------
        None.

        """
        self.iterCount = 0
        self.timer = time.time()
        self.y = empData
        self.names = paramNames

    
    def logLike(self,fun,numLags,theta,epsilon):
        """
        Calculates the NPSMLE for a parameter vector on a dataset

        Parameters
        ----------
        fun : Function handle
            Function that calculates the one-step-ahead simulations for the
            model being estimated.
        numLags : INT
            Number of lags to include in one-step-ahead simulation.
        theta : numpy 1D array of floats
            Parameter vector to be passed to the one-step-ahead simulation.
        epsilon : numpy 1D array of floats
            Vector of MC shocks to be passed to the one-step-ahead simulation.

        Returns
        -------
        logVec : numpy 1D array of floats
            Vector containing the contributions of each empirical observation 
            to the NPSMLE likelkelihood.

        """
        N = len(epsilon)
        logVec = np.zeros(len(self.y))
        
        for i in range(numLags,len(self.y)):
        
            ySim = fun(self.y[i-numLags:i],theta,epsilon)
            yDiff = ySim - self.y[i]
            b = np.std(yDiff)*(4/(3*N))**(1/5)
            kde_y = KernelDensity(bandwidth = b, kernel='gaussian')
            kde_y.fit([[0]])
            logprob = kde_y.score_samples(yDiff)
            logVec[i] = sum(logprob)
            
        return logVec

    def callback(self, xk, *_):
        """
        Used by the optimiser in each iteration to provide feedback on
        the state of the optimisation
    
        Parameters
        ----------
        xk : numpy 1D array of floats
            Current parameter values.
        *_ : placholder
            ignores further arguments (callbacks are optimiser dependent).

        Returns
        -------
        None.

        """
        newTimer = time.time()
        
        # Create header on first iteration (counter is 0)
        if not self.iterCount:
            header = '{:^11s}  '.format('iteration')
            header += '{:^11s}  '.format('time')
            # Use names if provided
            if len(xk) == len(self.names):
                for j, name in enumerate(self.names):
                    header += f"{name:^11s}  "
            else:
                for j, _ in enumerate(xk):
                    tmp = 'param-{:d}'.format(j+1)
                    header += f"{tmp:^11s}  "
                
            print(header)
            print(u'\u2500' * 13*(j+3))
            
        # Increment counter, calculate time on interation
        self.iterCount += 1
        line = '{:^11d}  '.format(self.iterCount)
        line+= '{:>11.4e}  '.format(newTimer - self.timer)
        
        # Format parameter values and display
        for param in xk:
            line += f"{param:>11.4e}  "
        print(line)
        self.timer = newTimer

class smm:
    """
    Simulated method of moments class
    Provides a simulated distance between moments, follwoing McFadden, (1989), 
    Econometrica

        Attributes:
            iterCount (int): Number of times the callback has been called
            timer (float): timestamp used in callback method
            T (int): length of the empirical data
            momentFun: handle to the moment-generating function
            empMoments: (target) moments of the empirical data
            W: weights matrix            
            names (list): list of parameter names (for display)


        Methods:
            __init__:
                Initialises the class
            dist:
                Calculates SMM distance for a parameter vector on a dataset
            callback: 
                Used by the optimiser in each iteration to provide feedback on
                the state of the optimisation

    """
    
    def __init__(self,
                 empData, 
                 momentFun,
                 paramNames = [],
                 bootstrap = False, 
                 W = None, 
                 numBoot=1000, 
                 b=5):
        """
        Initialises the class

        Parameters
        ----------
        empData : numpy 1D array of floats
            Empirical data to be used in the estimation.
        momentFun : function
            moment-generating function for the empirical/simulated data.
        paramNames : list of strings, optional
            Parameter names for the callback function. The default is [].
        bootstrap : Boolean, optional
            Set the W matrix using a bootstrap. The default is False.
        W : numpy 2D array, optional
            User-provided weights matrix. The default is None, which generates
            an identity matrix for W.
        numBoot : INT, optional
            Number of bootstrap replications. The default is 1000.
        b : INT, optional
            Width of the bootstrap window. The default is 5.

        Returns
        -------
        None.

        """
        self.iterCount = 0
        self.timer = time.time()
        self.T = len(empData)
        self.momentFun = momentFun
        self.empMoments = self.momentFun(empData)
        self.names = paramNames
        if bootstrap:
            btstrp = StationaryBootstrap(b,empData)
            sig = btstrp.cov(self.momentFun,numBoot)            
            self.W = np.linalg.inv(sig)
        elif W is None:
            self.W = np.identity(len(self.empMoments))
        else:
            self.W = W
        
    def dist(self,fun,theta,N=1000):
        """
        Calculates SMM distance for a parameter vector on a dataset

        Parameters
        ----------
        fun : Function handle
            Function that runs the simulations for the model being estimated.
        theta : numpy 1D array of floats
            Parameter vector to be passed to the simulation.
        N : INT, optional
            Number of MC replications. The default is 1000.

        Returns
        -------
        dist : float
            SMM distance between empirical and simulated moments.

        """
        
        # Iterate over replications
        for rep in range(N):
            simData = fun(self.T,theta,rep)
            if rep == 0:
                simMoments = self.momentFun(simData)
            else:
                simMoments += self.momentFun(simData)
        simMoments /= N
        
        # Calculate distance metric
        diff = (simMoments - self.empMoments) / self.empMoments
        dist = (diff[None,:] @ self.W @ diff[:,None]).flatten()[0]
    
        return dist
        
        
    def callback(self, xk, *_):
        """
        Used by the optimiser in each iteration to provide feedback on
        the state of the optimisation
    
        Parameters
        ----------
        xk : numpy 1D array of floats
            Current parameter values.
        *_ : placholder
            ignores further arguments (callbacks are optimiser dependent).

        Returns
        -------
        None.

        """
        newTimer = time.time()
        
        # Create header on first iteration (counter is 0)
        if not self.iterCount:
            header = '{:^11s}  '.format('iteration')
            header += '{:^11s}  '.format('time')
            # Use names if provided
            if len(xk) == len(self.names):
                for j, name in enumerate(self.names):
                    header += f"{name:^11s}  "
            else:
                for j, _ in enumerate(xk):
                    tmp = 'param-{:d}'.format(j+1)
                    header += f"{tmp:^11s}  "
                
            print(header)
            print(u'\u2500' * 13*(j+3))
            
        # Increment counter, calculate time on interation
        self.iterCount += 1
        line = '{:^11d}  '.format(self.iterCount)
        line+= '{:>11.4e}  '.format(newTimer - self.timer)
        
        # Format parameter values and display
        for param in xk:
            line += f"{param:>11.4e}  "
        print(line)
        self.timer = newTimer
#-----------------------------------------------------------------------------
def autocorr(x):
    """
    Calculate the autocorrelation function

    Parameters
    ----------
    x : numpy 1D array of floats
        Input data.

    Returns
    -------
    numpy 1D array of floats
        Autocorrelation function of x.

    """    
    raw = np.correlate(x, x, mode='full')
    result = raw[raw.size // 2:]
    return result/result[0]
#-----------------------------------------------------------------------------
def formatTableText(array,formatStr):
    """
    Convert a 2D numpy array into a 2D list of formatted strings 

    Parameters
    ----------
    array : 2D nd.array
        array containg values to be formatted.
    formatStr : str
        format required for strings.

    Returns
    -------
    tableValuesFormatted : List
        2D list of formatted strings.

    """
    
    tableValuesFormatted = []
    for row in array:
        rowFormatted = []
        for cellValue in row:
            rowFormatted.append(formatStr.format(cellValue))
            
        tableValuesFormatted.append(rowFormatted)
        
    return tableValuesFormatted