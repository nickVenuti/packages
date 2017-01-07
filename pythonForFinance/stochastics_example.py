# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:23:31 2015

@author: nmvenuti
Stocastic Example
Python for Finance
"""

#Module imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import path
path.append("C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance")
from bsm_functions import *


#Parameters
S0=100.0 #inital value
K=105.0 #strike price
T=1.0 #maturity
r=0.05 #riskless short rate
sigma=.2 #volatility
M=50 #number of steps
dt=T/M #length of time interval
I=250000 #number of paths


#create benchmark for future Monte Carlo simulation
benchmark_value=bsm_call_value(S0,K,T,r,sigma)

##################################
######Monte Carlo simulations#####
##################################

#
#1 Monte Carlo valuation of European call options with pure python
#Python for finance script mcs_pure_python.py
#
from time import time
from math import exp,sqrt,log
from random import gauss,seed

seed(20000)
t0=time()

#Simulating I paths with M time steps
S=[]
for i in range(I):
    path=[]
    for t in range(M+1):
        if t==0:
            path.append(S0)
        else:
            z = gauss(0,1)
            St=path[t-1]*exp((r-.5*sigma**2)*dt+sigma*sqrt(dt)*z)
            path.append(St)
    S.append(path)

#Calculating the Monte Carlo estimator
C0=exp(-r*T)*sum([max(path[-1]-K,0) for path in S])/I

#Results output
tpy=time()-t0
print "European Option Value %7.3f" % C0
print "Duration in Seconds %7.3f"%tpy
#European Option Value   7.999
#Duration in Seconds  21.761


#
#2 Monte Carlo valuation of European call options with NumPy
#Python for finance script mcs_vector_numpy.py
#
import math
import numpy as np
from time import time
np.random.seed(20000)
t0=time()

#Simulating I paths with M time steps
S=np.zeros((M+1,I))
S[0]=S0
for t in range(1,M+1):
    z=np.random.standard_normal(I) #psuedorandom numbers
    S[t]=S[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*math.sqrt(dt)*z)

#Calculating the Monte Carlo estimator
C0=math.exp(-r*T)*np.sum(np.maximum(S[-1]-K,0))/I

#Results output
tpy=time()-t0
print "European Option Value %7.3f" % C0
print "Duration in Seconds %7.3f"%tpy
#European Option Value   8.037
#Duration in Seconds   0.741

#
#3 Monte Carlo valuation of European call options with NumPy and log Euler discretization of SDE
#Python for finance script mcs_full_vector_numpy.py
#
import math
from numpy import *
from time import time
np.random.seed(20000)
t0=time()

#Simulating I path with M time steps
S=S0*exp(cumsum((r-0.5*sigma**2)*dt + sigma*math.sqrt(dt)*random.standard_normal((M+1,I)),axis=0))
S[0]=S0

#Calculating the Monte CArlo estimator
C0=math.exp(-r*T)*sum(maximum(S[-1]-K,0))/I

#Results output
tpy=time()-t0
print "European Option Value %7.3f" % C0
print "Duration in Seconds %7.3f"%tpy
#European Option Value   8.166
#Duration in Seconds   0.908

#Plot results from first ten simulations
plt.plot(S[:,:10])
plt.grid(True)
plt.xlabel('time step')
plt.ylabel('index level')

#Plot frequencies of the simulated index levels at the end of the simulation period
plt.hist(S[-1],bins=50)
plt.grid(True)
plt.xlabel('index level')
plt.ylabel('frequency')

#Plot freqencies for option's end-of-period inner values
plt.hist(np.maximum(S[-1]-K,0),bins=50)
plt.grid(True)
plt.xlabel('option inner value')
plt.ylabel('frequency')
plt.ylim(0,50000)

#Calculate how many simulations indicate worthless call options at expiration
sum(S[-1]<K)
#133533