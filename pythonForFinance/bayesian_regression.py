# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 08:28:44 2016

@author: nmvenuti
Bayesian Regression
Python for Finance
"""
import warnings
warnings.simplefilter('ignore')
import pymc3
import numpy as np
np.random.seed(1000)
import matplotlib.pyplot as plt

#Introductory example
x=np.linspace(0,10,500)
y=4+2*x+np.random.standard_normal(len(x))*2

#Fit ols regression as benchmark
reg=np.polyfit(x,y,1)

#Plot
plt.figure(figsize=(8,4))
plt.scatter(x,y,c=y,marker='v')
plt.plot(x,reg[1]+reg[0]*x,lw=2.0)
plt.colorbar()
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')


#Review coefficents for linear regression
reg
#array([ 2.03384161,  3.77649234])

#Markov Chain Monte Carlo Sampling
with pmymc3.Model() as model:
    #Define priors
    alpha=pymc3.Normal('alpha',mu=0,sd=20)
    beta=pymc3.Normal('beta',mu=0,sd=20)
    sigma=pymc3.Uniform('sigma',lower=0,upper=10)
    
    #Define linear regression
    y_est=alpha+beta*x
    
    #Defineliklihood
    likelihood=pymc3.Normal('y',mu=y_est,sd=sigma,observed=y)
    
    #Inference
    start=pymc3.find_MAP()#Find starting value by optimization
    step=pymc3.NUTS(start=start)#Instantiatie MCMC sampling algo
    trace=pymc3.sample(100,step,start=start,progressbar=False)#draw 100 posterior samples using NUTS sampling

trace[0]
    