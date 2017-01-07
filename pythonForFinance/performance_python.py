# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 14:38:29 2016

@author: nmvenuti
Performance Python
Python for Finance
"""

#Define conveinence function to sytematically compare different performance packages

def perf_comp_data(func_list,data_list, rep=3, number =1):
    '''Function to compare the performance of different functions
    
    Parameters
    ==========
    func_list: list with function names stored as strings
    data_list: list with data set names stored as strings
    rep: int, number of repitions of the whole comparison
    number: int, numer of executions of each function
    '''
    from timeit import repeat
    res_list={}
    for name in enumerate(func_list):
        stmt = name[1] + '('+data_list[name[0]]+')'
        setup = "from __main__ import "+name[1] + ', ' + data_list[name[0]]
        results = repeat(stmt=stmt,setup=setup,repeat=rep,number=number)
        res_list[name[1]]=sum(results)/rep
        res_sort = sorted(res_list.iteritems(),key=lambda(k,v): (v,k))
        for item in res_sort:
            rel=item[1]/res_sort[0][1]
            print 'function: ' + item[0] + ', av. time sec: %9.5f, ' % item[1] + 'relative: %6.1f' % rel


from math import *
import numpy as np
import numexpr as ne
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
import numba as nb

def f(x):
    return abs(cos(x))**0.5+sin(2+3*x)

#example of performance difference
I=500000
a_py=range(I)
a_np=np.arange(I)

#Standard python with explicit looping
def f1(a):
    res=[]
    for x in a:
        res.append(f(x))
    return res

#iterator approach with implicit looping
def f2(a):
    return [f(x) for x in a]

#Iterator approach with implicit looping and using eval
def f3(a):
    ex = 'abs(cos(x)) **0.5 + sin(2+3*x)'
    return [eval(ex) for x in a]

#Numpy Vectorization implementation
def f4(a):
    return(np.abs(np.cos(a))**0.5 + np.sin(2+3*a))

#Single-threaded implementation using numexpr
def f5(a):
    ex = 'abs(cos(a)) **0.5 + sin(2+3*a)'
    ne.set_num_threads(1)
    return ne.evaluate(ex)

#Multi-threaded implementation using numexpr
def f6(a):
    ex = 'abs(cos(a)) **0.5 + sin(2+3*a)'
    ne.set_num_threads(16)
    return ne.evaluate(ex)



timer=time.time()
r1=f1(a_py)
r2=f2(a_py)
r3=f3(a_py)
r4=f4(a_np)
r5=f5(a_np)
r6=f6(a_np)

print time.time()-timer
#9.15799999237 seconds

#Verify all functions produced correct results
print np.allclose(r1,r2)
print np.allclose(r1,r3)
print np.allclose(r1,r4)
print np.allclose(r1,r5)
print np.allclose(r1,r6)

#Check time differences between each implementation
func_list=['f1','f2','f3','f4','f5','f6']
data_list=['a_py','a_py','a_py','a_np','a_np','a_np']

perf_comp_data(func_list,data_list)

#function: f6, av. time sec:   0.02028, relative:    1.0
#function: f4, av. time sec:   0.04144, relative:    2.0
#function: f5, av. time sec:   0.04434, relative:    2.2
#function: f2, av. time sec:   0.31310, relative:   15.4
#function: f1, av. time sec:   0.34285, relative:   16.9
#function: f3, av. time sec:   8.29713, relative:  409.1

#Paralleization

def bsm_mcs_valuation(strike):
    '''Dynamic Black-Scholes-Merton Monte Carlo estimator for European calls.
    
    Parameters:
    ==========
    
    strike: float, strike price of the option
    
    Results:
    ========
    value: float, estimate for the present value of call option
    '''
    
    import numpy as np
    S0=100.
    T=1.0
    r = 0.05
    vola=0.2
    M=50
    I=20000
    dt=T/M
    rand = np.random.standard_normal((M+1,I))
    S=np.zeros((M+1,I))
    S[0]=S0
    for t in range(1,M+1):
        S[t] = S[t-1]*np.exp((r-0.5*vola**2)*dt+vola*np.sqrt(dt)*rand[t])
    value=(np.exp(-r*T)*np.sum(np.maximum(S[-1]-strike,0))/I)
    return value

def seq_value(n):
    '''Sequential option valuation.
    
    Parameters
    ==========
    
    n: int, number of options valuations/strikes
    '''
    import numpy as np
    strikes=np.linspace(80,120,n)
    option_values = []
    for strike in strikes:
        option_values.append(bsm_mcs_valuation(strike))
    return strikes, option_values

n=100

timer=time.time()
strikes, option_values = seq_value(n)
print time.time()-timer
#4.88499999046 seconds

#Plot results
plt.figure(figsize=(8,5))
plt.plot(strikes,option_values, 'b')
plt.plot(strikes,option_values, 'r.')
plt.grid(True)
plt.xlabel('strikes')
plt.ylabel('European call option values')


#IPython.Paraellel not working 

def simulate_brownian_motion(p):
    import math
    #time steps and paths
    M,I=p
    S0=100
    r=0.05
    sigma=0.2
    T=1.0
    dt=T/M
    paths=np.zeros((M+1,I))
    paths[0]=S0
    for t in range(1,M+1):
        paths[t]=paths[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*math.sqrt(dt)*np.random.standard_normal(I))
    return paths

#Test function
paths = simulate_brownian_motion((5,2))
paths

#Not working keeps getting hung up. Do more research on MP package
#I=5
#M=2
#t=1
#w=1
#times=[]
#t0=time.time()
#pool=mp.Pool(processes=w)
#result = pool.map(simulate_brownian_motion,t*[(M,I),])
#times.append(time.time()-t0)
#pool.close()
#pool.join()
#
##Run simulation on laptop with 2 cores/ 4 threads
#I=10000
#M=100
#t=1
#times=[]
#for w in range(1,5):
#    t0=time.time()
#    pool=mp.Pool(processes=w)
#    result = pool.map(simulate_brownian_motion,t*[(M,I),])
#    times.appeand(time.time()-t0)


###########################
#####Dynamic compiling#####
###########################

def f_py(I,J):
    res=0
    for i in range(I):
        for j in range(J):
            res +=int(cos(log(1)))
    return res

I,J=5000,5000
#Normal python
timer=time.time()
f_py(I,J)
print time.time()-timer

#8.77200007439 seconds

#Numpy vectorization
def f_np(I,J):
    a=np.ones((I,J),dtype=np.float64)
    return int(np.sum(np.cos(np.log(a)))),a

timer=time.time()
a=f_np(I,J)
print time.time()-timer
#0.531999826431 seconds

#Check number of bytes made
a[1].nbytes
#200000000 bytes (200MB)

#Can use Numba to reduce this issue
#Just add 'jit' to pythonic version of function
f_nb=nb.jit(f_py)

#Numba
timer=time.time()
f_nb(I,J)
print time.time()-timer
#0.363000154495 seconds

#Systematic comparison
func_list=['f_py','f_np','f_nb']
data_list=3*['I,J']
perf_comp_data(func_list,data_list)

#function: f_nb, av. time sec:   0.00000, relative:    1.0
#function: f_np, av. time sec:   0.59245, relative: 159847.9
#function: f_py, av. time sec:   8.65098, relative: 2334101.3

#Binomial option pricing

#First through looping
def binomial_py(strike):
    '''Binomial option pricing via looping.
    
    Parameters
    ==========
    strike price of the European call option
    '''
    #Parameters definitions
    #Model and option parameters
    S0=100 #Inital index value
    T=1. #call option maturity
    r=0.05 #constant short rate
    vola=0.20 #constant volatitity factor of diffusion
    
    #time parameters
    M=1000 #time steps
    dt=T/M #length of time interval
    df=exp(-r*dt) #discount factor per time interval
    
    #Binomial parameters
    u=exp(vola*sqrt(dt)) #up-movement
    d=1/u #down-movement
    q=(exp(r*dt)-d)/(u-d) #martingale probability
    
    #Loop 1- index values
    S=np.zeros((M+1,M+1),dtype=np.float64)
    S[0,0]=S0
    z1=0
    for j in xrange(1,M+1,1):
        z1=z1+1
        for i in xrange(z1+1):
            S[i,j]=S[0,0]*(u**j)*(d**(i*2))
    
    #Loop 2- Inner values
    iv=np.zeros((M+1,M+1))
    z2=0
    for j in xrange(0,M+1,1):
        for i in xrange(z2+1):
            iv[i,j]=max(S[i,j]-strike,0)
        z2=z2+1
    
    #Loop 3 Valuation
    pv=np.zeros((M+1,M+1),dtype=np.float64)
    pv[:,M]=iv[:,M]
    z3=M+1
    for j in xrange(M-1,-1,-1):
        z3=z3-1
        for i in xrange(z3):
            pv[i,j]=(q*pv[i,j+1]+(1-q)*pv[i+1,j+1])*df
    return pv[0,0]

timer=time.time()
print round(binomial_py(100),3)
#10.449
print time.time()-timer
#1.25199985504 seconds

#Compare using value from Monte Carlo based BSM
timer=time.time()
print round(bsm_mcs_valuation(100),3)
#10.342
print time.time()-timer
#0.0570001602173 seconds

#Numpy vectorization of binomal
def binomial_np(strike):
    import numpy as np
    '''Binomial option pricing via looping.
    
    Parameters
    ==========
    strike price of the European call option
    '''
    #Parameters definitions
    #Model and option parameters
    S0=100 #Inital index value
    T=1. #call option maturity
    r=0.05 #constant short rate
    vola=0.20 #constant volatitity factor of diffusion
    
    #time parameters
    M=1000 #time steps
    dt=T/M #length of time interval
    df=exp(-r*dt) #discount factor per time interval
    
    #Binomial parameters
    u=exp(vola*sqrt(dt)) #up-movement
    d=1/u #down-movement
    q=(exp(r*dt)-d)/(u-d) #martingale probability
    
    #Index levels with Numpy
    mu=np.arange(M+1)
    mu=np.resize(mu,(M+1,M+1))
    md=np.transpose(mu)
    mu=u**(mu-md)
    md=d**md
    S=S0*mu*md
    
    #Valuation loop
    pv=np.maximum(S-strike,0)
    z=0
    for t in range(M-1,-1,-1): #backwards iteration
        pv[0:M-z,t]=(q*pv[0:M-z,t+1]+(1-q)*pv[1:M-z+1,t+1])*df
        z+=1
    return pv[0,0]

#Also create a numba version of first operation
binomial_nb=nb.jit(binomial_py)
binomial_nb(100)

#Do a comparison
func_list=['binomial_py','binomial_np','binomial_nb']
K=100
data_list=3*['K']
perf_comp_data(func_list,data_list)

#function: binomial_nb, av. time sec:   0.03793, relative:    1.0
#function: binomial_np, av. time sec:   0.21697, relative:    5.7
#function: binomial_py, av. time sec:   1.24832, relative:   32.9

#################
#####CPython#####
#################

import pyximport
pyximport.install()
import sys
sys.path.append('C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance/')
from nested_loop import f_cy
