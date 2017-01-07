# -*- coding: utf-8 -*-
"""
Created on Thu Jan 07 12:38:57 2016

@author: nmvenuti
Chapter 10
Stochastics
Python for Finance
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as scs
from time import time
from sys import path
path.append("C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance")
from bsm_functions import bsm_call_value

###################################
#####Generating Random Numbers#####
###################################

#Generate 10 random numbers over the open interval 0-1
npr.rand(10)
#array([ 0.82813023,  0.61078765,  0.28999069,  0.99743849,  0.4192417 ,
#        0.89472269,  0.28824616,  0.31344492,  0.98056146,  0.08913687])

#Generate 5 by 5 array
npr.rand(5,5)

#array([[ 0.88018877,  0.96000449,  0.34081715,  0.9155355 ,  0.8986386 ],
#       [ 0.99936746,  0.11768543,  0.59510899,  0.78592699,  0.72573625],
#       [ 0.82957943,  0.01944854,  0.94681664,  0.1633732 ,  0.02743246],
#       [ 0.97902348,  0.18486651,  0.83445623,  0.01227522,  0.03366613],
#       [ 0.07204403,  0.6369458 ,  0.24576336,  0.42611319,  0.96336514]])

#Create 10 random numbers over the range of 5-10
a=5.
b=10.
npr.rand(10)*(b-a)+a
#array([ 6.02596373,  6.17723021,  7.43374721,  6.39375839,  7.60811951,
#        7.25530294,  6.67820088,  5.05944411,  8.79010565,  9.44972477])

#Also works for multidimensional matricies
npr.rand(5,5)*(b-a)+a
#array([[ 6.81008632,  7.33409289,  5.30328809,  8.69413098,  6.17007685],
#       [ 5.65460548,  5.70057994,  6.00284309,  8.64914273,  7.15979475],
#       [ 8.49266639,  5.7077728 ,  6.11968728,  9.11413766,  5.01906044],
#       [ 6.1321334 ,  9.18780288,  8.39855802,  6.36971474,  9.10708535],
#       [ 5.85267596,  7.71138635,  7.99555873,  9.86878004,  8.36897964]])


#Visualize random draws
sample_size=500
rn1=npr.rand(sample_size,3)
rn2=npr.randint(0,10,sample_size)
rn3=npr.sample(size=sample_size)
a=[0,25,50,75,100]
rn4=npr.choice(a,sample_size)

fig,((ax1,ax2),(ax3,ax4))=plt.subplots(nrows=2,ncols=2,figsize=(7,7))

ax1.hist(rn1,bins=25,stacked=True)
ax1.set_title('rand')
ax1.set_ylabel('frequency')
ax1.grid(True)
ax2.hist(rn2,bins=25)
ax2.set_title('randint')
ax2.grid(True)
ax3.hist(rn3,bins=25)
ax3.set_title('sample')
ax3.set_ylabel('frequency')
ax3.grid(True)
ax4.hist(rn4,bins=25)
ax4.set_title('choice')
ax4.grid(True)

#Visualize random draws from distributions

sample_size=500
rn1=npr.standard_normal(sample_size)
rn2=npr.normal(100,20,sample_size)
rn3=npr.chisquare(df=0.5,size=sample_size)
rn4=npr.poisson(lam=1.0,size=sample_size)

fig,((ax1,ax2),(ax3,ax4))=plt.subplots(nrows=2,ncols=2,figsize=(7,7))

ax1.hist(rn1,bins=25,stacked=True)
ax1.set_title('standard normal')
ax1.set_ylabel('frequency')
ax1.grid(True)
ax2.hist(rn2,bins=25)
ax2.set_title('normal(100,20)')
ax2.grid(True)
ax3.hist(rn3,bins=25)
ax3.set_title('chi square')
ax3.set_ylabel('frequency')
ax3.grid(True)
ax4.hist(rn4,bins=25)
ax4.set_title('Poisson')
ax4.grid(True)

####################
#####Simulation#####
####################

#Example parameters for BSM equation
S0=100 #Inital value
r=0.05 #constant short rate
sigma=0.25 #constant volatility
T=2.0 #time in years
I=10000 #number of iterations

#Estimation of BSM for 10000 iterations at time 1
ST1=S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*npr.standard_normal(I))

#Plot the simulation
plt.hist(ST1,bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.grid(True)

#Data appears to be log-normal
#Attempt to derive the values for the random variables
ST2=S0*npr.lognormal((r-0.5*sigma**2)*T,sigma*np.sqrt(T),size=I)

#Plot the simulation
plt.hist(ST2,bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.grid(True)

#Create function call print_statistics to review simulations

def print_statistics(a1,a2):
    '''Prints Selected Statistics
    
    Parameters
    ==========
    a1,a2: ndarry objects; Results from simulation
    '''
    
    stat1=scs.describe(a1)
    stat2=scs.describe(a2)
    print "%14s %14s %14s" % ('statistics','data set 1', 'data set 2')
    print "%14s %14s %14s" % ('size', stat1[0], stat2[0])
    print "%14s %14s %14s" % ('min', stat1[1][0], stat2[1][0])
    print "%14s %14s %14s" % ('max', stat1[1][1], stat2[1][1])
    print "%14s %14s %14s" % ('mean', stat1[2], stat2[2])
    print "%14s %14s %14s" % ('std', stat1[3], stat2[3])
    print "%14s %14s %14s" % ('skew', stat1[4], stat2[4])
    print "%14s %14s %14s" % ('kurtosis', stat1[5], stat2[5])
    
print_statistics(ST1,ST2)
#    statistics     data set 1     data set 2
#          size          10000          10000
#           min  26.3956531607  26.8534249581
#           max  452.333308129  414.745584093
#          mean  110.521978517  110.265627308
#           std   1666.4714234  1626.80629508
#          skew  1.15376809596  1.16386147348
#      kurtosis  2.60310366236  2.44106559907


##############################
#####Stochastic Processes#####
##############################

#Euler discret Schastic differential equation in BSM
S0=100 #Inital value
r=0.05 #constant short rate
sigma=0.25 #constant volatility
T=2.0 #time in years
I=10000 #number of iterations
M=50
dt=T/M
S=np.zeros((M+1,I))
S[0]=S0
for t in range(1,M+1):
    S[t]=S[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*npr.standard_normal(I))

plt.hist(S[-1],bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.grid(True)

#First 4 moments are very close to thsoe resulting from static simulation
for i in range(4):
    print 'S[%1s]' % -(i+1)
    print_statistics(S[-(i+1)],ST2)

#S[-1]
#    statistics     data set 1     data set 2
#          size          10000          10000
#           min  30.1830538424  26.8534249581
#           max  403.062068357  414.745584093
#          mean  110.068121775  110.265627308
#           std  1627.80992803  1626.80629508
#          skew  1.21483592665  1.16386147348
#      kurtosis  2.69160498213  2.44106559907
#S[-2]
#    statistics     data set 1     data set 2
#          size          10000          10000
#           min  30.5720150594  26.8534249581
#           max  387.552908953  414.745584093
#          mean  109.767076664  110.265627308
#           std  1579.35452654  1626.80629508
#          skew  1.19448506148  1.16386147348
#      kurtosis   2.6038211928  2.44106559907
#S[-3]
#    statistics     data set 1     data set 2
#          size          10000          10000
#           min  29.0018321565  26.8534249581
#           max  382.527112653  414.745584093
#          mean   109.50652701  110.265627308
#           std   1543.7789049  1626.80629508
#          skew   1.1711574295  1.16386147348
#      kurtosis  2.43790466649  2.44106559907
#S[-4]
#    statistics     data set 1     data set 2
#          size          10000          10000
#           min  26.8938282979  26.8534249581
#           max  411.600601223  414.745584093
#          mean   109.25340256  110.265627308
#           std  1509.80815978  1626.80629508
#          skew  1.17121662706  1.16386147348
#      kurtosis  2.54089941538  2.44106559907

#Plot first ten paths taken
plt.plot(S[:,:10],lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.grid(True)

#Square-root diffusion
#Parameters
x0=0.05
kappa=3.
theta=.02
sigma=.1
I=10000
M=50
dt=T/M

def srd_euler():
    xh=np.zeros((M+1,I))
    x1=np.zeros_like(xh)
    xh[0]=x0
    x1[0]=x0
    for t in range(1,M+1):
        xh[t]=(xh[t-1]
        +kappa*(theta-np.maximum(xh[t-1],0))*dt
        +sigma*np.sqrt(np.maximum(xh[t-1],0))*np.sqrt(dt)*npr.standard_normal(I))
    x1=np.maximum(xh,0)
    return x1
x1=srd_euler()

#Plot results of simulation
plt.hist(x1[-1],bins=50)
plt.xlabel('value')
plt.ylabel('frequency')
plt.grid(True)

#First 10 simulated paths illistrate negative average drift due to x0>theta and the convergence to theta
plt.plot(x1[:,:10],lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.grid(True)

#Implementation of exact discretization for square reoot diffusion

def srd_exact():
    x2=np.zeros((M+1,I))
    x2[0]=x0
    for t in range(1,M+1):
        df=4*theta*kappa/sigma**2
        c=(sigma**2*(1-np.exp(-kappa*dt)))/(4*kappa)
        nc=np.exp(-kappa*dt)/c*x2[t-1]
        x2[t]=c*npr.noncentral_chisquare(df,nc,size=I)
    return x2
x2=srd_exact()

#Plot results of simulation
plt.hist(x2[-1],bins=50)
plt.xlabel('value')
plt.ylabel('frequency')
plt.grid(True)

#First 10 simulated paths again illistrating negative average drift due and convergence to theta
plt.plot(x2[:,:10],lw=1.5)
plt.xlabel('time')
plt.ylabel('index level')
plt.grid(True)

#Compare performance of exact vs Eulers methods
print_statistics(x1[-1],x2[-1])
#    statistics     data set 1     data set 2
#          size          10000          10000
#           min 0.00530605817028 0.00475956580422
#           max 0.0524200182337 0.0492069932201
#          mean 0.0200334404663 0.0201001749988
#           std 3.56852491451e-05 3.4859654195e-05
#          skew 0.560764661686 0.613547729908
#      kurtosis 0.445183818816 0.613019620951

#Very similar, only difference is speed (Euler's method is faster as shown below)
I=250000
timer=time()
x1=srd_euler()
print time()-timer
#1.01099991798

timer=time()
x2=srd_exact()
print time()-timer
#1.41499996185

#Stochastic volatility
So=100.
r=0.05
v0=0.1
kappa=3.0
theta=0.25
sigma=0.1
rho=0.6
T=1.0

#Calculate correlation between two stochastic processes 
#Will be done throughthe Cholesky decomp of the correlation matrix
corr_mat=np.zeros((2,2))
corr_mat[0,:]=[1.,rho]
corr_mat[1,:]=[rho,1.]
cho_mat=np.linalg.cholesky(corr_mat)
cho_mat
#array([[ 1. ,  0. ],
#       [ 0.6,  0.8]])

M=50
I=10000
#Set 0 for index process and set 1 for volatility
ran_num=npr.standard_normal((2,M+1,I))

#use euler schme of square-root diffusion process to model the volatility process
dt=T/M
v=np.zeros_like(ran_num[0])
vh=np.zeros_like(v)
v[0]=v0
vh[0]=v0
for t in range(1,M+1):
    ran=np.dot(cho_mat,ran_num[:,t,:])
    vh[t]=(vh[t-1]+kappa*(theta-np.maximum(vh[t-1],0))*dt
        +sigma*np.sqrt(np.maximum(vh[t-1],0))*np.sqrt(dt)*ran[1])
v=np.maximum(vh,0)

#Use the exact Euler scheme w/ correlation to modele geometric Brownian motion
S=np.zeros_like(ran_num[0])
S[0]=S0
for t in range(1,M+1):
    ran=np.dot(cho_mat,ran_num[:,t,:])
    S[t]=S[t-1]*np.exp((r-0.5*v[t])*dt+np.sqrt(v[t])*ran[0]*np.sqrt(dt))

#Plot results in histogram for both index and volatility
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9,5))
ax1.hist(S[-1],bins=50)
ax1.set_xlabel('index level')
ax1.set_ylabel('frequency')
ax1.grid(True)
ax2.hist(v[-1],bins=50)
ax2.set_xlabel('volatility')
ax2.grid(True)

#Inspect fist ten paths of each simulated process
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(7,6))
ax1.plot(S[:,:10],lw=1.5)
ax1.set_ylabel('index level')
ax1.set_xlabel('time')
ax1.grid(True)
ax2.plot(v[:,:10],lw=1.5)
ax2.set_ylabel('volatility')
ax2.grid(True)

#Review summary statistics
print_statistics(S[-1],v[-1])
#    statistics     data set 1     data set 2
#          size          10000          10000
#           min  16.2659886268 0.174203691543
#           max  941.002856257 0.322917171213
#          mean  107.878591946 0.243185909791
#           std  2716.80099099 0.000407590707965
#          skew  1.89871237541 0.164998532689
#      kurtosis  10.5697352798 0.0369036987401


#Jump diffusion
S0=100.
r=0.05
sigma=0.2
lamb=0.75
mu=-0.6
delta=0.25
T=1.

#Generate three sets of random independent numbers
#Calculate resulting index at end of time period
M=50
I=10000
dt=T/M
rj=lamb*(np.exp(mu+0.5*delta**2)-1)
S=np.zeros((M+1,I))
S[0]=S0
sn1=npr.standard_normal((M+1,I))
sn2=npr.standard_normal((M+1,I))
poi=npr.poisson(lamb*dt,(M+1,I))
for t in range(1,M+1):
    S[t]=S[t-1]*np.exp((r-rj-0.5*sigma**2)*dt
        +sigma*np.sqrt(dt)*sn1[t]
        +(np.exp(mu+delta*sn2[t])-1)
        *poi[t])
    S[t]=np.maximum(S[t],0)

#Plot histogram of index
plt.hist(S[-1],bins=50)
plt.xlabel('index level')
plt.ylabel('frequency')
plt.grid(True)

#Plot first ten paths
plt.plot(S[:,:10],lw=1.5)
plt.ylabel('index level')
plt.xlabel('time')
plt.grid(True)

#Variance Reduction

#Compare statistics from different modes of random number generation
print "%15s %15s" % ('mean','std')
print 31 *"-"
for i in range(1,31,2):
    npr.seed(1000)
    sn=npr.standard_normal(i**2*10000)
    print "%15.12f %15.12f" % (sn.mean(),sn.std())

#           mean             std
#-------------------------------
#-0.0118703945584   1.00875243072
#-0.00281566729799   1.00272953635
#-0.00384777670358   1.00059404416
#-0.00305811337431   1.00108634533
#-0.00168512653818   1.00163084959
#-0.0011752120073   1.00134768464
#-0.000803969035726   1.00015908143
#-0.000601970954254  0.999506522127
#-0.000147787693343  0.999571756099
#-0.000313035580558  0.999646153704
#-0.000178447060942  0.999677277878
#9.65017093566e-05  0.999684346792
#-0.000135677013257  0.999823841902
#-1.57269856488e-05  0.999906493379
#-3.93685191444e-05   1.00006309195

#Random numbers somehow get better the larger the number of draws become
#Though still not perfect at max (30^2*10000)

#Using anithetic variates to reduce variance in numpy
print "%15s %15s" % ('mean','std')
print 31 *"-"
for i in range(1,31,2):
    npr.seed(1000)
    sn=npr.standard_normal(i**2*10000/2)
    sn=np.concatenate((sn,-sn))
    print "%15.12f %15.12f" % (sn.mean(),sn.std())

#           mean             std
#-------------------------------
# 0.000000000000  1.009653753942
#-0.000000000000  1.000413716783
# 0.000000000000  1.002925061201
#-0.000000000000  1.000755212673
# 0.000000000000  1.001636910076
#-0.000000000000  1.000726758438
#-0.000000000000  1.001621265149
# 0.000000000000  1.001203722778
#-0.000000000000  1.000556669784
# 0.000000000000  1.000113464185
#-0.000000000000  0.999435175324
# 0.000000000000  0.999356961431
#-0.000000000000  0.999641436845
#-0.000000000000  0.999642768905
#-0.000000000000  0.999638303451

#Fixes first moment(mean), but not second(std)

#Moment matching addresses both the 1st and second
sn=npr.standard_normal(10000)
sn.mean()
#-0.001165998295162494
sn.std()
#0.99125592020460496
sn_new=(sn-sn.mean())/sn.std()

sn_new.mean()
#-2.3803181647963357e-17
sn_new.std()
#0.99999999999999989

#Function to account of these issues
def gen_sn(M,I,anti_paths=True,mo_match=True):
    '''Function to generate random numbers for simulation.
    
    Parameters
    ==========
    M: int
        number of time intervals for discretization
    I: int
        number of paths to be simulated
    anti_paths: Boolean
        use of antithetic variates
    mo_match: Boolean
        use of moment matching
    '''
    if anti_paths is True:
        sn=npr.standard_normal((M+1,I/2))
        sn=np.concatenate((sn,-sn),axis=1)
    else:
        sn=npr.standard_normal((M+1,I))
    if mo_match is True:
        sn = (sn-sn.mean())/sn.std()
    return sn

###################
#####Valuation#####
###################

#European options

#Simulating only the index level at maturity
S0=100.
r=0.05
sigma=0.25
T=1.
I=50000
def gbm_mcs_stat(K):
    '''Valuation of European call option in Black-Scholes-Merton by Monte Carlo
    Simulation(of index level at maturity)
    
    Parameters
    ==========
    K: Float, postive strike price of the option
    
    Returns:
    =======
    C0: float, esmtimated present value of Europen call option
    '''
    sn=gen_sn(1,I)
    
    #Simulate index level at maturity
    for t in range(1,M+1):
        ST=S0*np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*sn[1])
    
    #Calaculate payoff at maturity
    hT=np.maximum(ST-K,0)
    
    #Calculate MCS estimator
    C0=np.exp(-r*T)*1/I*np.sum(hT)
    return C0

gbm_mcs_stat(K=105.)
#10.044221852841922

#Dynamic simulation approach w/ put and call options
def gbm_mcs_dyna(K,option='call',M=50):
    '''Valuation of European options in BSM by MCS(of index level paths)
        Parameters
    ==========
    K: Float, postive strike price of the option
    M: Int, number of intervals of discretization
    option: string, Type of the option to be valued ('call','put')
    Returns:
    =======
    C0: float, esmtimated present value of Europen call option
    '''
    
    dt=T/M
    #Simulation of index level paths
    S=np.zeros((M+1,I))
    S[0]=S0
    sn=gen_sn(M,I)
    for t in range(1,M+1):
        S[t]=S[t-1]*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*sn[t])
    #case-based calculation of payoff
    if option =='call':
        hT=np.maximum(S[-1]-K,0)
    else:
        hT=np.maximum(K-S[-1],0)
    #Calculation of MCS estimator
    C0=np.exp(-r*T)*1/I*np.sum(hT)
    return C0

#Check values
gbm_mcs_dyna(K=110.,option='call')
#7.7744903303620134
gbm_mcs_dyna(K=110.,option='put')
#13.306783385663836

#compare simulation methods generated above with benchmark value from BSM function
k_min=80.
k_max=120.1
step=5.
k_list=np.arange(k_min,k_max+1,step)
results=np.zeros(((int((k_max-k_min)/step)+1),3))
np.random.seed(200000)
for i in range(len(k_list)):
    k=k_list[i]
    results[i,0]=gbm_mcs_stat(k) #Static
    results[i,1]=gbm_mcs_dyna(k) #Dynamic
    results[i,2]=bsm_call_value(S0,k,T,r,sigma) #Analytic

#Plot results of static versus analytic
fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(8,6))
ax1.plot(k_list,results[:,2],'b',label='analytical')
ax1.plot(k_list,results[:,0],'ro',label='static')
ax1.set_ylabel('European Call Option Value')
ax1.grid(True)
ax1.legend(loc=0)
ax1.set_ylim(ymin=0)
wi =1.
ax2.bar(k_list-wi/2,(results[:,2]-results[:,0])/results[:,2]*100,wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('difference in %')
ax2.set_xlim(left=75,right=125)
ax2.grid(True)

#Plot results of dynamic versus analytic
fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(8,6))
ax1.plot(k_list,results[:,2],'b',label='analytical')
ax1.plot(k_list,results[:,1],'ro',label='dynamic')
ax1.set_ylabel('European Call Option Value')
ax1.grid(True)
ax1.legend(loc=0)
ax1.set_ylim(ymin=0)
wi =1.
ax2.bar(k_list-wi/2,(results[:,2]-results[:,1])/results[:,2]*100,wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('difference in %')
ax2.set_xlim(left=75,right=125)
ax2.grid(True)


#American Options

#LSM algo for American call and put options
def gbm_mcs_amer(K,option='call'):
    '''Valuation of AMerican option in BSM by MCS by LSM algo
    
    Parameters
    ==========
    K: positive strike price of the option (float)
    option: type of the option to be valued ('call','float')(string)
    
    Returns
    =======
    C0: estimated present value of European call option
    '''
    dt=T/M
    df=np.exp(-r*dt)
    S=np.zeros((M+1,I))
    S[0]=S0
    sn=gen_sn(M,I)
    for t in range(1,M+1):
        S[t]=S[t-1]*np.exp((r-0.5*sigma**2)*dt
            + sigma*np.sqrt(dt)*sn[t])
    #Case-based calculation payoff
    if option=='call':
        h=np.maximum(S-K,0)
    else:
        h=np.maximum(K-S,0)
    
    #LSM Algo
    V=np.copy(h)
    for t in range(M-1,0,-1):
        reg=np.polyfit(S[t],V[t+1]*df,7)
        C=np.polyval(reg,S[t])
        V[t]=np.where(C>h[t],V[t+1]*df,h[t])
    
    #MCS estimator
    C0=df*1/I*np.sum(V[1])
    return C0

#Check values
gbm_mcs_amer(110.,option='call')
#7.7789332794493156
gbm_mcs_amer(110.,option='put')
#13.614023206242445

#The european value of an option represents a lower bound to the American
#options value (difference is  known as early exercise premium)

#compare values of european and american options
k_min=80.
k_max=120.1
step=5.
k_list=np.arange(k_min,k_max+1,step)
results=np.zeros(((int((k_max-k_min)/step)+1),2))
np.random.seed(200000)
for i in range(len(k_list)):
    k=k_list[i]
    results[i,0]=gbm_mcs_dyna(k,option='put') #European
    results[i,1]=gbm_mcs_amer(k,option='put') #American


#Plot results of static versus analytic
fig,(ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(8,6))
ax1.plot(k_list,results[:,0],'b',label='European')
ax1.plot(k_list,results[:,1],'ro',label='American')
ax1.set_ylabel('call option value')
ax1.grid(True)
ax1.legend(loc=0)
ax1.set_ylim(ymin=0)
wi =1.
ax2.bar(k_list-wi/2,(results[:,1]-results[:,0])/results[:,0]*100,wi)
ax2.set_xlabel('strike')
ax2.set_ylabel('early exercise premium in %')
ax2.set_xlim(left=75,right=125)
ax2.grid(True)

#######################
#####Risk Measures#####
#######################

#Value at risk (VAR)

#BSM example
S0=100
r=0.05
sigma=0.25
T=30/365.
I=10000
ST=S0*np.exp((r-0.5*sigma**2)*T
    +sigma*np.sqrt(T)*npr.standard_normal(I))

#Simulate absolute P/L
R_gbm=np.sort(ST-S0)

#Plot performance
plt.hist(R_gbm,bins=50)
plt.xlabel('absolute return')
plt.ylabel('frequency')
plt.grid(True)

#Get different VAR percentiles
percs=[0.01,.1,1.,2.5,5.,10.]
var=scs.scoreatpercentile(R_gbm,percs)
print "%16s %16s" % ('Confidence Level','Value-at-Risk')
print 33*'-'
for pair in zip(percs,var):
    print '%16.2f %16.2f ' % (100-pair[0],-pair[1])

#Confidence Level    Value-at-Risk
#---------------------------------
#           99.99            21.96 
#           99.90            20.59 
#           99.00            15.14 
#           97.50            13.02 
#           95.00            10.95 
#           90.00             8.51 

#Jump example
dt=30./365/M
rj=lamb*(np.exp(mu+0.5*delta**2)-1)
S=np.zeros((M+1,I))
S[0]=S0
sn1=npr.standard_normal((M+1,I))
sn2=npr.standard_normal((M+1,I))
poi=npr.poisson(lamb*dt,(M+1,I))
for t in range(1,M+1):
    S[t]=S[t-1]*(np.exp((r-rj-0.5*sigma**2)*dt
                +sigma*np.sqrt(dt)*sn1[t])
                +(np.exp(mu+delta*sn2[t])-1)
                *poi[t])
    S[t]=np.maximum(S[t],0)
R_jd=np.sort(S[-1]-S0)

#Plot peformance
plt.hist(R_jd,bins=50)
plt.xlabel('absolute return')
plt.ylabel('frequency')
plt.grid(True)

#Get different VAR percentiles
percs=[0.01,.1,1.,2.5,5.,10.]
var=scs.scoreatpercentile(R_jd,percs)
print "%16s %16s" % ('Confidence Level','Value-at-Risk')
print 33*'-'
for pair in zip(percs,var):
    print '%16.2f %16.2f ' % (100-pair[0],-pair[1])

#Confidence Level    Value-at-Risk
#---------------------------------
#           99.99            79.57 
#           99.90            73.39 
#           99.00            58.55 
#           97.50            47.96 
#           95.00            26.72 
#           90.00             8.87 

#Compare both VAR measures directly
percs=list(np.arange(0.,10.1,0.1))
gbm_var=scs.scoreatpercentile(R_gbm,percs)
jd_var=scs.scoreatpercentile(R_jd,percs)

plt.plot(percs,gbm_var,'b',lw=1.5,label='GBM')
plt.plot(percs,jd_var,'r',lw=1.5,label='JD')
plt.legend(loc=4)
plt.grid(True)
plt.xlabel('100-CI[%]')
plt.ylabel('VAR')
plt.ylim(ymax=0)

#Credit Value Adjustments
S0=100.
r=0.05
sigma=0.2
T=1.
I=100000
ST=S0*np.exp((r - 0.5 * sigma ** 2) * T
        +sigma * np.sqrt(T) * npr.standard_normal(I))

#Set fixed average loss level L and fixed probability of default p
L=0.5
p=0.01

#Use Poisson distribution taking into account that default can only occur once
D=npr.poisson(p * T, I)
D=np.where(D > 1, 1 ,D)

CVaR=np.exp(-r*T)*1/I*np.sum(L*D*ST)
CVaR
#0.53967797315032717

#The PV of the asset, adjusted for the credit risk is as follows (discrepencies due to numerical errors)
S0_CVA=np.exp(-r*T)*1/I*np.sum((1-L*D)*ST)
S0_adj=S0-CVaR

print S0_CVA #99.5399416434
print S0_adj #99.4186268737

#Number of defaults
np.count_nonzero(L*D*ST)
#988

plt.hist(L*D*ST,bins=50)
plt.xlabel('loss')
plt.ylabel('frequency')
plt.grid(True)
plt.ylim(ymax=175)


