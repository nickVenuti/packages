# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:58:30 2016

@author: nmvenuti
Chapter 11
Statistics
Python for Finance
"""

import numpy as np
import scipy.stats as scs
import scipy.optimize as sco
import scipy.interpolate as sci
import statsmodels.api as sm
from sklearn.decomposition import KernelPCA
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import pandas.io.data as web

np.random.seed(1000)

#Set up function for geometric brownian motion

def gen_paths(S0,r,sigma,T,M,I):
    '''Generates Monte Carlo paths for geometric Brownian motion
    
    Paramters
    =========
    
    S0:float initial stock/index value
    r:float constant short rate
    sigma: float constant volatility
    T: float final time horizon
    M: into nber of time steps/intervals
    I: int number of paths to be simulated
    
    Returns
    =======
    paths: ndarray, shape (M+1,I) simulated paths fiven the parameters
    '''
    dt=float(T)/M
    paths=np.zeros((M+1,I),np.float64)
    paths[0]=S0
    for t in range(1,M+1):
        rand=np.random.standard_normal(I)
        rand = (rand-rand.mean())/rand.std()
        paths[t]=paths[t-1]*np.exp((r-0.5*sigma**2)*dt
                +sigma*np.sqrt(dt)*rand)
    return paths

S0=100.
r=0.05
sigma=0.2
T=1.0
M=50
I=250000

paths=gen_paths(S0,r,sigma,T,M,I)

#plot output for first ten paths
plt.plot(paths[:,:10])
plt.grid(True)
plt.xlabel('time steps')
plt.ylabel('index level')


#interested in the logistic daily returns
log_returns=np.log(paths[1:]/paths[0:-1])

#Create print statistics funcion
def print_statistics(array):
    '''Prints selected statistics.
    
    Parameters
    ==========
    
    array:darray object to generates statistics on
    '''
    sta=scs.describe(array)
    print "%14s %15s" % ('statistic','value')
    print 30*"-"
    print"%14s %15.5f" % ('size',sta[0])
    print"%14s %15.5f" % ('min',sta[1][0])
    print"%14s %15.5f" % ('max',sta[1][1])
    print"%14s %15.5f" % ('mean',sta[2])
    print"%14s %15.5f" % ('std',np.sqrt(sta[3]))
    print"%14s %15.5f" % ('skew',sta[4])
    print"%14s %15.5f" % ('kurtosis',sta[5])

#example on log returns
print_statistics(log_returns.flatten())
#     statistic           value
#------------------------------
#          size  12500000.00000
#           min        -0.15664
#           max         0.15371
#          mean         0.00060
#           std         0.02828
#          skew         0.00055
#      kurtosis         0.00085

#Compare distribution with normal
plt.hist(log_returns.flatten(),bins=70,normed=True,label='frequency')
plt.grid(True)
plt.xlabel('log-return')
plt.ylabel('freq')
x=np.linspace(plt.axis()[0],plt.axis()[1])
plt.plot(x,scs.norm.pdf(x,loc=r/M,scale=sigma/np.sqrt(M)),'r',lw=2.0,label='pdf')
plt.legend()

#Create quantile-quantile(qwq-plots)
sm.qqplot(log_returns.flatten()[::500],line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')

#Create normality function
def normality_test(arr):
    '''Tests for normality distribution of givven data set.
    
    Parameters
    ==========
    array: ndarray object to generates statistics on
    '''
    
    print 'Skew of data set %14.3f' %scs.skew(arr)
    print 'Skew test p value %14.3f' %scs.skewtest(arr)[1]
    print 'Kurt of data set %14.3f' %scs.kurtosis(arr)
    print 'Kurt test p value %14.3f' %scs.kurtosistest(arr)[1]
    print 'Normal test p value %14.3f' %scs.normaltest(arr)[1]

#Test
normality_test(log_returns.flatten())

#Skew of data set          0.001
#Skew test p value          0.430
#Kurt of data set          0.001
#Kurt test p value          0.541
#Normal test p value          0.607

#Check of end of period data is log-normal
f, (ax1,ax2)=plt.subplots(1,2,figsize=(9,4))
ax1.hist(paths[-1],bins=30)
ax1.grid(True)
ax1.set_xlabel('index level')
ax1.set_ylabel('freq')
ax1.set_title('regular data')
ax2.hist(np.log(paths[-1]),bins=30)
ax2.grid(True)
ax2.set_xlabel('log index level')
ax2.set_title('log data')

#Review the statistics
print_statistics(paths[-1])

#     statistic           value
#------------------------------
#          size    250000.00000
#           min        42.74870
#           max       233.58435
#          mean       105.12645
#           std        21.23174
#          skew         0.61116
#      kurtosis         0.65182


print_statistics(np.log(paths[-1]))

#     statistic           value
#------------------------------
#          size    250000.00000
#           min         3.75534
#           max         5.45354
#          mean         4.63517
#           std         0.19998
#          skew        -0.00092
#      kurtosis        -0.00327

#Shows high p values (strong support for the normal distribution)
normality_test(np.log(paths[-1]))

#Skew of data set         -0.001
#Skew test p value          0.851
#Kurt of data set         -0.003
#Kurt test p value          0.744
#Normal test p value          0.931

#Compare data to normal pdf
log_data=np.log(paths[-1])
plt.hist(log_data,bins=70,normed=True,label='observed')
plt.grid(True)
plt.xlabel('index levels')
plt.ylabel('freq')
x=np.linspace(plt.axis()[0],plt.axis()[1])
plt.plot(x,scs.norm.pdf(x,log_data.mean(),log_data.std()),'r',lw=2.0,label='pdf')
plt.legend()

#Show qq=plot
sm.qqplot(log_data,line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')


#Real word comparison
symbols=['^GDAXI','^GSPC','YHOO','MSFT']
data =pd.DataFrame()
for sym in symbols:
    data[sym]=web.DataReader(sym,data_source='yahoo',start='1/1/2006')['Adj Close']
data=data.dropna()
data.info()

#DatetimeIndex: 2494 entries, 2006-01-03 to 2016-01-08
#Data columns (total 4 columns):
#^GDAXI    2494 non-null float64
#^GSPC     2494 non-null float64
#YHOO      2494 non-null float64
#MSFT      2494 non-null float64
#dtypes: float64(4)
#memory usage: 97.4 KB

#Check head of data
data.head()
#                 ^GDAXI        ^GSPC       YHOO       MSFT
#Date                                                      
#2006-01-03  5460.680176  1268.800049  40.910000  21.373719
#2006-01-04  5523.620117  1273.459961  40.970001  21.477242
#2006-01-05  5516.529785  1273.479980  41.529999  21.493169
#2006-01-06  5536.319824  1285.449951  43.209999  21.429462
#2006-01-09  5537.109863  1290.150024  43.419998  21.389646

#Normalize data to starting value of 100 and plot
(data/data.ix[0]*100).plot(figsize=(8,6))

#Calculate log returns
log_returns=np.log(data/data.shift(1))
log_returns.head()

#Plot histograms of log-returns
log_returns.hist(bins=50,figsize=(9,6))

#Check statisitical tests
for sym in symbols:
   print '\nResults for symbol %s ' % sym
   print 30*'-'
   log_data=np.array(log_returns[sym].dropna())
   print_statistics(log_data)

#Results for symbol ^GDAXI 
#------------------------------
#     statistic           value
#------------------------------
#          size      2493.00000
#           min        -0.07739
#           max         0.10797
#          mean         0.00024
#           std         0.01465
#          skew         0.00069
#      kurtosis         5.69289
#
#Results for symbol ^GSPC 
#------------------------------
#     statistic           value
#------------------------------
#          size      2493.00000
#           min        -0.09470
#           max         0.10957
#          mean         0.00017
#           std         0.01319
#          skew        -0.31461
#      kurtosis        10.08762
#
#Results for symbol YHOO 
#------------------------------
#     statistic           value
#------------------------------
#          size      2493.00000
#           min        -0.24636
#           max         0.39182
#          mean        -0.00012
#           std         0.02548
#          skew         0.52206
#      kurtosis        31.37285
#
#Results for symbol MSFT 
#------------------------------
#     statistic           value
#------------------------------
#          size      2493.00000
#           min        -0.12458
#           max         0.17063
#          mean         0.00036
#           std         0.01782
#          skew         0.08752
#      kurtosis        10.08800

#Kurtosis test seems far from normal for each data set

#Check qq-plots for GSPC and MSFT
sm.qqplot(log_returns['^GSPC'].dropna(),line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')

sm.qqplot(log_returns['MSFT'].dropna(),line='s')
plt.grid(True)
plt.xlabel('theoretical quantiles')
plt.ylabel('sample quantiles')

#Both show fat tails

#Run normality tests
for sym in symbols:
   print '\nResults for symbol %s ' % sym
   print 30*'-'
   log_data=np.array(log_returns[sym].dropna())
   normality_test(log_data)

#Results for symbol ^GDAXI 
#------------------------------
#Skew of data set          0.001
#Skew test p value          0.989
#Kurt of data set          5.693
#Kurt test p value          0.000
#Normal test p value          0.000
#
#Results for symbol ^GSPC 
#------------------------------
#Skew of data set         -0.315
#Skew test p value          0.000
#Kurt of data set         10.088
#Kurt test p value          0.000
#Normal test p value          0.000
#
#Results for symbol YHOO 
#------------------------------
#Skew of data set          0.522
#Skew test p value          0.000
#Kurt of data set         31.373
#Kurt test p value          0.000
#Normal test p value          0.000
#
#Results for symbol MSFT 
#------------------------------
#Skew of data set          0.088
#Skew test p value          0.074
#Kurt of data set         10.088
#Kurt test p value          0.000
#Normal test p value          0.000

#Strongly rejects hypothesis that data is normally distributed

################################
#####Portfolio Optimization#####
################################

#Select target assests
symbols=['AAPL','MSFT','YHOO','DB','GLD']
noa=len(symbols)

#Get adjusted closing price of stocks until 2014-09-12

data=pd.DataFrame()

for sym in symbols:
        data[sym] = web.DataReader(sym,data_source='yahoo',end='2014-09-12')['Adj Close']

data.columns=symbols

#Normal data
(data/data.ix[0]*100).plot(figsize=(8,5))

#Calculte mean-variance
rets=np.log(data/data.shift(1))

#Annualized returns
rets.mean()*252

#AAPL    0.267080
#MSFT    0.114505
#YHOO    0.196165
#DB     -0.125174
#GLD     0.016054
#dtype: float64


#Covariance matrix
rets.cov()*252
#          AAPL      MSFT      YHOO        DB       GLD
#AAPL  0.072784  0.020459  0.023243  0.041027  0.005231
#MSFT  0.020459  0.049402  0.024244  0.046089  0.002105
#YHOO  0.023243  0.024244  0.093349  0.051538 -0.000864
#DB    0.041027  0.046089  0.051538  0.177517  0.008777
#GLD   0.005231  0.002105 -0.000864  0.008777  0.032406

#Calculate portfolio returns given a weight for each stock
#Assumed ranodmly selected weights for now

#Portfolio mean return
weights=np.random.random(noa)
weights/=np.sum(weights)

#Expected return
np.sum(rets.mean()*weights)*252
#0.06442232237746258

#Expected variance
np.dot(weights.T,np.dot(rets.cov()*252,weights))
#0.024930239766391204

#Epxected portfolio standard deviation or volatility
np.sqrt(np.dot(weights.T,np.dot(rets.cov()*252,weights)))
#0.1578931276730916

#Use monte carlo to show array of different returns and variances of portfolios
prets=[]
pvols=[]
for p in range(2500):
    weights=np.random.rand(noa)
    weights/=np.sum(weights)
    prets.append(np.sum(rets.mean()*weights*252))
    pvols.append(np.sqrt(np.dot(weights.T,np.dot(rets.cov()*252,weights))))

prets=np.array(prets)
pvols=np.array(pvols)

#Plot results
plt.figure(figsize=(8,4))
plt.scatter(pvols,prets,c=prets/pvols,marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

#Define portfolio stats function
def statistics(weights):
    '''Returns portfolio statistics.
    
    Parameters
    =========
    pret: float expected portfolio returns
    pvol: float expected portfolio volatility
    sharpe: float shape ratios for rf=0
    '''
    weights=np.array(weights)
    pret=np.sum(rets.mean()*weights)*252
    pvol=np.sqrt(np.dot(weights.T,np.dot(rets.cov()*252,weights)))
    sharpe=pret/pvol
    return np.array([pret,pvol,sharpe])

#Create minimization function for sharpe ratio to acheive maximum sharpe ratio
def min_func_sharpe(weights):
    return -statistics(weights)[2]

#define constraints and bounds
cons=({'type':'eq','fun':lambda x: np.sum(x)-1})
bnds=tuple((0,1) for x in range(noa))

#Assume inital call is equal distribution

#Calculate optimal portfolio with maximum sharpe ratio
opts=sco.minimize(min_func_sharpe,noa*[1./noa],method='SLSQP',bounds=bnds,constraints=cons)

#Review results
opts
#status: 0
# success: True
#    njev: 5
#    nfev: 36
#     fun: -1.0630084842320222
#       x: array([  6.61851669e-01,   8.64635738e-02,   2.51684758e-01,
#         0.00000000e+00,   1.23876874e-16])
# message: 'Optimization terminated successfully.'
#     jac: array([ -1.82956457e-04,  -7.02321529e-04,   7.18012452e-04,
#         1.51409794e+00,   1.54869258e-03,   0.00000000e+00])
#     nit: 5

#Optimal potfolio allocations
opts['x'].round(3)
#array([ 0.662,  0.086,  0.252,  0.   ,  0.   ])

#Review expected portfolio statistics
statistics(opts['x'].round(3))
#array([ 0.23608794,  0.22209416,  1.06300832])

#Calculate minimum variance portfolio
def min_function_variance(weights):
        return statistics(weights)[1]*2

optv=sco.minimize(min_function_variance,noa*[1./noa],method='SLSQP',bounds=bnds,constraints=cons)

optv
#  status: 0
# success: True
#    njev: 5
#    nfev: 35
#     fun: 0.27046622701255724
#       x: array([ 0.10812856,  0.24854224,  0.10944638,  0.        ,  0.53388282])
# message: 'Optimization terminated successfully.'
#     jac: array([ 0.27052099,  0.27016991,  0.27055536,  0.38773764,  0.27057482,  0.        ])
#     nit: 5

#min varaince potfolio allocations
optv['x'].round(3)
#array([ 0.108,  0.249,  0.109,  0.   ,  0.534])

#Review expected portfolio statistics
statistics(optv['x'].round(3))
#array([ 0.08731149,  0.13523311,  0.64563698])

#Calculating the efficient frontier
#define constraints and bounds
cons=({'type':'eq','fun':lambda x: np.sum(x)-1})
bnds=tuple((0,1) for x in range(noa))

#define minimization function
def min_func_port(weights):
    return statistics(weights)[1]

#Minimization
trets=np.linspace(0.0,0.25,50)
tvols=[]
for tret in trets:
    cons=({'type':'eq','fun':lambda x: statistics(x)[0]-tret}
        ,{'type':'eq','fun':lambda x: np.sum(x)-1})    
    res=sco.minimize(min_func_port,noa*[1./noa,],method='SLSQP',
                     bounds=bnds,constraints=cons)
    tvols.append(res['fun'])
tvols=np.array(tvols)


#Plot efficient fontier
plt.figure(figsize=(8,4))
plt.scatter(pvols,prets,
            c=prets/pvols,marker='o')#random potfolio composition
plt.scatter(tvols,trets,c=trets/tvols,marker='x')#efficient frontier
plt.plot(statistics(opts['x'])[1],statistics(opts['x'])[0],'r*',markersize=15.0)#Portfolio with highest sharpe ratio
plt.plot(statistics(optv['x'])[1],statistics(optv['x'])[0],'y*',markersize=15.0)#Portfolio with minimum variance
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe Ratio')

#Capital markets line
ind=np.argmin(tvols)
evols=tvols[ind:]
erets=trets[ind:]

#Perform interpolation
tck=sci.splrep(evols,evols)

#Define continuously differntialable function through interpolation
def f(x):
    '''Efficient frontier function (slines approx).'''
    return sci.splev(x,tck,der=0)

def df(x):
    '''First derivative of efficent frontier func.'''
    return sci.splev(x,tck,der=1)

#Numerically solve system of equations to get cml
def equations(p,rf=0.01):
    eq1=rf-p[0]
    eq2=rf+p[1]*p[2]-f(p[2])
    eq3=p[1]-df(p[2])
    return eq1, eq2, eq3

#perform optimization to generate line
opt=sco.fsolve(equations,[0.01,1,0])

opt    
#array([  1.00000000e-02,   9.99734217e-01,   3.75808539e+01])

plt.figure(figsize=(8,4))
plt.scatter(pvols,prets,c=(prets-0.01)/pvols,marker='o')
plt.plot(evols,erets,'g',lw=4.0)
cx=np.linspace(0.0,0.3)
plt.plot(cx,opt[0]+opt[1]*cx,lw=1.5)
plt.plot(opt[2],f(opt[2]),'r*', markersize=15.0)
plt.grid(True)
plt.axhline(0,color='k',ls='--',lw=2.0)
plt.axvline(0,color='k',ls='--',lw=2.0)
plt.xlabel('expected volatilities')
plt.ylabel('expected returns')
plt.colorbar(label='Sharpe ratio')


######################################
#####Principal Component Analysis#####
######################################

#DAX symbols
symbols=['ADS.DE','ALV.DE', 'BAS.DE', 'BAYN.DE', 'BEI.DE', 
         'BMW.DE', 'CBK.DE', 'CON.DE', 'DAI.DE', 'DBK.DE', 
         'DB1.DE', 'DPW.DE','DTE.DE', 'EOAN.DE', 'FRE.DE', 'FME.DE', 'HEI.DE', 
         'HEN3.DE', 'IFX.DE','LHA.DE', 'LIN.DE','LXS.DE','MRK.DE','SDF.DE',  
         'MUV2.DE', 'RWE.DE', 'SAP.DE', 'SIE.DE', 'TKA.DE', 'VOW3.DE','^GDAXI']

data=pd.DataFrame()
for sym in symbols:
    data[sym]=web.DataReader(sym,data_source='yahoo')['Close']


#Separate out the DAX
dax=pd.DataFrame(data.pop('^GDAXI'))

#Normalize datasets
scale_function=lambda x:(x-x.mean())/x.std()
data.apply(scale_function)
#Perform PCA
pca=KernelPCA().fit(data.apply(scale_function).fillna(0))

#Review eigenvalues (only look at first ten)
pca.lambdas_[:10].round()

#Get relative weight
get_we=lambda x: x/x.sum()
get_we(pca.lambdas_)[:10]

#First compoonent explains~65% of variability

#consutruct pca index with just the first component
pca=KernelPCA(n_components=1).fit(data.apply(scale_function).fillna(0))

dax['PCA_1']=pca.transform(-data.fillna(0))

dax.apply(scale_function).plot(figsize=(8,4))

#Add in more components
pca=KernelPCA(n_components=5).fit(data.apply(scale_function).fillna(0))

pca_components=pca.transform(data.fillna(0))

weights=get_we(pca.lambdas_)

dax['PCA_5']=np.dot(pca_components,weights)

dax.apply(scale_function).plot(figsize=(8,4))

#############################
#####Bayesian Regression#####
#############################
#Done in python 3.5