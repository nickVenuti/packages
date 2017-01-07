# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 09:03:47 2015

@author: nmvenuti
"""

#
#Valuation of European call options in Black-Scholes-Merton model
#including Vega function and implied volatilities estimation
#bsm_functions.py
#adapted from Python for Finance
#

#Analytical Back-Scholes-Merton (BSM) Formula

def bsm_call_value(S0,K,T,r,sigma):
    ''''Valuation of European call option in BSM model.
    Analytical formula.
    
    Parameters
    ==========
    S0: float
        inital stock/index level
    K: float
        strike prices
    T: float
        maturity date (in year fractions)
    r: float
        Constant risk-free short rate
    sigma: float
        volatitlity factor in diffusion term
    
    Returns
    ======
    value: float
        preent value of the Europen call option
    '''
    from math import log,sqrt,exp
    from scipy import stats
    
    S0=float(S0)
    d1=(log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*sqrt(T))
    d2=(log(S0/K)+(r-0.5*sigma**2)*T)/(sigma*sqrt(T))
    value=(S0*stats.norm.cdf(d1,0.0,1.0)-K*exp(-r*T)*stats.norm.cdf(d2,0.0,1.0))
    #Note: stats.norm.cdf returns cumulative distribution function for normal distribution
    return value

#BSM Vega function

def bsm_vega(S0,K,T,r,sigma):
    '''Vega of Europran option in BSM model.
    
    Parameters
    ==========
    S0: float
        inital stock/index level
    K: float
        strike prices
    T: float
        maturity date (in year fractions)
    r: float
        Constant risk-free short rate
    sigma: float
        volatitlity factor in diffusion term
    
    Returns
    ======
    vega: float
        partial derivative of BSM formula with respect to sigma, i.e. Vega
    '''
    from math import log,sqrt
    from scipy import stats
    
    S0=float(S0)
    d1=(log(S0/k)+(r+0.5*sigma**2)*T/(sigma*sqrt(T)))
    vega=S0*stats.norm.cdf(d1,0.0,1.0)*sqrt(T)
    return vega

#Implied volatility function

def bsm_call_imp_vol(S0,K,T,r,C0,sigma_est, it=100):
    '''Implied volatility of European call option in BSM model.
    
    Parameters
    ==========
    S0: float
        inital stock/index level
    K: float
        strike prices
    T: float
        maturity date (in year fractions)
    r: float
        Constant risk-free short rate
    sigma_est: float
        estimate of impl. volatility 
    it: integer
        number of iterations
    
    Returns:
    =======
    sigma_est: float
        numerically estimated implied volatility
    '''
    
    for i in range(it):
        sigma_est-=((bsm_call_value(S0,K,T,r,sigma_est)-C0)/bsm_vega(S0,K,T,r,sigma_est))
    return sigma_est
