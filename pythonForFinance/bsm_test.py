# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 09:36:41 2015

@author: nmvenuti
"""

'''
BSM test using VSTOXX index
Following pgs 54-59 of Python for Finance
'''
#Module imports
import pandas as pd
import matplotlib.pyplot as plt
from sys import path
path.append("C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance")
from bsm_functions import *


#Assuming t=0 on 3/31/14
V0=17.6639
r=0.01


path = 'C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance/'

store = pd.HDFStore(path + 'index_option_series.h5', 'r')

#import futures and options data to calculate remaining inputs
h5=pd.HDFStore('./source/vstoxx_data_31032014.h5','r')
futures_data=h5['futures_data'] #VSTOXX futures data
options_data=h5['options_data'] #VSTOXX call option data
h5.close()

#check futurs and options data

futures_data

options_data.info()

options_data[['DATA','MATURITY','TTM','STRIKE','PRICE']].head()

#add new column for implied volatilities
options_data['IMP_VOL']=0.0

#calculate implied volatilties for all call options w/ tolerance level for moneyness 50%
tol=0.5
#iterating over all option quotes
for option in option_data.index:
    #selecting future value
    forward=futures_data[futures_date['MATURITY'] ==\
        options_data.loc[option]['MATURITY']]['PRICE'].values[0]
    #extracting futures within tolerance limit
    if(forward*(1-tol)<options_data.loc[option]['STRIKE']<forward*(1+tol)):
        imp_vol=bsm_call_imp_vol(
                V0,
                options_data.loc[option]['STRIKE'],
                options_data.loc[option]['TTM'],
                r,
                options_data.loc[option]['PRICE'],
                sigma_est=2,
                it=100)
        options_data['IMP_VOL'].loc[option]=imp_vol

#plot implied volatilities
plot_data=options_data[options_data['IMP_VOL']>0]

maturities = sorted(set(options_data['MATURITY']))

plt.figure(figsize=(8,6))
for maturity in maturities:
    #select data for maturity
    data=plot_data[options_data.Maturity == maturity]
    plt.plot(data['STRIKE'],data['IMP_VOL'],lable=maturity.date(),lw=1.5)
    plt.plot(data['STRIKE'],data['IMP_VOL'],'r.')
plt.grid(True)
plt.xlable('strike')
plt.ylable('implied volatility of volatitlity')
plt.legend()
plt.show()

#group data for simplicity
keep = ['PRICE','IMP_VOL']
group_data=plot_data.groupby(['MATURITY','STRIKE'])[keep]
group_data=group_data.sum()
group_data.head()