# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 15:22:51 2015

@author: nmvenuti
Technical analyis example
Python for Finance
"""

import numpy as np
import pandas as pd
import pandas.io.data as web

#Get S&P 500 data from 1/1/2000 to 4/4/2014
sp500=web.DataReader('^GSPC',data_source='yahoo', start='1/1/2000',end='4/14/2014')
sp500.info()

#Plot closing prices
sp500['Close'].plot(grid=True,figsize=(8,5))

#Generate trend data for 42d and 252d rolling averages
sp500['42d']=np.round(pd.rolling_mean(sp500['Close'],window=42),2)
sp500['252d']=np.round(pd.rolling_mean(sp500['Close'],window=252),2)

#Verify columns accruately created
sp500[['Close','42d','252d']].tail()

#Plot analysis columns
sp500[['Close','42d','252d']].plot(grid=True,figsize=(8,5))

'''
Trend analysis shows the following:
Buy signal (go long): the 42d trend is for the first time SD points above the 252d trend

Wait(park in cash): the 42d trend is within a range of +/- SD points around the 252 trend

Sell signal(go short): the 42d trend is for the first time SD points below the 252 trend
'''

#Add 42d-252d column
sp500['42-252']=sp500['42d']-sp500['252d']

sp500['42-252'].tail()

#Assume a value of 50 for signal threshold and generate regime column
SD=50
sp500['Regime']=np.where(sp500['42-252']>SD,1,0)
sp500['Regime']=np.where(sp500['42-252']<-SD,-1,sp500['Regime'])
sp500['Regime'].value_counts()

#Plot regime
sp500['Regime'].plot(lw=1.5)
plt.ylim([-1.1,1.1])

#Calculate daily log returns for market
sp500['Market']=np.log(sp500['Close']/sp500['Close'].shift(1))

#Calculate strategy returns
sp500['Strategy']=sp500['Regime'].shift(1)*sp500['Market']
sp500[['Market','Strategy']].cumsum().apply(np.exp).plot(grid=True,figsize=(8,5))