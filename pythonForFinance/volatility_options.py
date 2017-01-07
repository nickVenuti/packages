# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:43:27 2016

@author: nmvenuti
Chapter 19
Volatility Options
"""
import numpy as np
import pandas as pd

#Get VSTOXX Data
url='http://www.stoxx.com/download/historical_values/h_vstoxx.txt'
vstoxx_index=pd.read_csv(url,index_col=0,header=2,parse_dates=True,dayfirst=True)

vstoxx_index.info()
#<class 'pandas.core.frame.DataFrame'>
#DatetimeIndex: 4334 entries, 1999-01-04 to 2016-01-12
#Data columns (total 9 columns):
#V2TX    4334 non-null float64
#V6I1    3885 non-null float64
#V6I2    4334 non-null float64
#V6I3    4274 non-null float64
#V6I4    4334 non-null float64
#V6I5    4334 non-null float64
#V6I6    4317 non-null float64
#V6I7    4334 non-null float64
#V6I8    4320 non-null float64
#dtypes: float64(9)
#memory usage: 338.6 KB

#For current analysis only looking at Q1 2014
vstoxx_index=vstoxx_index[('2013/12/31'<vstoxx_index.index)&('2014/4/1'>vstoxx_index.index)]
