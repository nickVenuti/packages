# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 09:52:33 2015

@author: nmvenuti
"""

'''
Extract VSTOXX data for analysis
Methodology from Python for Finance and Eurex Exchange
'''
import numpy as np
import pandas as pd

#download vstoxx historic data
url = 'http://www.stoxx.com/download/historical_values/h_vstoxx.txt'
vstoxx_index=pd.read_csv(url,index_col=0,header=2,parse_dates=True,dayfirst=True)

#check data from rip to ensure complete
vstoxx_index.info()


#