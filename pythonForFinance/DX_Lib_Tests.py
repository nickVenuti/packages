# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 09:30:19 2016

@author: nmvenuti
"""
import sys
sys.path.append('C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance/DX_Lib')
from constant_short_rate import *
from market_environment import *

#Test constant_short_rate
import datetime as dt

dates=[dt.datetime(2015,1,1),dt.datetime(2015,7,1),dt.datetime(2016,1,1)]
csr=constant_short_rate('csr',0.05)
csr.get_discount_factors(dates,dtobjects=True)
get_year_deltas(dates)

#test market environment
me_1=market_environment('me_1',dt.datetime(2015,1,1))

me_1.add_list('symbols',['APPL','MSFT','FB'])

me_1.get_list('symbols')

me_2=market_environment('me_2',dt.datetime(2015,1,1))

me_2.add_constant('volatility',0.2)

#add instance of discounting class
me_2.add_curve('short_rate',csr)

me_2.get_curve('short_rate')

me_1.add_environment(me_2)

me_1.get_curve('short_rate')

me_1.constants

me_1.lists

me_1.get_curve('short_rate').short_rate