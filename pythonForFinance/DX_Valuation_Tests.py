# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:10:03 2016

@author: nmvenuti
Testing for derivatives valuation
"""

import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance/DX_Lib')
from dx import *
from dx_simulation import *
from dx_valuation import *
import numpy as np

#Set up Market environment and assume gbm for underlying
me_gbm=market_environment('me_gbm',dt.datetime(2015,1,1))

me_gbm.add_constant('initial_value',36.)
me_gbm.add_constant('volatility',0.2)
me_gbm.add_constant('final_date',dt.datetime(2015,12,31))
me_gbm.add_constant('currency','EUR')
me_gbm.add_constant('frequency','M')
me_gbm.add_constant('paths',10000)

csr=constant_short_rate('csr',0.06)
me_gbm.add_curve('discount_curve',csr)

gbm=geometric_brownian_motion('gbm',me_gbm)

#Set up market environment for option itself
me_call=market_environment('me_call',me_gbm.pricing_date)
me_call.add_constant('strike',40.)
me_call.add_constant('maturity',dt.datetime(2015,12,31))
me_call.add_constant('currency','EUR')

#Payoff funcion for call
payoff_func='np.maximum(maturity_value-strike,0)'

#Perform european call
eur_call=valuation_mcs_european('eur_call',underlying=gbm,mar_env=me_call,payoff_func=payoff_func)

#Get present value
eur_call.present_value()
#2.090911

#Delta of the option is postiive (which is expected for Euro Option). Present value of the option
#increases with increasing inital value of the underlying
eur_call.delta()
#0.5421

#Vega-Shows increase in present value of the option given an increas in the inital volatility of 1%
eur_call.vega()
#14.2782

#Present value,Delta, vega for inital values of the underlying ranging from 24-26 Euro
s_list=np.arange(34.,46.1,2.)
p_list=[]
d_list=[]
v_list=[]
for s in s_list:
    eur_call.update(initial_value=s)
    p_list.append(eur_call.present_value(fixed_seed=True))
    d_list.append(eur_call.delta())
    v_list.append(eur_call.vega())

plot_option_stats(s_list,p_list,d_list,v_list)

#Perform same analysis with mixture between Asian and regular payoff
#Derived payoff dependent on both the simulated maturity value and max value
payoff_func='np.maximum(0.33*(maturity_value+max_value)-40,0)'

eur_as_call=valuation_mcs_european('eur_as_call',underlying=gbm,mar_env=me_call,payoff_func=payoff_func)

#Present value,Delta, vega for inital values of the underlying ranging from 24-26 Euro
s_list=np.arange(34.,46.1,2.)
p_list=[]
d_list=[]
v_list=[]
for s in s_list:
    eur_as_call.update(initial_value=s)
    p_list.append(eur_as_call.present_value(fixed_seed=True))
    d_list.append(eur_as_call.delta())
    v_list.append(eur_as_call.vega())

plot_option_stats(s_list,p_list,d_list,v_list)

#American options tests
#Set up Market environment and assume gbm for underlying
me_gbm=market_environment('me_gbm',dt.datetime(2015,1,1))

me_gbm.add_constant('initial_value',36.)
me_gbm.add_constant('volatility',0.2)
me_gbm.add_constant('final_date',dt.datetime(2015,12,31))
me_gbm.add_constant('currency','EUR')
me_gbm.add_constant('frequency','W')
me_gbm.add_constant('paths',50000)

csr=constant_short_rate('csr',0.06)
me_gbm.add_curve('discount_curve',csr)

gbm=geometric_brownian_motion('gbm',me_gbm)

#Set up payoff function for American option put
payoff_func='np.maximum(strike-instrument_values,0)'

#First run with 1 year maturity and strike price of 40
me_am_put=market_environment('me_am_put',dt.datetime(2015,1,1))

me_am_put.add_constant('maturity',dt.datetime(2015,12,31))
me_am_put.add_constant('strike',40.)
me_am_put.add_constant('currency','EUR')

am_put=valuation_mcs_american('am_put',underlying=gbm,mar_env=me_am_put,payoff_func=payoff_func)


#Conduct valuation
am_put.present_value(fixed_seed=True,bf=5)
#4.466516

#Represents a lower bound of the mathmatically correct AMerican option value. 
#Therefore, we would expect the numerial estimate to line under the true value
#in any real case

#Try to replicate the Table 1 of the oringial paper
ls_table=[]
for initial_value in (36.,38.,40.,42.,44.):
    for volatility in (0.2,0.4):
        for maturity in (dt.datetime(2015,12,31),dt.datetime(2016,12,31)):
            me_am_put.add_constant('maturity',maturity)
            me_gbm.add_constant('initial_value',initial_value)
            me_gbm.add_constant('volatility',volatility)
            gbm=geometric_brownian_motion('gbm',me_gbm)
            am_put=valuation_mcs_american('am_put',underlying=gbm,mar_env=me_am_put,payoff_func=payoff_func)
            ls_table.append([initial_value,volatility,maturity,am_put.present_value(fixed_seed=True,bf=5)])

print "S0  | Vola | T | Value "
print 22*"-"
for r in ls_table:
    print '%2.0f | %3.1f | %1.0f | %5.3f '% (r[0],r[1],r[2].year-2014,r[3])

#S0  | Vola | T | Value 
#----------------------
#36 | 0.2 | 1 | 4.467 
#36 | 0.2 | 2 | 4.529 
#36 | 0.4 | 1 | 7.023 
#36 | 0.4 | 2 | 8.007 
#38 | 0.2 | 1 | 3.214 
#38 | 0.2 | 2 | 3.490 
#38 | 0.4 | 1 | 6.065 
#38 | 0.4 | 2 | 7.201 
#40 | 0.2 | 1 | 2.263 
#40 | 0.2 | 2 | 2.656 
#40 | 0.4 | 1 | 5.206 
#40 | 0.4 | 2 | 6.493 
#42 | 0.2 | 1 | 1.553 
#42 | 0.2 | 2 | 2.029 
#42 | 0.4 | 1 | 4.464 
#42 | 0.4 | 2 | 5.868 
#44 | 0.2 | 1 | 1.056 
#44 | 0.2 | 2 | 1.543 
#44 | 0.4 | 1 | 3.805 
#44 | 0.4 | 2 | 5.311 
