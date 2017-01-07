# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:48:51 2016

@author: nmvenuti
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance/DX_Lib')
from dx import *


#Set up market environment for gbm
me_gbm=market_environment('me_gbm',dt.datetime(2015,1,1))

me_gbm.add_constant('initial_value',36.)
me_gbm.add_constant('volatility',0.2)
me_gbm.add_constant('currency','EUR')
me_gbm.add_constant('model','gbm')

#Set up market environment for Am Put
me_am_put=market_environment('me_am_put',dt.datetime(2015,1,1))
me_am_put.add_constant('maturity',dt.datetime(2015,12,31))
me_am_put.add_constant('strike',40.)
me_am_put.add_constant('currency','EUR')

payoff_func='np.maximum(strike-instrument_values,0)'

#Set up derivative postion
am_put_pos=derivatives_position(name='am_put_pos',quantity=3,underlying='gbm',mar_env=me_am_put,otype='American',payoff_func=payoff_func)

#Print deriviative information
am_put_pos.get_info()


#derivativies_portfolio testing
me_jd=market_environment('me_jd',me_gbm.pricing_date)

#Add jd specific parameters
me_jd.add_constant('lambda',0.3)
me_jd.add_constant('mu',-0.75)
me_jd.add_constant('delta',0.1)

#Add other parameters from gbm
me_jd.add_environment(me_gbm)

#needed for portfolio valuation
me_jd.add_constant('model','jd')

#Conduct a European call option based on this new simulation
me_eur_call=market_environment('me_eur_call',me_jd.pricing_date)

me_eur_call.add_constant('maturity',dt.datetime(2015,6,30))
me_eur_call.add_constant('strike',38.)
me_eur_call.add_constant('currency','EUR')

payoff_func='np.maximum(maturity_value-strike,0)'

eur_call_pos=derivatives_position(name='eur_call_pos',quantity=5,underlying='jd',mar_env=me_eur_call,otype='European',payoff_func=payoff_func)

#Set up relevent market
underlyings={'gbm':me_gbm,'jd':me_jd}
positions={'am_put_pos':am_put_pos,'eur_call_pos':eur_call_pos}

#Compile market environment for the portfolio valuation
#Discounting object for the valuation
csr=constant_short_rate('csr',0.06)

val_env=market_environment('general',me_gbm.pricing_date)
val_env.add_constant('frequency','W')
val_env.add_constant('paths',25000)
val_env.add_constant('starting_date',val_env.pricing_date)
val_env.add_constant('final_date',val_env.pricing_date)
val_env.add_curve('discount_curve',csr)


#Create porfolio
portfolio=derivatives_portfolio(name='portfolio',positions=positions,val_env=val_env,assets=underlyings,fixed_seed=True)

#Get statistics
portfolio.get_statistics()

#           name  quant.     value curr.  pos_value  pos_delta  pos_vega
#0  eur_call_pos       5  2.814638   EUR  14.073190     3.3605   42.7900
#1    am_put_pos       3  4.472021   EUR  13.416063    -2.0895   30.5181


#Get position overview
portfolio.get_positions()
#--------------------------------------------------
#NAME
#eur_call_pos 
#
#QUANTITY
#5 
#
#UNDERLYING
#jd 
#
#MARKET ENVIRONMENT
#
# **Constants**
#paths 25000
#currency EUR
#maturity 2015-06-30 00:00:00
#frequency W
#strike 38.0
#final_date 2015-12-31 00:00:00
#starting_date 2015-01-01 00:00:00
#
# **Lists**
#time_grid [datetime.datetime(2015, 1, 1, 0, 0) datetime.datetime(2015, 1, 4, 0, 0)
# datetime.datetime(2015, 1, 11, 0, 0) datetime.datetime(2015, 1, 18, 0, 0)
# datetime.datetime(2015, 1, 25, 0, 0) datetime.datetime(2015, 2, 1, 0, 0)
# datetime.datetime(2015, 2, 8, 0, 0) datetime.datetime(2015, 2, 15, 0, 0)
# datetime.datetime(2015, 2, 22, 0, 0) datetime.datetime(2015, 3, 1, 0, 0)
# datetime.datetime(2015, 3, 8, 0, 0) datetime.datetime(2015, 3, 15, 0, 0)
# datetime.datetime(2015, 3, 22, 0, 0) datetime.datetime(2015, 3, 29, 0, 0)
# datetime.datetime(2015, 4, 5, 0, 0) datetime.datetime(2015, 4, 12, 0, 0)
# datetime.datetime(2015, 4, 19, 0, 0) datetime.datetime(2015, 4, 26, 0, 0)
# datetime.datetime(2015, 5, 3, 0, 0) datetime.datetime(2015, 5, 10, 0, 0)
# datetime.datetime(2015, 5, 17, 0, 0) datetime.datetime(2015, 5, 24, 0, 0)
# datetime.datetime(2015, 5, 31, 0, 0) datetime.datetime(2015, 6, 7, 0, 0)
# datetime.datetime(2015, 6, 14, 0, 0) datetime.datetime(2015, 6, 21, 0, 0)
# datetime.datetime(2015, 6, 28, 0, 0) datetime.datetime(2015, 6, 30, 0, 0)
# datetime.datetime(2015, 7, 5, 0, 0) datetime.datetime(2015, 7, 12, 0, 0)
# datetime.datetime(2015, 7, 19, 0, 0) datetime.datetime(2015, 7, 26, 0, 0)
# datetime.datetime(2015, 8, 2, 0, 0) datetime.datetime(2015, 8, 9, 0, 0)
# datetime.datetime(2015, 8, 16, 0, 0) datetime.datetime(2015, 8, 23, 0, 0)
# datetime.datetime(2015, 8, 30, 0, 0) datetime.datetime(2015, 9, 6, 0, 0)
# datetime.datetime(2015, 9, 13, 0, 0) datetime.datetime(2015, 9, 20, 0, 0)
# datetime.datetime(2015, 9, 27, 0, 0) datetime.datetime(2015, 10, 4, 0, 0)
# datetime.datetime(2015, 10, 11, 0, 0)
# datetime.datetime(2015, 10, 18, 0, 0)
# datetime.datetime(2015, 10, 25, 0, 0) datetime.datetime(2015, 11, 1, 0, 0)
# datetime.datetime(2015, 11, 8, 0, 0) datetime.datetime(2015, 11, 15, 0, 0)
# datetime.datetime(2015, 11, 22, 0, 0)
# datetime.datetime(2015, 11, 29, 0, 0) datetime.datetime(2015, 12, 6, 0, 0)
# datetime.datetime(2015, 12, 13, 0, 0)
# datetime.datetime(2015, 12, 20, 0, 0)
# datetime.datetime(2015, 12, 27, 0, 0)
# datetime.datetime(2015, 12, 31, 0, 0)]
#
#**Curves**
#discount_curve <constant_short_rate.constant_short_rate object at 0x00000000192FF978>
#
# OPTION TYPE
#European 
#
#PAYOFF FUNCTION
#np.maximum(maturity_value-strike,0)
#
#--------------------------------------------------
#
#--------------------------------------------------
#NAME
#am_put_pos 
#
#QUANTITY
#3 
#
#UNDERLYING
#gbm 
#
#MARKET ENVIRONMENT
#
# **Constants**
#paths 25000
#currency EUR
#maturity 2015-12-31 00:00:00
#frequency W
#strike 40.0
#final_date 2015-12-31 00:00:00
#starting_date 2015-01-01 00:00:00
#
# **Lists**
#time_grid [datetime.datetime(2015, 1, 1, 0, 0) datetime.datetime(2015, 1, 4, 0, 0)
# datetime.datetime(2015, 1, 11, 0, 0) datetime.datetime(2015, 1, 18, 0, 0)
# datetime.datetime(2015, 1, 25, 0, 0) datetime.datetime(2015, 2, 1, 0, 0)
# datetime.datetime(2015, 2, 8, 0, 0) datetime.datetime(2015, 2, 15, 0, 0)
# datetime.datetime(2015, 2, 22, 0, 0) datetime.datetime(2015, 3, 1, 0, 0)
# datetime.datetime(2015, 3, 8, 0, 0) datetime.datetime(2015, 3, 15, 0, 0)
# datetime.datetime(2015, 3, 22, 0, 0) datetime.datetime(2015, 3, 29, 0, 0)
# datetime.datetime(2015, 4, 5, 0, 0) datetime.datetime(2015, 4, 12, 0, 0)
# datetime.datetime(2015, 4, 19, 0, 0) datetime.datetime(2015, 4, 26, 0, 0)
# datetime.datetime(2015, 5, 3, 0, 0) datetime.datetime(2015, 5, 10, 0, 0)
# datetime.datetime(2015, 5, 17, 0, 0) datetime.datetime(2015, 5, 24, 0, 0)
# datetime.datetime(2015, 5, 31, 0, 0) datetime.datetime(2015, 6, 7, 0, 0)
# datetime.datetime(2015, 6, 14, 0, 0) datetime.datetime(2015, 6, 21, 0, 0)
# datetime.datetime(2015, 6, 28, 0, 0) datetime.datetime(2015, 6, 30, 0, 0)
# datetime.datetime(2015, 7, 5, 0, 0) datetime.datetime(2015, 7, 12, 0, 0)
# datetime.datetime(2015, 7, 19, 0, 0) datetime.datetime(2015, 7, 26, 0, 0)
# datetime.datetime(2015, 8, 2, 0, 0) datetime.datetime(2015, 8, 9, 0, 0)
# datetime.datetime(2015, 8, 16, 0, 0) datetime.datetime(2015, 8, 23, 0, 0)
# datetime.datetime(2015, 8, 30, 0, 0) datetime.datetime(2015, 9, 6, 0, 0)
# datetime.datetime(2015, 9, 13, 0, 0) datetime.datetime(2015, 9, 20, 0, 0)
# datetime.datetime(2015, 9, 27, 0, 0) datetime.datetime(2015, 10, 4, 0, 0)
# datetime.datetime(2015, 10, 11, 0, 0)
# datetime.datetime(2015, 10, 18, 0, 0)
# datetime.datetime(2015, 10, 25, 0, 0) datetime.datetime(2015, 11, 1, 0, 0)
# datetime.datetime(2015, 11, 8, 0, 0) datetime.datetime(2015, 11, 15, 0, 0)
# datetime.datetime(2015, 11, 22, 0, 0)
# datetime.datetime(2015, 11, 29, 0, 0) datetime.datetime(2015, 12, 6, 0, 0)
# datetime.datetime(2015, 12, 13, 0, 0)
# datetime.datetime(2015, 12, 20, 0, 0)
# datetime.datetime(2015, 12, 27, 0, 0)
# datetime.datetime(2015, 12, 31, 0, 0)]
#
#**Curves**
#discount_curve <constant_short_rate.constant_short_rate object at 0x00000000192FF978>
#
# OPTION TYPE
#American 
#
#PAYOFF FUNCTION
#np.maximum(strike-instrument_values,0)
#
#---------


#Get unique values
portfolio.valuation_objects['am_put_pos'].present_value()
#4.450573

portfolio.valuation_objects['eur_call_pos'].delta()
#0.6498

#These two factors are not correlated as seen through loking at path 777 below
path_no=777
path_gbm=portfolio.underlying_objects['gbm'].get_instrument_values()[:,path_no]
path_jd=portfolio.underlying_objects['jd'].get_instrument_values()[:,path_no]

plt.figure(figsize=(7,4))
plt.plot(portfolio.time_grid,path_gbm,'r',label='gbm')
plt.plot(portfolio.time_grid,path_jd,'b',label='jd')
plt.xticks(rotation=30)
plt.legend(loc=0)
plt.grid(True)

#Consider same case with correlated risk factors
correlations=[['gbm','jd',0.9]]
port_corr=derivatives_portfolio(name='portfolio',positions=positions,val_env=val_env,assets=underlyings,fixed_seed=True)

#Get statistics
port_corr.get_statistics()
#           name  quant.     value curr.  pos_value  pos_delta  pos_vega
#0  eur_call_pos       5  2.796689   EUR  13.983445     3.4855   42.7900
#1    am_put_pos       3  4.472021   EUR  13.416063    -2.0895   30.5181

#no noticible difference in portfolio performance, however, difference is seen graphically
path_gbm=port_corr.underlying_objects['gbm'].get_instrument_values()[:,path_no]
path_jd=port_corr.underlying_objects['jd'].get_instrument_values()[:,path_no]

plt.figure(figsize=(7,4))
plt.plot(port_corr.time_grid,path_gbm,'r',label='gbm')
plt.plot(port_corr.time_grid,path_jd,'b',label='jd')
plt.xticks(rotation=30)
plt.legend(loc=0)
plt.grid(True)

#Look at frequency distribution of the portfolio present value
pv1=5*port_corr.valuation_objects['eur_call_pos'].present_value(full=True)[1]
pv2=3*port_corr.valuation_objects['am_put_pos'].present_value(full=True)[1]

plt.hist([pv1,pv2],bins=25,label=['Europea call','American Put'])
plt.axvline(pv1.mean(),color='r',ls='dashed',lw=1.5,label='call mean=%4.2f'%pv1.mean())
plt.axvline(pv2.mean(),color='r',ls='dotted',lw=1.5,label='put mean=%4.2f'%pv2.mean())
plt.xlim(0,80)
plt.ylim(0,10000)
plt.grid()
plt.legend()
