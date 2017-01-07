# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:24:28 2016

@author: nmvenuti
Simulation Tests
"""
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance/DX_Lib')
from dx import *
from dx_simulation import *

#Set up market evironment for simulation
me_gbm=market_environment('me_gbm',dt.datetime(2015,1,1))
me_gbm.add_constant('initial_value',36.)
me_gbm.add_constant('volatility',0.2)
me_gbm.add_constant('final_date',dt.datetime(2015,12,31))
me_gbm.add_constant('currency','EUR')
me_gbm.add_constant('frequency','M')
me_gbm.add_constant('paths',10000)

csr=constant_short_rate('csr',0.05)
me_gbm.add_curve('discount_curve',csr)

#Define simulation
gbm=geometric_brownian_motion('gbm',me_gbm)

#Generate and review time grid
gbm.generate_time_grid()

gbm.time_grid
#array([datetime.datetime(2015, 1, 1, 0, 0),
#       datetime.datetime(2015, 1, 31, 0, 0),
#       datetime.datetime(2015, 2, 28, 0, 0),
#       datetime.datetime(2015, 3, 31, 0, 0),
#       datetime.datetime(2015, 4, 30, 0, 0),
#       datetime.datetime(2015, 5, 31, 0, 0),
#       datetime.datetime(2015, 6, 30, 0, 0),
#       datetime.datetime(2015, 7, 31, 0, 0),
#       datetime.datetime(2015, 8, 31, 0, 0),
#       datetime.datetime(2015, 9, 30, 0, 0),
#       datetime.datetime(2015, 10, 31, 0, 0),
#       datetime.datetime(2015, 11, 30, 0, 0),
#       datetime.datetime(2015, 12, 31, 0, 0)], dtype=object)
gbm.get_instrument_values()
#Generate simulated instrument values and review
paths_1=gbm.get_instrument_values()
print paths_1

#Generate instrument values for higher volatility
gbm.update(volatility=0.5)
paths_2=gbm.get_instrument_values()
print paths_2

#Print two paths
plt.figure(figsize=(8,4))
p1=plt.plot(gbm.time_grid,paths_1[:,:10], 'b')
p2=plt.plot(gbm.time_grid,paths_2[:,:10],'r-.')
plt.grid(True)
l1=plt.legend([p1[0],p2[0]],['low volatility','high volatility'],loc=2)
plt.gca().add_artist(l1)
plt.xticks(rotation=30)

#Test jump diffusion package
me_jd=market_environment('me_jd',dt.datetime(2015,1,1))

#Add jd specific parameters
me_jd.add_constant('lambda',0.3)
me_jd.add_constant('mu',-0.75)
me_jd.add_constant('delta',0.1)

#Add in me_gbm
me_jd.add_environment(me_gbm)

#Set up inital jump diffusion
jd=jump_diffusion('jd',me_jd)

paths_3=jd.get_instrument_values()

#Update lambda in jump diffusion
jd.update(lamb=0.9)
paths_4=jd.get_instrument_values()

#Print two paths
plt.figure(figsize=(8,4))
p1=plt.plot(gbm.time_grid,paths_3[:,:10], 'b')
p2=plt.plot(gbm.time_grid,paths_4[:,:10],'r-.')
plt.grid(True)
l1=plt.legend([p1[0],p2[0]],['low intensity','high intensity'],loc=2)
plt.gca().add_artist(l1)
plt.xticks(rotation=30)

#Create square root diffusion market environment
me_srd=market_environment('me_srd',dt.datetime(2015,1,1))
me_srd.add_constant('initial_value',.25)
me_srd.add_constant('volatility',0.05)
me_srd.add_constant('final_date',dt.datetime(2015,12,31))
me_srd.add_constant('currency','EUR')
me_srd.add_constant('frequency','W')
me_srd.add_constant('paths',10000)
me_srd.add_constant('kappa',4.0)
me_srd.add_constant('theta',0.2)

#Required but not needed for srd
me_srd.add_curve('discount_curve',constant_short_rate('r',0.0))

srd=square_root_diffusion('srd',me_srd)

#Get first ten instrument values
srd_paths=srd.get_instrument_values()[:,:10]

#Plot results
plt.figure(figsize=(8,4))
plt.plot(srd.time_grid,srd_paths)
plt.axhline(me_srd.get_constant('theta'),color='r',ls='--',lw=2.0)
plt.grid(True)
plt.xticks(rotation=30)