# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 13:39:24 2016

@author: nmvenuti
"""

import numpy as np
import pandas as pd

import itertools as it
import pandas.io.data as web
import scipy.stats as scs
import scipy.optimize as sco
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt
import time
import multiprocessing as mp

from sknn import mlp
#Set up multiprocessing
cores=mp.cpu_count()
pool=mp.Pool(processes=cores)


#Define functions
def rnn(layers,dataDF,trainSplit,lagConstant,iterations=100):
    
    #Split create x matrix
    x=dataDF.shift(1).dropna()
    
    #Extract prediction columns    
    predColumns=x.columns
    #Add lag
    for lag in range(2,lagConstant+2):
        xLag=x.shift(lag)
        xLag.columns=[col+'lag'+str(lag-1) for col in xLag.columns]
        try:
            xBind=xBind.join(xLag,how='left')
        except:
            xBind=xLag
    x=x.join(xBind,how='left')
    x=x.dropna()
    y=rets_sp500.ix[x.index.values]
    xTrain=x.ix[x.index.values[:trainSplit*len(x)]]
    xTest=x.ix[x.index.values[trainSplit*len(x):]]
    
    yTrain=y.ix[y.index.values[:trainSplit*len(y)]]
    yTest=y.ix[y.index.values[trainSplit*len(y):]]

    signalNN = mlp.Regressor(layers,random_state=1,n_iter=iterations)
    signalNN.fit(xTrain.as_matrix(),yTrain[predColumns].as_matrix())
    yPred=signalNN.predict(xTest.as_matrix())
    mae=np.mean(np.abs(yPred-yTest[predColumns].as_matrix()),axis=0)
    return([signalNN,mae])


##########################
#####Extract Raw Data#####
##########################
start_time=time.time()
start=dt.datetime(2014,1,1)
end=dt.datetime(2015,12,31)

#S&P 500 symbols
sp500=['A', 'AA', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABK', 'ABT', 'ACE',
       'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADS', 'ADSK', 'ADT', 'AEE', 'AEP',
       'AES', 'AET', 'AFL', 'AGN', 'AIG', 'AIV', 'AIZ', 'AKAM', 'AKS', 'ALL',
       'ALLE', 'ALTR', 'ALXN', 'AMAT', 'AMD', 'AME', 'AMG', 'AMGN', 'AMP', 
       'AMT', 'AMZN', 'AN', 'ANF', 'ANR', 'ANTM', 'AON', 'APA', 'APC', 'APD',
       'APH', 'APOL', 'ARG', 'ATI', 'ATVI', 'AV', 'AVB', 'AVGO', 'AVP', 'AVY', 
       'AXP', 'AYE', 'AZO', 'BA', 'BAC', 'BAX', 'BBBY', 'BBT', 'BBY', 'BCR', 
       'BDX', 'BEAM', 'BEN', 'BF-B', 'BHI', 'BIG', 'BIIB', 'BJS', 'BK', 'BLK', 
       'BLL', 'BMC', 'BMS', 'BMY', 'BRCM', 'BRK-B', 'BS', 'BSX', 'BTU', 'BWA',
       'BXLT', 'BXP', 'C', 'CA', 'CAG', 'CAH', 'CAM', 'CAT', 'CB', 'CBE', 'CBG', 
       'CBS', 'CCE', 'CCI', 'CCK', 'CCL', 'CEG', 'CELG', 'CEPH', 'CERN', 'CF',
       'CFN', 'CHD', 'CHK', 'CHRW', 'CI', 'CINF', 'CL', 'CLF', 'CLX', 'CMA',
       'CMCSA', 'CMCSK', 'CME', 'CMG', 'CMI', 'CMS', 'CNP', 'CNX', 'COF', 
       'COG', 'COH', 'COL', 'COP', 'COST', 'COV', 'CPB', 'CPGX', 'CPWR', 
       'CRM', 'CSC', 'CSCO', 'CSRA', 'CSX', 'CTAS', 'CTL', 'CTSH', 'CTXS', 
       'CVC', 'CVH', 'CVS', 'CVX', 'D', 'DAL', 'DD', 'DE', 'DELL', 'DF',
       'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DISCA', 'DISCK', 'DLPH', 
       'DLTR', 'DNB', 'DNR', 'DO', 'DOV', 'DOW', 'DPS', 'DRI', 'DTE', 'DTV', 
       'DUK', 'DV', 'DVA', 'DVN', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 
       'EK', 'EL', 'EMC', 'EMN', 'EMR', 'ENDP', 'EOG', 'EP', 'EQIX', 'EQR',
       'EQT', 'ES', 'ESRX', 'ESS', 'ESV', 'ETFC', 'ETN', 'ETR', 'EW', 'EXC', 
       'EXPD', 'EXPE', 'F', 'FAST', 'FB', 'FCX', 'FDO', 'FDX', 'FE', 'FFIV',
       'FHN', 'FII', 'FIS', 'FISV', 'FITB', 'FLIR', 'FLR', 'FLS', 'FMC', 'FOSL',
       'FOX', 'FOXA', 'FRX', 'FSLR', 'FTI', 'FTR', 'GAS', 'GD', 'GE', 'GENZ',
       'GGP', 'GHC', 'GILD', 'GIS', 'GLW', 'GM', 'GMCR', 'GME', 'GNW', 'GOOG',
       'GOOGL', 'GPC', 'GPS', 'GR', 'GRA', 'GRMN', 'GS', 'GT', 'GWW', 'HAL', 
       'HAR', 'HAS', 'HBAN', 'HBI', 'HCA', 'HCBK', 'HCN', 'HCP', 'HD', 'HES', 
       'HIG', 'HNZ', 'HOG', 'HON', 'HOT', 'HP', 'HPE', 'HPQ', 'HRB', 'HRL', 
       'HRS', 'HSIC', 'HSP', 'HST', 'HSY', 'HUM', 'IBM', 'ICE', 'IFF', 'IGT', 
       'ILMN', 'INTC', 'INTU', 'IP', 'IPG', 'IR', 'IRM', 'ISRG', 'ITT', 'ITW', 
       'IVZ', 'JBHT', 'JBL', 'JCI', 'JCP', 'JDSU', 'JEC', 'JNJ', 'JNPR', 'JNS', 
       'JOY', 'JOYG', 'JPM', 'JWN', 'K', 'KEY', 'KG', 'KHC', 'KIM', 'KLAC', 'KMB',
       'KMI', 'KMX', 'KO', 'KORS', 'KR', 'KRFT', 'KSS', 'KSU', 'L', 'LB', 
       'LEG', 'LEN', 'LH', 'LIFE', 'LLL', 'LLTC', 'LLY', 'LM', 'LMT', 'LNC',
       'LO', 'LOW', 'LRCX', 'LSI', 'LUK', 'LUV', 'LVLT', 'LXK', 'LYB', 'M', 
       'MA', 'MAC', 'MAR', 'MAS', 'MAT', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ',
       'MDT', 'MEE', 'MET', 'MFE', 'MHFI', 'MHK', 'MHS', 'MI', 'MIL', 'MJN',
       'MKC', 'MLM', 'MMC', 'MMI', 'MMM', 'MNK', 'MNST', 'MO', 'MOLX', 'MON',
       'MOS', 'MPC', 'MRK', 'MRO', 'MS', 'MSFT', 'MSI', 'MTB', 'MU', 'MUR', 
       'MWW', 'MYL', 'NAVI', 'NBL', 'NBR', 'NDAQ', 'NE', 'NEE', 'NEM', 'NFLX',
       'NFX', 'NI', 'NKE', 'NLSN', 'NOC', 'NOV', 'NOVL', 'NRG', 'NSC', 'NSM',
       'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVLS', 'NWL', 'NWS', 'NWSA', 'NYT',
       'NYX', 'O', 'ODP', 'OI', 'OKE', 'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 
       'PBCT', 'PBI', 'PCAR', 'PCG', 'PCL', 'PCLN', 'PCP', 'PCS', 'PDCO', 
       'PEG', 'PEP', 'PETM', 'PFE', 'PFG', 'PG', 'PGN', 'PGR', 'PH', 'PHM', 
       'PKI', 'PLD', 'PLL', 'PM', 'PNC', 'PNR', 'PNW', 'POM', 'PPG', 'PPL',
       'PRGO', 'PRU', 'PSA', 'PSX', 'PVH', 'PWR', 'PX', 'PXD', 'PYPL', 'Q', 
       'QCOM', 'QEP', 'QRVO', 'R', 'RAI', 'RCL', 'RDC', 'REGN', 'RF', 'RHI',
       'RHT', 'RIG', 'RL', 'ROK', 'ROP', 'ROST', 'RRC', 'RRD', 'RSG', 'RSH',
       'RTN', 'S', 'SAIC', 'SBL', 'SBUX', 'SCG', 'SCHW', 'SE', 'SEE', 'SGP',
       'SHLD', 'SHW', 'SIAL', 'SIG', 'SII', 'SJM', 'SLB', 'SLE', 'SLG', 'SLM', 
       'SNA', 'SNDK', 'SNI', 'SO', 'SPG', 'SPLS', 'SRCL', 'SRE', 'STI', 'STJ', 
       'STR', 'STT', 'STX', 'STZ', 'SUN', 'SVU', 'SWK', 'SWKS', 'SWN', 'SWY',
       'SYF', 'SYK', 'SYMC', 'SYY', 'T', 'TAP', 'TDC', 'TE', 'TEG', 'TEL', 
       'TER', 'TGNA', 'TGT', 'THC', 'TIE', 'TIF', 'TJX', 'TLAB', 'TMK', 'TMO',
       'TRB', 'TRIP', 'TROW', 'TRV', 'TSCO', 'TSN', 'TSO', 'TSS', 'TWC', 'TWX',
       'TXN', 'TXT', 'TYC', 'UA', 'UAL', 'UHS', 'UNH', 'UNM', 'UNP', 'UPS',
       'URBN', 'URI', 'USB', 'UTX', 'V', 'VAR', 'VFC', 'VIAB', 'VLO', 'VMC',
       'VNO', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VZ', 'WAT', 'WBA', 'WDC', 'WEC', 
       'WFC', 'WFM', 'WFR', 'WHR', 'WIN', 'WLTW', 'WM', 'WMB', 'WMT', 'WPX', 
       'WRK', 'WU', 'WY', 'WYN', 'WYNN', 'X', 'XEC', 'XEL', 'XL', 'XLNX', 
       'XOM', 'XRAY', 'XRX', 'XTO', 'XYL', 'YHOO', 'YUM', 'ZBH', 'ZION', 'ZTS']

energyStocks=['AAV','ANW','ALJ','ALDW','AEUA','APC','AR','APA','ARCX','AT','ARP','ATW','BHI','BAS','BTE','BXE','BBL','BBG','BSM','BCEI','BP','BPT','BGG','BC','BPL','BWXT','CJES','COG','CRC','CPE','CPE^A','CNQ','CVE','CGG','CHK','CHKR','CVX','MY','SNP','CYD','XEC','CWEI','CLD','CEO','CNXC','CIE','CRK','CXO','COP','CNX','CLR','CLB','CPG','CRT','CAPL','CMI','CVI','CVRR','CELP','DKL','DK','DNR','DVN','DO','DRQ','ECT','ECR','EC','EMES','EMR','EEQ','EEP','ENB','ECA','NDRO','EGN','ERF','E','ESV','EOG','EPE','EQT','XCO','XOM','FTI','FELP','FET','FI','GEH','GEK','GE','GEB','GEL','GPRK','GLP','GLF','HK','HAL','HNR','HLX','HP','HES','HEP','HFC','HGT','ICD','IOC','IO','JONE','JOY','JPEP','KEG','PHG','KOS','LPI','MIC','MMP','MRO','MPC','MTDR','MTR','MPLX','MUR','MVO','NBR','NOV','NGS','NRP','NFX','NR','NGL','NE','NBL','NOA','NADL','NRT','NTI','DNOW','NS','NSH','NSS','OAS','OXY','OII','OIS','ROYT','PACD','PHX','PKD','PE','PBF','PBFX','PBA','PGH','PWE','PBT','PZE','PTR','PBR','PQ','PSX','PSXP','PES','PXD','PAA','PAGP','PDS','QEP','RRC','REN           ','RICE','RRMS','RDC','RES','RSPP','SBR','SJT','SN','SDT','SDR','PER','SSL','SLB','SDRL','SDLP','SEMG','SHLX','SM','SWN','SWNC','SRLP','STO','SGY','SU','SXL','SUN','SPN','TSO','TLLP','TTI','TPL','THR','TOT','TLP','RIG','RIGP','TREC','UNT','EGY','VLO','VLP','VET','VOC','VTTI','WTI','WFT','WNRL','WNR','WMLP','WLL','WG','INT','WPT','WPX','WPXP','YZC','YPF']

#Remove dead symbols
dead_sym=['BF.B', 'BRK.B', 'BXLT', 'CPGX', 'CSRA', 'HPE', 'KHC', 'PYPL', 'QRVO', 'WRK']

sp500=[sym for sym in energyStocks if sym not in dead_sym]



#Extract data from start to end
data_sp500=pd.DataFrame()
for sym in energyStocks:
    try:
        data_sp500[sym]=web.DataReader(sym,data_source='yahoo',start=start,end=end)['Adj Close']
    except:
        dead_sym.append(sym)
#    Remove columns with symbols with NAN
data_sp500=data_sp500.dropna(axis=1)    


##############################
#####Select Target Stocks#####
##############################

#Calculate log daily rets
rets_sp500=np.log(data_sp500/data_sp500.shift(1))
rets_sp500=rets_sp500.fillna(0)
rets_sp500.head()



#Optimize model
layers=[mlp.Layer('Tanh', units=len(rets_sp500.columns)*3),
    mlp.Layer('Tanh', units=len(rets_sp500.columns)*3),
    mlp.Layer('Tanh', units=len(rets_sp500.columns)*3),
    mlp.Layer('Linear')]

testRun=rnn(layers,rets_sp500,0.8,3,100)
np.mean(testRun[1])
np.mean(rets_sp500.as_matrix())























#Calculate mean rets and sd
mean_sp500=np.mean(rets_sp500)
sd_sp500=np.std(rets_sp500)


#Calculate correlation matrix and array
port=list(it.combinations(sp500,2))
port=pd.DataFrame(port)
port.columns=['sym_1','sym_2',]

mean_sp500=pd.DataFrame(mean_sp500)
mean_sp500['sym_1']=mean_sp500.index
mean_sp500.columns=['ret_1','sym_1']
port=pd.merge(port, mean_sp500, left_on='sym_1',right_on='sym_1')

mean_sp500.columns=['ret_2','sym_2']
port=pd.merge(port, mean_sp500, left_on='sym_2',right_on='sym_2')
port['corr']=port.apply(lambda x:scs.pearsonr(rets_sp500[x['sym_1']],rets_sp500[x['sym_2']])[0], axis=1)

#Drop uneeded data
del mean_sp500
del sd_sp500

# filter for positive gains

port_small=port[(port['ret_1']>0.) &( port['ret_2']>0.)]

#Create total average return column
port_small['tot_ret']=port_small['ret_1']+port_small['ret_2']

#Sort pandas by neg correlation
port_small=port_small.sort(['corr'],ascending=1)

#Target stocks are those with a corrleation at most at min
target_pairs=port_small[port_small['corr']< min_corr]
target_stocks=list(set(list(target_pairs['sym_1'])+list(target_pairs['sym_2'])))



#############################################
#####Optimize Portfolio of Target Stocks#####
#############################################

#Set up inital parameters
noa=len(target_stocks)
rets=rets_sp500[target_stocks]
rets_cov=rets.cov()*252


#Use monte carlo to show array of different returns and variances of portfolios
n=simulations
pweights=np.zeros((n,noa))
prets=np.zeros((n,1))
pvols=np.zeros((n,1))
for p in range(n):
    weights=np.random.rand(noa)
    weights/=np.sum(weights)
    pweights[p,:]=weights #Extract weights for portfolio test
    prets[p]=np.sum(rets.mean()*weights*252) #Add in portfolio returns
    pvols[p]=np.sqrt(np.dot(weights.T,np.dot(rets_cov,weights))) #Add in portfolio variance

#Find optimal portfolio
port_sim=pd.DataFrame(pweights,columns=target_stocks)
port_sim['ret']=prets
port_sim['vol']=pvols
port_sim['sharpe']=prets/pvols

#Find guess of optimal portfolio
est_weights=port_sim[port_sim['sharpe']==max(port_sim['sharpe'])][target_stocks]

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
opts=sco.minimize(min_func_sharpe,est_weights,method='SLSQP',bounds=bnds,constraints=cons)


#optimal portfolio weights
port_weights=pd.DataFrame()
port_weights['Stock']=target_stocks
port_weights['Weight']=opts['x']    

#Print statistics of optimal portfolio
optimal_portfolio=statistics(list(port_weights['Weight']))

print 'Annual Returns: %4.3f Annual Volatility: %4.3f Sharpe Ratio: %4.3f' % (optimal_portfolio[0],optimal_portfolio[1],optimal_portfolio[2])

