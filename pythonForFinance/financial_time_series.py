# -*- coding: utf-8 -*-
"""
Created on Mon Jan 04 14:32:38 2016

@author: nmvenuti
"""

'''
Financial Time Series Data
Python for Finance
'''
import numpy as np
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
import math
import time
from urllib import urlretrieve
import datetime as dt

######################
#####Pandas intro#####
######################

#Create basic dataframe
df=pd.DataFrame([10,20,30,40],columns=['numbers'],index=['a','b','c','d'])

df

#Show indicies
df.index

#show columns
df.columns

#selct via index value
df.ix['c']

#select via multiple indicies
df.ix[['a','b']]

#Select via index object
df.ix[df.index[0:2]]

#sum per column
df.sum()

#square every element
df.apply(lambda x: x**2)

#Can also perform math in numpy like operations

df**2

#Add new column
df['floats']=[1.5,2.5,3.5,4.5]

df

#Select just the column
df['floats']

#combine two dataframes by similar indicies

df['names']=pd.DataFrame(['Yves','Guido','Felix','Francesc'],index=['a','b','c','d'])
df

#Append data to dataframe
df =df.append(pd.DataFrame({'numbers':100,'floats':5.75,'names':'Henry'},index=['z']))

df

#Adding columns with joins
#Note very flexible with incoroporating missing values
#Just create temp object for demo
df.join(pd.DataFrame([1,2,9,16,25],index=['a','b','c','d','y'],columns=['squares']))


#Use outter join to incorporate non-existant indicies (Just like SQL)
df=df.join(pd.DataFrame([1,2,9,16,25],index=['a','b','c','d','y'],columns=['squares']),how='outer')
df

#Operations still work with NAN
#For example column wise means, standard deviation
df[['numbers','squares']].mean()
df[['numbers','squares']].std()

#Generate random data for testing purposes
a=np.random.standard_normal((9,4))
a.round(6)

#Convert numpy array to pandas dataframe
df=pd.DataFrame(a)
df

#Rename columns
df.columns=['No1','No2','No3','No4']
df

#Reference the value in Column No2 at index position 3
df['No2'][3]

#Convert index to date-time with the first date being January 2015, and each entry being monthly
dates=pd.date_range('2015-1-1',periods=9,freq='M')
dates

df.index=dates
df

#Convert dataframe back to array
np.array(df).round(6)

#Basic analytic methods straight from pandas
df.sum()

df.mean()

df.cumsum()

df.describe()

#Using np functions on dataframes

np.sqrt(df)

#Pandas is error tolerant and will calulate with NaNs as if they don't exist
np.sqrt(df).sum()

#One line ploting in pandas
df.cumsum().plot(lw=2.0)

#Panadas series data class

type(df)

type(df['No1'])

#Can perform many of the same task as with dataframes
df['No1'].cumsum().plot(style='r',lw=2)
plt.xlabel('date')
plt.ylabel('value')

#Groupby Operations
#Similar to Group by in SQL

df['quarter']=['Q1','Q1','Q1','Q2','Q2','Q2','Q3','Q3','Q3']
df

#Group data by quarter
groups=df.groupby('quarter')

#Get mean, max, and size of each value
groups.mean()
groups.max()
groups.size()

#Grouping on multiple columns
df['odd_even']=['odd','even','odd','even','odd','even','odd','even','odd']
df

groups=df.groupby(['quarter','odd_even'])
groups.mean()
groups.max()
groups.size()

########################
#####Financial Data#####
########################

#Extract German DAX index from yahoo! finance
DAX=web.DataReader(name='^GDAXI', data_source='yahoo',start='2000-1-1')
DAX.info()

#View tail of data
DAX.tail()

#Generate quick plot of DAX closing prices
DAX['Close'].plot(figsize=(8,5))

#Add daily log returns using formal loops
time1=time.time()
pd.options.mode.chained_assignment = None
DAX['Ret_loop']=0.0
for i in range(1,len(DAX)):
    DAX['Ret_loop'][i]=np.log(DAX['Close'][i]/DAX['Close'][i-1])

print time.time()-time1
#0.634999990463

#Add daily log returns using vectorization
time1=time.time()
DAX['Return']=np.log(DAX['Close']/DAX['Close'].shift(1))
print time.time()-time1    
#0.0510001182556

#Delete unused returns column
del DAX['Ret_loop']

#Create chart showing Close and returns
DAX[['Close','Return']].plot(subplots=True,style='b',figsize=(8,5))

#Accurately shows concept of Volatility clustering and leverage effect

#Add 42 day and 252 day rolling averages for trend analysis
DAX['42d']=pd.rolling_mean(DAX['Close'],window=42)
DAX['252d']=pd.rolling_mean(DAX['Close'],window=252)

DAX[['42d','252d']].tail()

#Create stock chart with 2 rolling means plotted alongside it
DAX[['Close','42d','252d']].plot(figsize=(8,5))

#Calculate moving historical volatility
DAX['Mov_Vol']=pd.rolling_std(DAX['Return'],window=252)*math.sqrt(252)

#Review hypothesis of the leverage effect that historical moving volatility
#tends to increase when markets come down, and vis versa

DAX[['Close','Mov_Vol','Return']].plot(subplots=True,style='b',figsize=(8,7))
#Trends seem to hold

#############################
#####Regression Analysis#####
#############################

es_url ='http://www.stoxx.com/download/historical_values/hbrbcpe.txt'
vs_url='http://www.stoxx.com/download/historical_values/h_vstoxx.txt'
urlretrieve(es_url,'C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance/Data/es.txt')
urlretrieve(vs_url,'C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance/Data/vs.txt')

#Import ES data, do some data cleaning
lines=open('C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance/Data/es.txt').readlines()
lines = [line.replace(' ','') for line in lines]

#Review head
lines[:5]

#Uneeded semicolon showing up between lines 3883 and 3990, remove and kick clean data to new file
new_file=open('C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance/Data/es50.txt','w')

#Write desired lines (Not inlcuing uneeded headers)
#Start by writing corrected third line of old file as first line of new file
new_file.writelines('date'+lines[3][:-1]+';DEL'+lines[3][-1])

#write remaining lines as normal
new_file.writelines(lines[4:])

new_file.close()

#Review new file
newlines = open('C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance/Data/es50.txt').readlines()
newlines[:5]

#Required cleaning done, ready to inport as dataframe
es_filepath='C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance/Data/es50.txt'
es=pd.read_csv(es_filepath,index_col=0,parse_dates=True,sep=';',dayfirst=True)

#Review data
np.round(es.tail())

#Remove helper column
del es['DEL']

es.info()

#We can also do lines 227 through 263 in 3 lines
cols=['SX5P','SX5E','SXXP','SXXE','SXXF','SXXA','DK5F','DKXF']
es=pd.read_csv(es_url,index_col=0,parse_dates=True,sep=";",dayfirst=True,header=None,skiprows=4,names=cols)
es.tail()

#VSTOXX is much cleaner and can easily be imported
vs=pd.read_csv(vs_url,index_col=0,header=2,parse_dates=True,sep=',',dayfirst=True)
vs.info()

#As VSTOXX data starts in January 1999, merge both data sets after that date
data=pd.DataFrame({'EUROSTOXX':es['SX5E'][es.index>dt.datetime(1999,1,1)]})
data=data.join(pd.DataFrame({'VSTOXX':vs['V2TX'][vs.index>dt.datetime(1999,1,1)]}))

#Fill NaN values with last available value
data=data.fillna(method='ffill')
data.info()

data.tail()

#Plot VSTOXX vs EUROSTOXX
data.plot(subplots=True,grid=True,style='b',figsize=(8,6))

#Create new dataframe of logistic daily returns
rets=np.log(data/data.shift(1))

#Plot daily return graphs
rets.plot(subplots=True,grid=True,style='b',figsize=(8,6))

#Conduct regression analysis
xdat=rets['EUROSTOXX']
ydat=rets['VSTOXX']
model=pd.ols(y=ydat,x=xdat)
model

#-------------------------Summary of Regression Analysis-------------------------
#
#Formula: Y ~ <x> + <intercept>
#
#Number of Observations:         4357
#Number of Degrees of Freedom:   2
#
#R-squared:         0.5404
#Adj R-squared:     0.5403
#
#Rmse:              0.0394
#
#F-stat (1, 4355):  5120.5780, p-value:     0.0000
#
#Degrees of Freedom: model 1, resid 4355
#
#-----------------------Summary of Estimated Coefficients------------------------
#      Variable       Coef    Std Err     t-stat    p-value    CI 2.5%   CI 97.5%
#--------------------------------------------------------------------------------
#             x    -2.8392     0.0397     -71.56     0.0000    -2.9170    -2.7614
#     intercept     0.0000     0.0006       0.03     0.9774    -0.0012     0.0012
#---------------------------------End of Summary---------------------------------

model.beta
#x           -2.839209
#intercept    0.000017
#dtype: float64

#Plot linear model along with data to show visualization of leverage effect
plt.plot(xdat,ydat,'r.')
ax=plt.axis()
x=np.linspace(ax[0],ax[1]+0.01)
plt.plot(x,model.beta[1]+model.beta[0]*x,'b',lw=2)
plt.grid(True)
plt.axis('tight')
plt.xlabel('Euro STOXX 50 returns')
plt.ylabel('VSTOXX returns')

#Output correlation between 2 financial time series
rets.corr()
#           EUROSTOXX    VSTOXX
#EUROSTOXX   1.000000 -0.735117
#VSTOXX     -0.735117  1.000000

#Plot 252day rolling corelations
pd.rolling_corr(xdat,ydat,window=252).plot(grid=True,style='b')

#############################
#####High Frequency Data#####
#############################

url1='http://hopey.netfonds.no/posdump.php?'
url2='date=%s%s%s&paper=APPL.O&csv_format=csv'
url=url1+url2
year='2014'
month='09'
days =['22','23','24','25']

APPL=pd.DataFrame()
for day in days:
    print url % (year, month,day)
    APPL.append(pd.read_csv(url % (year, month,day), index_col=0,header=0,parse_dates=True))
APPL.columns = ['bid','bdepth','bdeptht','offer','odepth','odeptht']

APPL.info()