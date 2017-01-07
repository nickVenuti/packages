# -*- coding: utf-8 -*-
"""
Created on Tue Jan 05 09:55:55 2016

@author: nmvenuti
I/O With Python
Python for Finance
"""
###############################
#####Basic I/O with Python#####
###############################

import numpy as np
import pandas as pd
import pandas.io.sql as pds
import sqlite3 as sq3
import tables as tb
import datetime as dt
import matplotlib.pyplot as plt
from random import gauss
import pickle
import time

#Path for data storage
path='C:/Users/nmvenuti/Desktop/Background Reading/Python for Finance/flash/data/'
#Generation of normally distributed randoms
a =[gauss(1.5,2) for i in range(1000000)]

#Write list to disk later through pickle file
pkl_file=open(path+'data.pkl','w')

#dump data to disk
time1=time.time()
pickle.dump(a,pkl_file)
print time.time()-time1

#2.65100002289 seconds
pkl_file

#Close pickle
pkl_file.close()

#Load pickle
pkl_file=open(path+'data.pkl','r')
time1=time.time()
b=pickle.load(pkl_file)
print time.time()-time1
#2.15299987793 seconds

#Check if both pickel files are the same
np.allclose(np.array(a),np.array(b))
#True

#Storing and retriving multiple pickle objects
pkl_file=open(path+'data.pkl','w')
time1=time.time()
pickle.dump(np.array(a),pkl_file)
print time.time()-time1
#0.730999946594 seconds

time1=time.time()
pickle.dump(np.array(a)**2,pkl_file)
print time.time()-time1
#0.691999912262 seconds

pkl_file.close()

#Read 2 n-darrays back into memory
pkl_file=open(path+'data.pkl','r')

x=pickle.load(pkl_file)

x

y=pickle.load(pkl_file)

y

#Pickle operates on First in First out princuple
#No way to store Meta data accociating data points with each dump
#Helpful to store data in dictionaries

pkl_file=open(path+'data.pkl','w')
pickle.dump({'x':x,'y':y},pkl_file)
pkl_file.close()

#This allows us to read the whole set of objects at once
pkl_file=open(path+'data.pkl','r')
data=pickle.load(pkl_file)
pkl_file.close()
for key in data.keys():
    print key,data[key][:4]

#Only problem with this is all data needs to be stored to pickle at once

#Reading and writing text files

#Create dummy data
rows=5000
a=np.random.standard_normal((rows,5))

a.round(4)

t=pd.date_range(start='2014/1/1',periods=rows,freq='H')

t

#write the data to csv
csv_file=open(path+'data.csv','w')

#create csv header
header='date,no1,no2,no3,no4,no5\n'
csv_file.write(header)

#Write each data row to csv
for t_,(no1,no2,no3,no4,no5) in zip(t,a):
    s='%s,%f,%f,%f,%f,%f\n' % (t_,no1,no2,no3,no4,no5)
    csv_file.write(s)
csv_file.close()

#Open and import data from csv
csv_file=open(path+'data.csv','r')

for i in range(5):
    print csv_file.readline()


#Can also read all content at onces using read lines methods
csv_file=open(path+'data.csv','r')
content=csv_file.readlines()
for line in content[:5]:
    print line

#Close csv
csv_file.close()

#######################
#####SQL Databases#####
#######################

#create base query
query='CREATE TABLE numbs (Date date, No1 real, No2 real)'

#Open DB connection
con=sq3.connect(path+'numbs.db')

#Run query and commit change
con.execute(query)
con.commit()

#Add a single row of data to db
con.execute('INSERT INTO numbs VALUES(?,?,?)',(dt.datetime.now(),.12,7.3))

#batch upload
data=np.random.standard_normal((10000,2)).round(5)

for row in data:
    con.execute('INSERT INTO numbs VALUES(?,?,?)',(dt.datetime.now(),row[0],row[1]))

con.commit()    

#Select 10 values from db at once
con.execute('SELECT * FROM numbs').fetchmany(10)

#Read data one line at a time
pointer=con.execute('SELECT * from numbs')

for i in range(3):
    print pointer.fetchone()

con.close()

###########################################
#####Reading and writing NumpPy Arrays#####
###########################################

dtimes=np.arange('2015-01-01 10:00:00', '2021-12-31 22:00:00',dtype='datetime64[m]')

len(dtimes)

#Set datatpyes similar to data type set up in SQL table

dty=np.dtype([('Date','datetime64[m]'),('No1','f'),('No2','f')])
data=np.zeros(len(dtimes),dtype=dty)

#Add dates
data['Date']=dtimes

#Add random numbers
a =np.random.standard_normal((len(dtimes),2)).round(5)

data['No1']=a[:,0]
data['No2']=a[:,1]

#Save numpy array to dick
time1=time.time()
np.save(path+'array',data)
print time.time()-time1
#0.457000017166 seconds

#Reading in pandas dataframe
time1=time.time()
np.load(path+'array.npy',)
print time.time()-time1
#0.0469999313354 seconds

##########################################
#####Reading and writing with Pandas######
##########################################

data=np.random.standard_normal((1000000,5)).round(5)

filename=path+'numbs'

#First use pandas and sqlite db
#First create db
query='CREATE TABLE numbers (No1 real, No2 real, No3 real, No4 real, No5 real)'

con = sq3.Connection(filename+'.db')

con.execute(query)

#apply execute many to mass upload data/commit
time1=time.time()
con.executemany('INSERT INTO numbers VALUES (?,?,?,?,?)',data)
con.commit()
print time.time()-time1
#11.3929998875 seconds

#read data in from sql lite
time1=time.time()
temp=con.execute('SELECT * FROM numbers').fetchall()
print time.time()-time1
#1.3259999752 seconds

#Read data into numpy arany, plot
time1=time.time()
query='SELECT * from numbers WHERE No1>0 AND No2<0'
res=np.array(con.execute(query).fetchall()).round(3)
print time.time()-time1
#0.734999895096 seconds

#Reduce data down to every hundreth result
res=res[::100]
plt.plot(res[:,0],res[:,1],'ro')
plt.grid(True)
plt.xlim(-0.5,4.5)
plt.ylim(-4.5,0.5)

#Using pandas sql 
time1=time.time()
data=pds.read_sql('SELECT * FROM numbers',con)
print time.time()-time1
#2.43600010872 seconds
#Bottleneck is sql database, however querying data that is now in memory is much faster using pandas

time1=time.time()
data[(data['No1']>0) & (data['No2']<0)]
print time.time()-time1
#0.0240001678467 seconds

#Using pandas for extremely complex queries
time1=time.time()
res=data[['No1','No2']][((data['No1']>0.5)|(data['No1']<-0.5))&((data['No2']<-1.)|(data['No2']>1))]
print time.time()-time1
#0.0339999198914 seconds

plt.plot(res.No1,res.No2, 'ro')
plt.grid(True)
plt.axis('tight')

#Pandas with HDF5

#writing
h5s=pd.HDFStore(filename+'.h5s','w')
time1=time.time()
h5s['data']=data
print time.time()-time1
#0.276000022888 seconds
h5s

h5s.close()

#reading
h5s=pd.HDFStore(filename+'.h5s','r')
time1=time.time()
temp=h5s['data']
print time.time()-time1
#0.0299999713898 seconds
h5s.close()

#Pandas and CSVs
#writing
time1=time.time()
data.to_csv(filename+'csv')
print time.time()-time1
#3.79099988937 seconds

#reading
time1=time.time()
x=pd.read_csv(filename+'csv')
print time.time()-time1
#0.949999809265 seconds

x[['No1','No2','No3','No4']].hist(bins=20)

#Pandas and excel
#writing
time1=time.time()
data[:100000].to_excel(filename+'.xlsx')
print time.time()-time1
#12.8150000572

#reading
time1=time.time()
pd.read_excel(filename+'.xlsx','Sheet1').cumsum().plot()
print time.time()-time1

################################
#####Fast I/O with PyTables#####
################################

#create new table
filename=path+'tab.h5'
h5=tb.open_file(filename,'w')

#generate 2,000,000 rows of data

rows=2000000

#Create row descriptions
row_des ={
    'Date':tb.StringCol(26,pos=1),
    'No1': tb.IntCol(pos=2),
    'No2': tb.IntCol(pos=3),
    'No3': tb.Float64Col(pos=4),
    'No4': tb.Float64Col(pos=5)
}

#Create table without compression
filters=tb.Filters(complevel=0)
tab=h5.create_table('/','ints_floats',row_des, title='Integers and Floats', expectedrows=rows, filters=filters)
tab

#/ints_floats (Table(0,)) 'Integers and Floats'
#  description := {
#  "Date": StringCol(itemsize=26, shape=(), dflt='', pos=0),
#  "No1": Int32Col(shape=(), dflt=0, pos=1),
#  "No2": Int32Col(shape=(), dflt=0, pos=2),
#  "No3": Float64Col(shape=(), dflt=0.0, pos=3),
#  "No4": Float64Col(shape=(), dflt=0.0, pos=4)}
#  byteorder := 'little'
#  chunkshape := (2621,)

pointer=tab.row

#Generate sample data
ran_int=np.random.randint(0,10000,size=(rows,2))
ran_float=np.random.standard_normal((rows,2)).round(5)

#write data to table
time1=time.time()
for i in range(rows):
    pointer['Date']=dt.datetime.now()
    pointer['No1']=ran_int[i,0]
    pointer['No2']=ran_int[i,1]
    pointer['No3']=ran_float[i,0]
    pointer['No4']=ran_float[i,1]
    pointer.append()
tab.flush()
print time.time()-time1
#6.64800000191 seconds

tab

#/ints_floats (Table(2000000,)) 'Integers and Floats'
#  description := {
#  "Date": StringCol(itemsize=26, shape=(), dflt='', pos=0),
#  "No1": Int32Col(shape=(), dflt=0, pos=1),
#  "No2": Int32Col(shape=(), dflt=0, pos=2),
#  "No3": Float64Col(shape=(), dflt=0.0, pos=3),
#  "No4": Float64Col(shape=(), dflt=0.0, pos=4)}
#  byteorder := 'little'
#  chunkshape := (2621,)

#Numpy version of same task

dty=np.dtype([('Date','S26'),('No1','<i4'),('No2','<i4'),('No3','<f8'),('No4','<f8')])

sarray=np.zeros(len(ran_int),dtype=dty)

sarray

time1=time.time()
sarray['Date']=dt.datetime.now()
sarray['No1']=ran_int[:,0]
sarray['No2']=ran_int[:,1]
sarray['No3']=ran_float[:,0]
sarray['No4']=ran_float[:,1]
print time.time()-time1
#0.123999834061 seconds

#Create table in H5
time1=time.time()
h5.create_table('/','int_floats_from_array',sarray,title='Integers and Floats',expectedrows=rows,filters=filters)
print time.time()-time1
#0.0769999027252 seconds
h5

#remove duplicate table
h5.remove_node('/','int_floats_from_array')

#Table object is similar to Pyton and NumpPy objects when it comes to slicing
tab[:3]

#Select one column
tab[:4]['No4']

#Perform numpy operations
time1=time.time()
np.sum(tab[:]['No3'])
print time.time()-time1
#0.0970001220703 seconds

time1=time.time()
np.sum(np.sqrt(tab[:]['No1']))
print time.time()-time1
#0.128000020981 seconds

time1=time.time()
plt.hist(tab[:]['No3'], bins=30)
plt.grid(True)
print len(tab[:]['No3'])
print time.time()-time1
#2000000
#0.31500005722 seconds

#Pandas like queries in PyTables
time1=time.time()
res=np.array([(row['No3'],row['No4']) for row in tab.where('((No3<-0.5) | (No3>0.5)) & ((No4<-1.0)|(No4>1.0))')])[::100]
print time.time()-time1
#0.508000135422 seconds
plt.plot(res.T[0],res.T[1],'ro')
plt.grid(True)

#Tables operations similar to numpy
time1=time.time()
values=tab.cols.No3[:]
values.max()
values.mean()
values.min()
values.std()
print time.time()-time1
#0.107000112534

#Working with compressed tables
filename=path+'tab.h5c'
h5c=tb.open_file(filename,'w')
filters = tb.Filters(complevel=4,complib='blosc')

tabc=h5c.create_table('/','ints_floats',sarray,title='Integers and Floats', expectedrows=rows, filters=filters)
time1=time.time()
res = np.array([(row['No3'],row['No4']) for row in tabc.where('((No3<-0.5) | (No3 > 0.5)) & ((No4<-1.0) | (No4>1.0))')])[::100]
print time.time()-time1
#0.512000083923 seconds

#reading compressed table into an array
time1=time.time()
arr_non=tab.read()
print time.time()-time1
#0.0870001316071 sec

time1=time.time()
arr_com=tabc.read()
print time.time()-time1
#0.154000043869 seconds

#While slower, takes up 80% less disk space

h5c.close()

#Creating PyTable arrays
arr_int=h5.create_array('/','integers',ran_int)
arr_float=h5.create_array('/','floats',ran_float)

h5

h5.close()

#Out of memory computations with PyTables
filename=path+'array.h5'
h5=tb.open_file(filename,'w')

#Create 1000 column row extendable array (EArray)
n=1000
ear=h5.createEArray(h5.root,'ear',atom=tb.Float64Atom(),shape=(0,n))

#populate chunkwise
time1=time.time()
rand=np.random.standard_normal((n,n))
for i in range(750):
    ear.append(rand)
ear.flush()
print time.time()-time1
#135.915999889 seconds

#Check size logcally and pyscially
ear
ear.size_on_disk

#Create target output array
out=h5.createEArray(h5.root, 'out',atom=tb.Float64Atom(),shape=(0,n))

#Create expresion ie y=3sin(x) +sqrt(abs(x))
expr=tb.Expr('3*sin(ear)+sqrt(abs(ear))')
expr.setOutput(out,append_mode=True)

time1=time.time()
expr.eval()
print time.time()-time1
#187.100000143 seconds

h5.close()


