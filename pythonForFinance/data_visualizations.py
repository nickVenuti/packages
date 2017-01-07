# -*- coding: utf-8 -*-
"""
Created on Mon Jan 04 08:03:15 2016

@author: nmvenuti
"""

'''
Data Visualization
Python for Finance
'''

import numpy as np
import matplotlib as mlb
import matplotlib.pyplot as plt

#Generate random numbers to plot
np.random.seed(1000)
y=np.random.standard_normal(20)

#plot the values
x=range(len(y))

plt.plot(x,y)
plt.plot(y)

#graphs are the same since Python infers x values from index values if x is not provided

#Can pass a method directly to graphic
plt.plot(y.cumsum())
plt.grid(True)
plt.axis('tight')

#Can also set x and y axis limits manually
plt.plot(y.cumsum())
plt.grid(True)
plt.xlim(-1,20)
plt.ylim(np.min(y.cumsum())-1,np.max(y.cumsum())+1)

#Adding lables and additional stylings
plt.figure(figsize=(7,4))
plt.plot(y.cumsum(),'b',lw=1.5)
plt.plot(y.cumsum(),'ro')
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')

#2 Dimensions Plotting
np.random.seed(2000)
y=np.random.standard_normal((20,2)).cumsum(axis=0)

#2-D data sets will plot the same as 1-D data sets if they are within same data structure
plt.figure(figsize=(7,4))
plt.plot(y,'b',lw=1.5)
plt.plot(y,'ro')
plt.grid(True)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')

#Adding data labels
#2-D data sets will plot the same as 1-D data sets if they are within same data structure
plt.figure(figsize=(7,4))
plt.plot(y[:,0],lw=1.5,label='1st')
plt.plot(y[:,1],lw=1.5,label='2nd')
plt.plot(y,'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')

#Illistrate issues with scaling
y[:,0]=100*y[:,0]
plt.figure(figsize=(7,4))
plt.plot(y[:,0],lw=1.5,label='1st')
plt.plot(y[:,1],lw=1.5,label='2nd')
plt.plot(y,'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value')
plt.title('A Simple Plot')

#Add two axis
fig, ax1=plt.subplots()
plt.plot(y[:,0],'b',lw=1.5,label='1st')
plt.plot(y[:,0],'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.ylabel('value 1st')
plt.title('A Simple Plot')
ax2=ax1.twinx()
plt.plot(y[:,1],'g',lw=1.5,label='2nd')
plt.plot(y[:,1],'ro')
plt.legend(loc=0)
plt.ylabel('value 2nd')

#Plot side by side 
plt.figure(figsize=(7,5))
plt.subplot(211)
plt.plot(y[:,0],lw=1.5,label='1st')
plt.plot(y[:,0],'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('value')
plt.title('A Simple Plot')
plt.subplot(212)
plt.plot(y[:,1],'g',lw=1.5,label='1st')
plt.plot(y[:,1],'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('value')

#Plotting differening Plot Types
plt.figure(figsize=(7,5))
plt.subplot(121)
plt.plot(y[:,0],lw=1.5,label='1st')
plt.plot(y[:,0],'ro')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.ylabel('value')
plt.title('1st Data Set')
plt.subplot(122)
plt.bar(np.arange(len(y)),y[:,1],width=0.5,color='g',label='2nd')
plt.grid(True)
plt.legend(loc=0)
plt.axis('tight')
plt.xlabel('index')
plt.title('2nd Data Set')

#Scatter plots
y=np.random.standard_normal((1000,2))

plt.figure(figsize=(7,5))
plt.plot(y[:,0],y[:,1],'ro')
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')

#Using the matplotlib catter function
plt.figure(figsize=(7,5))
plt.scatter(y[:,0],y[:,1],marker='o')
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')

#Add third dimension to data and plot with scatter
c=np.random.randint(0,10,len(y))
plt.figure(figsize=(7,5))
plt.scatter(y[:,0],y[:,1],c=c,marker='o')
plt.colorbar()
plt.grid(True)
plt.xlabel('1st')
plt.ylabel('2nd')
plt.title('Scatter Plot')

#Plotting histograms
plt.figure(figsize=(7,5))
plt.hist(y,label=['1st','2nd'],bins=25)
plt.grid(True)
plt.legend(loc=0)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('histogram')

#Stacked histograms
plt.figure(figsize=(7,5))
plt.hist(y,label=['1st','2nd'],color=['b','g'],stacked=True,bins=20)
plt.grid(True)
plt.legend(loc=0)
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('histogram')

#Boxplots
fig,ax=plt.subplots(figsize=(7,4))
plt.boxplot(y)
plt.grid(True)
plt.setp(ax,xticklabels=['1st','2nd'])
plt.xlabel('data set')
plt.ylabel('value')
plt.title('Boxplot')

#Creating LaTeX like plots
from matplotlib.patches import Polygon
def func(x):
    return 0.5*np.exp(x)+1

#set integral limits
a,b=0.5,1.5

x=np.linspace(0,2)
y=func(x)

fig,ax=plt.subplots(figsize=(7,5))
plt.plot(x,y,'b',linewidth=2)
plt.ylim(ymin=0)

#Illustrate the integral value, i.e. the areas under the function
#Between the lower and upper limits
Ix=np.linspace(a,b)
Iy=func(Ix)
verts=[(a,0)]+list(zip(Ix,Iy))+[(b,0)]
poly=Polygon(verts,facecolor='0.7',edgecolor='0.5')
ax.add_patch(poly)

plt.text(0.5*(a+b),1,r"$\int_a^b f(x)\mathrm{d}x$",horizontalalignment='center',fontsize=20)
plt.figtext(0.9,0.075,'$x$')
plt.figtext(0.075,0.9,'$f(x)$')
ax.set_xticks((a,b))
ax.set_xticklabels(('$a$','$b$'))
ax.set_yticks([func(a),func(b)])
ax.set_yticklabels(('$f(a)$','$f(b)$'))
plt.grid(True)

#Plotting financial plots
import matplotlib.finance as mpf

start=(2014,5,1)
end=(2014,6,30)

#Get historic prices of German DAX index ^GDAXI
quotes=mpf.quotes_historical_yahoo_ochl('^GDAXI',start,end)

#Data comes out as Open, High,Low,Close,and Volume
quotes[:2]

#Create candlestick plots
fig,ax=plt.subplots(figsize=(7,5))
fig.subplots_adjust(bottom=0.2)
mpf.candlestick_ochl(ax,quotes,width=0.6,colorup='b',colordown='r')
plt.grid(True)
ax.xaxis_date()
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(),rotation=30)

#Create daily summary plots
fig,ax=plt.subplots(figsize=(7,5))
fig.subplots_adjust(bottom=0.2)
mpf.plot_day_summary_oclh(ax,quotes,colorup='b',colordown='r')
plt.grid(True)
ax.xaxis_date()
plt.title('DAX Index')
plt.ylabel('index level')
plt.setp(plt.gca().get_xticklabels(),rotation=30)

#Plotting candlestick and barcharts
quotes=np.array(mpf.quotes_historical_yahoo_ochl('YHOO',start,end))
fig,(ax1,ax2)=plt.subplots(2,sharex=True,figsize=(8,6))
mpf.candlestick_ochl(ax1,quotes,width=0.6,colorup='b',colordown='r')
ax1.set_title('Yahoo Inc.')
ax1.grid(True)
ax1.xaxis_date()
plt.bar(quotes[:,0]-0.25,quotes[:,5],width=0.5)
ax2.set_ylabel('volume')
ax2.grid(True)
ax2.autoscale_view()
plt.setp(plt.gca().get_xticklabels(),rotation=30)

#3d financial plots

#Generate fake strike prices and times-to-maturity
strike=np.linspace(50,150,24)
ttm=np.linspace(0.5,25,24)
strike,ttm=np.meshgrid(strike,ttm)

#Generate fake implied volatities
iv=(strike-100)**2/(100*strike)/ttm

#generate 3D plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(9,6))
ax=fig.gca(projection='3d')
surf=ax.plot_surface(strike,ttm,iv,rstride=2,cstride=2,cmap=plt.cm.coolwarm,linewidth=0.5,antialiased=True)
ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility')

fig.colorbar(surf,shrink=0.5,aspect=5)


fig = plt.figure(figsize=(9,6))
ax=fig.add_subplot(111,projection='3d')
ax.view_init(30,60)
ax.scatter(strike,ttm,iv,zdir='z',s=25,c='b',marker='^')
ax.set_xlabel('strike')
ax.set_ylabel('time-to-maturity')
ax.set_zlabel('implied volatility')