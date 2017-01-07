# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 10:18:27 2015

@author: nmvenuti
"""

'''
Numpy Tutorial/Walkthrough
Python For Finance
'''
import numpy as np

#Creating an array
a=np.array([0,0.5,1,1.5,2])

#check type of a
type(a)

#index 2nd item through fourth item
a[1:4]

#Built in numpy functions

a.sum()

a.std()

a.cumsum()

#vectorization in np array
a*2

a**2

np.sqrt(a)

#Creation of additional dimensions
a
b= np.array([a,a**2])
b
#first row
b[0]

#third element of the first row
b[0,2]

#axis based operations
b.sum(axis=0) #sums column wise

b.sum(axis=1) #sums row wise

#Creation of zero array (functions also work with ones)
c=np.zeros((2,3,4),dtype='i',order='c')
c

#Can also use a like function to create zero or one arrays with same size
d=np.zeros_like(b,dtype='float',order='c')
d

#Note order is either C (c-llike) which is row wise, or F (Fortran like) which is column wise

#Get shape of array
np.shape(d)

#Generate random numbers using base python and calculate sum of all elements
import random
import time

I=5000
start = time.time()

#Create matrix
mat =[[random.gauss(0,1) for j in range(I)] for i in range(I)]

#get sum of all elements in matrix
reduce(lambda x,y: x+y, [reduce(lambda x,y: x+y, row) for row in mat])
end = time.time()
print(end - start)
#23.6640000343

#generate ranom numbers using numpy and calculate sum of all elements
start = time.time()

#Create matrix
mat = np.random.standard_normal((I,I))

#get sum of all elements in matrix
mat.sum()

end = time.time()
print(end - start)
#1.10799980164


#Creating a structured array
dt=np.dtype([('Name','S10'),('Age','i4'),('Height','f'),('Chidren/pets','i4',2)])
s=np.array([('Smith',45,1.83,[0,1]),('Jones',53,1.72,(2,2))],dtype=dt)

s
#array([('Smith', 45, 1.8300000429153442, [0, 1]),
#       ('Jones', 53, 1.7200000286102295, [2, 2])], 
#      dtype=[('Name', 'S10'), ('Age', '<i4'), ('Height', '<f4'), ('Chidren/pets', '<i4', (2,))])

#Can now access columns by name
print s['Name']
print s['Age'][s['Name']=='Smith']