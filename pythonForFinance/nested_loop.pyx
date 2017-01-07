# -*- coding: utf-8 -*-
"""
Created on Wed Jan 06 09:30:12 2016

@author: nmvenuti
"""

#################
#####CPython#####
#################

#Basic python
def f_py(I,J):
    res=0.
    for i in range(I):
        for j in range(J*I):
            res+=1
    return res

I,J=500,500
timer=time.time()
print f_py(I,J)
#125000000.0
print time.time()-timer
#8.20900011063 seconds

#Numpy vectorizaton will quickly not be feasible due to memory constraints
import pyximport

#Cpython method
def f_cy(int I, int J):
    cdef double res =0
    for i in range(I):
        for j in range(I*J):
            res +=1
    return res