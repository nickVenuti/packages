# -*- coding: utf-8 -*-
"""
Created on Wed Jan 06 15:49:40 2016

@author: nmvenuti
Chapter 9 Mathmatical Tools
Approximation Methods, Convex Optimization, 
Integration, Symbolic Computation
Python for Finance
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import statsmodels.api as sm
import scipy.interpolate as spi
import scipy.optimize as spo
from math import sqrt
import scipy.integrate as sci
from matplotlib.patches import Polygon
import sympy as sy

###############################
#####Approximation Methods#####
###############################

def f(x):
    return np.sin(x)+0.5*x

#Plot the function from -2pi to 2 pi
x=np.linspace(-2*np.pi,2*np.pi,50)
plt.plot(x,f(x),'b')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

#using polyfit and polyval from numpy to approximate the equation
reg=np.polyfit(x,f(x),deg=1)
ry=np.polyval(reg,x)

#Plot the predicted equation compared to the actual function
plt.plot(x,f(x),'b',label='f(x)')
plt.plot(x,ry,'r.',label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

#A lot left to be desired

#Show similar steps with degree of 5
reg=np.polyfit(x,f(x),deg=5)
ry=np.polyval(reg,x)

#Plot the predicted equation compared to the actual function
plt.plot(x,f(x),'b',label='f(x)')
plt.plot(x,ry,'r.',label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

#BEtter but still misses

#Show similar steps with degree of 7
reg=np.polyfit(x,f(x),deg=7)
ry=np.polyval(reg,x)

#Plot the predicted equation compared to the actual function
plt.plot(x,f(x),'b',label='f(x)')
plt.plot(x,ry,'r.',label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

#Pretty close

np.allclose(f(x),ry)
#False
#Not perfect

#Calculate MSE
np.sum((f(x)-ry)**2)/len(x)
#0.0017769134759517584

#Individual basis functions
#i.e. monomials up to order 3

matrix=np.zeros((3+1,len(x)))
matrix[3,:]=x**3
matrix[2,:]=x**2
matrix[1,:]=x
matrix[0,:]=1

#Use least squares optimization to find regression coefficients

reg=np.linalg.lstsq(matrix.T,f(x))[0]

reg
#array([  1.51371459e-14,   5.62777448e-01,  -1.11022302e-15, -5.43553615e-03])

#take dot product to get regression estimates

ry=np.dot(reg,matrix)

#Plot estimation based on regression
plt.plot(x,f(x),'b',label='f(x)')
plt.plot(x,ry,'r.',label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

#We know that there is a sin in the function, sqitch out column three of matrix

matrix[3,:]=np.sin(x)
reg=np.linalg.lstsq(matrix.T,f(x))[0]
reg=np.linalg.lstsq(matrix.T,f(x))[0]

reg
#array([  2.10690746e-16,   5.00000000e-01,   0.00000000e+00, 1.00000000e+00])
         
ry=np.dot(reg,matrix)

#Plot estimation based on regression
plt.plot(x,f(x),'b',label='f(x)')
plt.plot(x,ry,'r.',label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

#Verify closeness
np.allclose(f(x),ry)
#True

#Essentially zero mse
np.sum((f(x)-ry)**2)/len(x)
#2.1035777224575145e-31

#Applying similar techniques to noisy data
xn=np.linspace(-2*np.pi,2*np.pi,50)
xn=xn+0.15*np.random.standard_normal(len(xn))

yn=f(xn)+0.25*np.random.standard_normal(len(xn))

reg=np.polyfit(xn,yn,7)
ry=np.polyval(reg,xn)

plt.plot(xn,yn,'b^',label='f(x)')
plt.plot(xn,ry,'r.',label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

#Apply similar techniqques to unsorted data
xu=np.random.rand(50)*4*np.pi-2*np.pi
yu=f(xu)

reg=np.polyfit(xu,yu,7)
ry=np.polyval(reg,xu)

plt.plot(xu,yu,'b^',label='f(x)')
plt.plot(xu,ry,'r.',label='regression')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')


#Multi-dimensions
def fm((x,y)):
    return np.sin(x)+0.25*x+np.sqrt(y)+0.05*y**2

x=np.linspace(0,10,20)
y=np.linspace(0,10,20)

#generate 2-d grid out of 1-d array
X,Y=np.meshgrid(x,y)

Z=fm((X,Y))

#yields 1-d arrays from 2-d grids
x=X.flatten()
y=Y.flatten()

#Plot data
fig=plt.figure(figsize=(9,6))
ax=fig.gca(projection='3d')
surf=ax.plot_surface(X,Y,Z,rstride=2,cstride=2,cmap=mpl.cm.coolwarm,linewidth=0.5,antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
fig.colorbar(surf,shrink=0.5,aspect=5)

#Leverging knowledge of basic function, perform LSE
matrix=np.zeros((len(x),6+1))
matrix[:,6]=np.sqrt(y)
matrix[:,5]=np.sin(x)
matrix[:,4]=y**2
matrix[:,3]=x**2
matrix[:,2]=y
matrix[:,1]=x
matrix[:,0]=1

#Using OLS function in statsmodels.api
model=sm.OLS(fm((x,y)),matrix).fit()

#Get r squared and model parameters
model.rsquared
#1.0
a=model.params
a
#array([  4.60742555e-15,   2.50000000e-01,   8.40086094e-16,
#        -1.52655666e-16,   5.00000000e-02,   1.00000000e+00,
#         1.00000000e+00])

#Create reg_func to create estimate from modeling
def ref_func(a,(x,y)):
    f6=a[6]*np.sqrt(y)
    f5=a[5]*np.sin(x)
    f4=a[4]*y**2
    f3=a[3]*x**2
    f2=a[2]*y
    f1=a[1]*x
    f0=a[0]
    return(f6+f5+f4+f3+f2+f1+f0)

RZ=ref_func(a,(X,Y))

fig=plt.figure(figsize=(9,6))
ax=fig.gca(projection='3d')
surf1=ax.plot_surface(X,Y,Z,rstride=2,cstride=2,cmap=mpl.cm.coolwarm,linewidth=0.5,antialiased=True)
surf2=ax.plot_wireframe(X,Y,RZ,rstride=2,cstride=2,label='regression')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
ax.legend()
fig.colorbar(surf,shrink=0.5,aspect=5)

#Interpolation
x=np.linspace(-2*np.pi,2*np.pi,25)
def f(x):
    return np.sin(x)+0.5*x

#Apply spline function
ipo=spi.splrep(x,f(x),k=1)
iy=spi.splev(x,ipo)

#Plot output of spline function compared to raw data
plt.plot(x,f(x),'b',label='f(x)')
plt.plot(x,iy,'r.',label='interpolation')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

#Checkcloseness
np.allclose(iy,f(x))
#True

#Check interpolated values within smaller interval

xd=np.linspace(1.,3.,50)
iyd=spi.splev(xd,ipo)

#Plot smaller interval
plt.plot(xd,f(xd),'b',label='f(x)')
plt.plot(xd,iyd,'r.',label='interpolation')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

#Evident that the function is not continuously differentiable

#Attempt similar exercise with cubic spline for smaller interval
ipo=spi.splrep(x,f(x),k=3)
iyd=spi.splev(xd,ipo)

#Plot output of spline function compared to raw data
plt.plot(xd,f(xd),'b',label='f(x)')
plt.plot(xd,iyd,'r.',label='interpolation')
plt.legend(loc=0)
plt.grid(True)
plt.xlabel('x')
plt.ylabel('f(x)')

#Checkcloseness
np.allclose(iyd,f(xd))
#False

#MSE is really small though
sum((iyd-f(xd))**2)/len(xd)
#1.1349319851436255e-08

##############################
#####Convext Optimization#####
##############################

#Create dataset in which we want to find global and local minimums
def fm((x,y)):
    return(np.sin(x)+0.05*x**2+np.sin(y)+0.05*y**2)

x=np.linspace(-10,10,50)
y=np.linspace(-10,10,50)
X,Y=np.meshgrid(x,y)
Z=fm((X,Y))

#Plot dataset
fig=plt.figure(figsize=(9,6))
ax=fig.gca(projection='3d')
surf=ax.plot_surface(X,Y,Z, rstride=2, cstride=2, cmap=mpl.cm.coolwarm,linewidth=0.5,antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
fig.colorbar(surf,shrink=0.5,aspect=5)

#Find global optimum
#Update original data function to extract x,y,z
def fo((x,y)):
    z=np.sin(x)+0.05*x**2+np.sin(y)+0.05*y**2
    if output==True:
        print '%8.4f %8.4f %8.4f' % (x,y,z)
    return z
output=True

#Brute force optimization by 5 steps in x and y to optimize z
spo.brute(fo,((-10,10.1,5),(-10,10.1,5)),finish=None)

#array([ 0.,  0.])

#Use 0.1 steps
output=False
opt1=spo.brute(fo,((-10,10.1,0.1),(-10,10.1,0.1)),finish=None)
opt1
#array([-1.4, -1.4])

fm(opt1)
#-1.7748994599769203

#Local optimzation
output2=True
opt2=spo.fmin(fo,opt1,xtol=0.001,ftol=0.001,maxiter=15,maxfun=20)
opt2
#array([-1.42702972, -1.42876755])
fm(opt2)
#-1.7757246992239009

#Best to find the relative global optimum before performing the local optimization to find tighter constraint. Otherwise may not
#find best result due to 'basin hoping'
#example

output=False
opt3=spo.fmin(fo,(2.0,2.0),maxiter=250)
#Optimization terminated successfully.
#         Current function value: 0.015826
#         Iterations: 46
#         Function evaluations: 86
opt3
#array([ 4.2710728 ,  4.27106945])

fm(opt3)
#0.015825753274680499

#Much better results shown prior methodology

#Constrained optimization

#Example function to be minimized
def Eu((s,b)):
    return -(0.5*sqrt(s*15+b*5)+0.5*sqrt(s*5+b*12))

#Set function constraints
cons=({'type':'ineq','fun':lambda(s,b):100-s*10-b*10})

#budget constraint
bnds=((0,1000),(0,1000))

#Use minimize function with inital gues of  5 and 5
result=spo.minimize(Eu,[5,5],method='SLSQP',bounds=bnds,constraints=cons)

result
#  status: 0
# success: True
#    njev: 5
#    nfev: 21
#     fun: -9.700883611487832
#       x: array([ 8.02547122,  1.97452878])
# message: 'Optimization terminated successfully.'
#     jac: array([-0.48508096, -0.48489535,  0.        ])
#     nit: 5

#Optimal parameters
result['x']
#array([ 8.02547122,  1.97452878])

#Optimal value (note negated since we wanted to find max value, but used min function)
-result['fun']
#9.700883611487832

#Confirm constraints is binding (ie the investor invests all of the $100)
np.dot(result['x'],[10,10])
#99.999999999999986


#####################
#####Integration#####
#####################

#Create test function
def f(x):
    return np.sin(x)+0.5*x

#Going to integrate the function from 0.5 to 9.5
####First graph the function
#Define space of function
a=0.5
b=9.5
x=np.linspace(0,10)
y=f(x)

#Plot line
fig,ax=plt.subplots(figsize=(7,5))
plt.plot(x,y,'b',linewidth=2)
plt.ylim(ymin=0)

#area under the function
#between lower and upper limit
#Adds shading
Ix=np.linspace(a,b)
Iy=f(Ix)
verts=[(a,0)]+list(zip(Ix,Iy))+[(b,0)]
poly=Polygon(verts,facecolor='0.7',edgecolor='0.5')
ax.add_patch(poly)

#Add labels
plt.text(0.75*(a+b),1.5,r"$\int_a^b f(x)dx$",horizontalalignment='center',fontsize=20)
plt.figtext(0.9,0.075,'$x$')
plt.figtext(0.075,0.9,'$f(x)$')
ax.set_xticks((a,b))
ax.set_yticks([f(a),f(b)])

#Numerical integration
sci.fixed_quad(f,a,b)[0] #fixed Gaussian quadrature
#24.366995967084602
sci.quad(f,a,b)[0] #adaptive quadrature
#24.374754718086752
sci.romberg(f,a,b)#Romberg integration
#24.374754718086713

xi=np.linspace(0.5,9.5,25)

sci.trapz(f(xi),xi) #trapezodial rule
#24.352733271544516
sci.simps(f(xi),xi) #Simpsons rule
#24.374964184550748

#Integration by simulation using monte carlo methods
for i in range(1,20):
    np.random.seed(1000)
    x=np.random.random(i*10)*(b-a)+a
    print np.sum(f(x))/len(x)*(b-a)

#24.8047622793
#26.5229188983
#26.2655475192
#26.0277033994
#24.9995418144
#23.8818101416
#23.5279122748
#23.507857659
#23.6723674607
#23.6794104161
#24.4244017079
#24.2390053468
#24.115396925
#24.4241919876
#23.9249330805
#24.1948421203
#24.1173483782
#24.1006909297
#23.7690510985  

##############################
#####Symbolic Computation#####
##############################

#Symbol class
x=sy.Symbol('x')
y=sy.Symbol('y')

type(x)
#sympy.core.symbol.Symbol

#Sympy has a large amount of preprogrammed mathematical fucntion defintions
sy.sqrt(x)
#sqrt(x)

#Sympy will automatically simplify math functions
3+sy.sqrt(x)-4**2
#sqrt(x) - 13

#Define arbiraty functions using symbols
f=x**2+3+0.5*x**2+3/2

sy.simplify(f)
#1.5*x**2 + 4

sy.init_printing(pretty_print=False, use_unicode=False)

print sy.pretty(f)
#     2    
#1.5*x  + 4

print sy.pretty(sy.sqrt(x)+0.5)
#  ___      
#\/ x  + 0.5

#Solving equations

#Example solve x^2-1=0
sy.solve(x**2-1)
#[1,1]

sy.solve(x**2-1-3)
#[2,2]

sy.solve(x**3+0.5*x**2-1)
#[0.858094329496553, -0.679047164748276 - 0.839206763026694*I, -0.679047164748276 + 0.839206763026694*I]

sy.solve(x**2+y**2)
#[{x: -I*y}, {x: I*y}]

#Integration
a,b=sy.symbols('a b')
print sy.pretty(sy.Integral(sy.sin(x)+0.5*x,(x,a,b)))
#  b                    
#  /                    
# |                     
# |  (0.5*x + sin(x)) dx
# |                     
#/                      
#a  

int_func=sy.integrate(sy.sin(x)+0.5*x,x)
print sy.pretty(int_func)
#      2         
#0.25*x  - cos(x)

#Evaluate integral from 9.5 to 0.5
Fb=int_func.subs(x,9.5).evalf()
Fa=int_func.subs(x,0.5).evalf()
Fb-Fa
#24.3747547180867

#Symbolic integration
int_func_limits=sy.integrate(sy.sin(x)+0.5*x,(x,a,b))
print sy.pretty(int_func_limits)
#           2         2                  
#    -0.25*a  + 0.25*b  + cos(a) - cos(b)

int_func_limits.subs({a:0.5,b:9.5}).evalf()
#24.3747547180868

#Quantified integration
sy.integrate(sy.sin(x)+0.5*x,(x,0.5,9.5))
#24.3747547180868

#Differentiation

#Use it to check integration
int_func.diff()
#0.5*x + sin(x)

#Use differentionation to solve convex minimization problem from earlier
f=(sy.sin(x)+.05*x**2+sy.sin(y)+.05*y**2)

#Get partial derivatives in terms of x and y

del_x=sy.diff(f,x)
del_x
#0.1*x + cos(x)

del_y=sy.diff(f,y)
del_y
#0.1*y + cos(y)

#using educated guess we can achieve results similar to global minimum,local minimum method
x0=sy.nsolve(del_x,-1.5)
y0=sy.nsolve(del_y,-1.5)

print x0, y0
#-1.42755177876459 -1.42755177876459
f.subs({x:x0,y:y0}).evalf()
#-1.77572565314742

#Without having the best guess estimates may prove inaccruate due to only finding a local minimum
x0=sy.nsolve(del_x,1.5)
y0=sy.nsolve(del_y,1.5)

print x0, y0
#1.74632928225285 1.74632928225285
f.subs({x:x0,y:y0}).evalf()
#2.27423381055640