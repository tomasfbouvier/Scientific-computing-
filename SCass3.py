# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


Xstart2= np.load("Xstart2.npy")
Xstart3= np.load("Xstart3.npy")
Xstart4= np.load("Xstart4.npy")
Xstart5= np.load("Xstart5.npy")
Xstart6= np.load("Xstart6.npy")
Xstart7= np.load("Xstart7.npy")
Xstart8= np.load("Xstart8.npy")
Xstart9= np.load("Xstart9.npy")

NA=np.newaxis

def dist(points):
	displacement = points[:,NA] - points[NA,:]
	distance=np.sqrt(np.sum(displacement*displacement , axis=-1))
	return(distance)

def LJ(sigma,epsilon, derivative):
	def V(points):
		distance=dist(points)
		if(derivative==0):
			potential=  4*epsilon*((sigma/distance)**(12)- (sigma/distance)**(6))
		elif(derivative==1):
			potential= - 4*epsilon*(12*(sigma**(12)/distance**(13))- 6*(sigma**(6)/distance**(7)))
		else:
			print("derivative degree too high")
			return
		E=0
		for i in range(np.shape(potential)[0]):
			for j in range( i+1, np.shape(potential)[1]):
				E+= potential[i,j]
		return(E)
	return(V)

def bisection_root(f,a,b,tolerance= 1e-13):
	n_calls=0
	m= a+(b-a)/2
	while(abs(f(m))>tolerance):
		m=a+(b-a)/2.;
		if(np.sign(f(a))== np.sign(f(m))):
			a=m
		else:
			b=m
		n_calls+=1
	return(m, n_calls)

def newton_root(f, df, x0, tolerance=1e-12, max_iterations=30):
	x=x0
	iter=0
	while(abs(f(x))>tolerance):
		x-=f(x)/df(x)
		iter+=1
		if(iter==max_iterations):
			break
	return(x, iter)
def super_root_finder(f,df, a,b, tolerance=1e-12):
	n_calls=0
	x= a+(b-a)/2
	while(abs(f(x))>tolerance):
		x-=f(x)/df(x)
		if(x<a or x>b):
			x=a+(b-a)/2.;
			if(np.sign(f(a))== np.sign(f(x))):
				a=x
			else:
				b=x
		n_calls+=1
	return(x, n_calls)

def LJgradient(sigma, epsilon):
	def gradV(X):
		d=X[:,NA] - X[NA,:]
		r= np.sqrt( np.sum( d*d, axis=-1))
		np.fill_diagonal(r, 1)
		T= 6*(sigma**6)*(r**(-7))- 12*(sigma**12)*(r**(-13))
		u=d/r[:,:,NA]
		return(4*epsilon*np.sum(T[:,:,NA]*u,axis=1))
	return gradV

def linesearch(F, X0, d, alpha_max, tolerance, max_iterations):
	def f(alpha):
		aux= d*F(X0+alpha*d)
		l=np.sum(aux)
		l2= np.sum(abs(aux))
		return(l)
	alpha, n_iteractions = bisection_root(f,0,alpha_max,tolerance= 1e-13)
	return(alpha, n_iteractions)

def golden_section_min(f,a,b,tolerance=1e-3):
	tau=(np.sqrt(5)-1)/2
	x1=a + (1-tau)*(b-a)
	f1=f(x1)
	x2=a+ tau*(b-a)
	f2=f(x2)
	while(abs(a-b)>tolerance):
		if(f1>f2):
			a=x1
			x1=x2
			f1=f2
			x2=a+tau*(b-a)
			f2=f(x2)
		else:
			b=x2
			x2=x1
			f2=f1
			x1=a+(1-tau)*(b-a)
			f1=f(x1)

	return(x2)

def BFGS(f, gradf, X0, linesearch, tolerance=1e-9, max_iterations=10000):
	X=np.array(np.copy(X0))
	f=flattenfunction(f); gradf=flattengradient(gradf)
	B= np.identity(len(X))*np.linalg.norm(gradf(X))/0.01
	N_calls=0
	converged=True
	y=300
	while(abs(np.linalg.norm(y))>tolerance):
		s= np.linalg.inv(B)@((-gradf(X)))
		alpha=1
		if(linesearch==True):
			def f1d(alpha):
				return(f(X+alpha*s))
			alpha= golden_section_min(f1d, -1., 1.)
		s*=alpha
		X_opt= np.copy(X);
		X+= s
		y=np.array([gradf(X)-gradf(X_opt)])
		B= B+ (np.outer(y,y)/np.dot(y,s) - np.outer(B@s, B@s)/np.dot(s, B@s))
		N_calls+=1
		if(N_calls==max_iterations):
			converged=False
			break
	return(X_opt, N_calls, converged)

def simulated_anealing(f, X0):
	alpha=1-1e-5
	gamma=0.5
	x=np.array(np.copy(X0))
	f=flattenfunction(f);
	T=0.1
	while(T>10**(-4)):
		deltax=np.random.rand(len(x))
		deltaE= gamma*(f(x+deltax)-f(x))
		if(deltaE>0):
			r=np.e**(-deltaE/T)
			p=np.random.rand()
			if(p<r):
				x+=deltax
		else:
			x+=deltax
		T*=alpha
	return(x)

def f0(x):
	x=[x,0,0]
	return(LJ0(np.array([x,x1])))
def df0(x):
	x=[x,0,0]
	return(LJ1(np.array([x,x1])))
def f1(x):
	x=[x,0,0]
	return(LJ0(np.array([x,x1,x2,x3])))

def grad0(x):
	x=[x,0,0]
	return(LJg(np.array([x,x1])))

def grad1(x):
	x=[x,0,0]
	return(LJg(np.array([x,x1,x2,x3])))

def flattenfunction(f):
	return lambda X: f(X.reshape(-1,3))

def flattengradient(f):
	return lambda X: f(X.reshape(-1,3)).reshape(-1)

def fg(alpha):
	return(LJ0(X0+alpha*d))


xarray=np.linspace(3,11, 1000)

x1=[0,0,0]
x2=[14,0,0]
x3=[7,3.2,0]
x4=[3.8,0,0]

LJ0=LJ(3.401, 0.997, 0)
LJ1=LJ(3.401, 0.997, 1)

print(bisection_root(f0,2,6))
print(newton_root(f0,df0, 2))
print(super_root_finder(f0,df0, 2,6))


result=[]
for i in range(len(xarray)):
	result.append(f0(xarray[i]))

plt.figure()
plt.plot(xarray, result)
plt.xlabel("d ($\AA$) ")
plt.ylabel("V (KJ/mol)")
plt.show()

LJg=LJgradient(3.401, 0.997)

print(LJg(np.array([x1,x2,x3,x4])))



result2=[]
result=[]
for i in range(len(xarray)):
	result.append(f0(xarray[i]))
	result2.append(LJg(np.array([[xarray[i],0,0],x1]))[0,0])


plt.figure()
plt.plot(xarray, result,label= "LJ" )


plt.plot(xarray, result2, label=" $  \\nabla LJ $ ")
plt.legend(loc="best", fontsize=12)

plt.xlabel("d ($\AA$) ")
plt.ylabel("V (KJ/mol)")
plt.show()

X0=np.array([[4,0,0],[0,0,0], [14,0,0], [7,3.2, 0]])
d=-LJg(X0)

alpha, n_iteractions =linesearch(LJg, X0, d, 1 , 10**(-12), 10000)
print("alpha",alpha)
print(" ", d*LJg(X0+alpha*d))

#WEEEK 2

alphag=golden_section_min(fg, 0, 1)
r0=golden_section_min(f0,1,9)
print("bisection method finding", r0)

X_opt,N_calls, converged= BFGS(LJ0, LJg, Xstart9, True)
X_opt= X_opt.reshape(-1,3)

D=dist(X_opt);

print(sum(abs(D-r0)/r0 <= 0.01), converged, N_calls)
print("D=", D)
print("minimum", LJ0(X_opt))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(np.shape(X_opt)[0]):
    xs = X_opt[i,0]
    ys = X_opt[i,1]
    zs = X_opt[i,2]
    ax.scatter(xs, ys, zs, c='b', s=50)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

size=5

X_opt =simulated_anealing(LJ0, np.random.rand(size*3))
X_optaux= X_opt.reshape(-1,3)

fantes=LJ0(X_optaux)

X_opt,N_calls, converged= BFGS(LJ0, LJg, X_opt, True)
X_opt= X_opt.reshape(-1,3)

fdespues=LJ0(X_opt)
D=dist(X_opt);

print("D=", D)
print("minimums", fantes, fdespues)
print(sum(abs(D-r0)/r0 <= 0.01), converged, N_calls)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(np.shape(X_opt)[0]):
    xs = X_opt[i,0]
    ys = X_opt[i,1]
    zs = X_opt[i,2]
    ax.scatter(xs, ys, zs, c='b', s=50)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

