# !/usr/bin/python
# -*- coding: utf-8 -*-

from watermatrices import Amat, Bmat, yvec
import numpy as np
import matplotlib.pyplot as plt

def max_norm(M):
	size=np.array(np.shape(M))
	norms=[]
	for i in range(size[1]):
		norms.append(sum(abs(M[:,i])))
	return(max(norms))

def cond_numb(M):
	invM=np.linalg.inv(M)
	return(max_norm(M)*max_norm(invM))

def lu_factorize(M): #c)
	U=M
	L=np.zeros([np.shape(np.array(M))[0],np.shape(np.array(M))[1]])

	for j in range(np.shape(np.array(M))[1]):
		for i in range(j+1, np.shape(np.array(M))[0]):
			m=(U[i,j]/U[j,j])
			U[i,:]-=U[j,:]*m
			L[i,j]=m
	L+=np.identity(np.shape(np.array(L))[0])
	return (L, U)

def forward_subst(L,z): #c)
	y=np.zeros(len(z))
	for i in range(len(z)):
		y[i]= z[i]-sum(L[i,:]*y[:])
	return (y)

def back_substitute(U,y): #c)
	x=np.zeros(len(y))
	for i in range(len(y)-1,-1,-1):
		x[i]= (y[i]-sum(U[i,:]*x[:]))/U[i,i]
	return (x)

def solve_alpha(omega):
	L,U=lu_factorize(E-omega*S)
	y=forward_subst(L,z)
	x=back_substitute(U,y)
	alpha=sum(z[:]*x[:])
	return (alpha)

def householder_QR(A): #f)
	size=np.shape(A)
	I=np.identity(size[0])
	RO=np.copy(A)
	Q=I
	for i in range (size[1]):
		a=RO[:,i]; e=np.zeros(len(a)); e[i]=1
		alpha= -(np.sign(a[i]))*np.linalg.norm(a[i:])
		v= np.array([np.append((np.zeros(i)),(a[i:]))[:]  -alpha*e[:]])
		H= (I-2*(v.T.dot(v))/(v.dot(v.T)))
		RO= H.dot(RO)
		Q= H.dot(Q)
	return(Q.T,RO) # If I had the possibility I would return transformed b in order to avoid computing Q and therefore affording time

def householder_fast(A): #f)
	size=np.shape(A)
	I=np.identity(size[0])
	R=np.copy(A)
	VR=np.zeros([size[0]+1,size[1]])
	for i in range (size[1]):
		a=R[:,i]; e=np.zeros(len(a)); e[i]=1
		alpha= -(np.sign(a[i]))*np.linalg.norm(a[i:])
		v= np.array([np.append((np.zeros(i)),(a[i:]))[:]  -alpha*e[:]])
		H= (I-2*(v.T.dot(v))/(v.dot(v.T)))
		R= H.dot(R)
		for j in range(i, size[0]):
			VR[j+1,i]=v[0,j]
			VR[j,j:]=R[j,j:]
	return(VR)

def least_squares(A,b): #f)
	Q,RO= householder_QR(A)
	b= Q.T.dot(b)
	R=RO[:np.shape(RO)[1], :np.shape(RO)[1]]
	c1= b[:np.shape(RO)[1]]
	x= back_substitute(R,c1)
	return(x)

# Definition of variables
E=(np.hstack((np.vstack((Amat,Bmat)),np.vstack((Bmat,Amat)))))
I=np.identity(np.shape(np.array(Amat))[0])
O=np.zeros([np.shape(np.array(Amat))[0],np.shape(np.array(Amat))[1]])
S=(np.hstack((np.vstack((I,O)),np.vstack((O,-I)))))
z=np.hstack((yvec,-yvec))
omega=[0.800,1.146,1.400]
delta_omega=1./2.*10**(-3)

# Calculation of condition numbers and bounds for a) and b)
n=[]
bound=[]
for w in omega:
	n.append(cond_numb(E-w*S))
	bound.append(cond_numb(E-w*S)*max_norm(delta_omega*S)/max_norm(E-w*S))

print(n)
print(np.log10(n)-3)
print("bound= ",bound)

# Solving the equation for the cases in d)
alphas=[]
deltalphas=[]
deltomega=1./2.*10**(-3)
for i in range(len(omega)):
	alphas.append(solve_alpha(omega[i]))
	deltalphas.append(solve_alpha(omega[i]+deltomega)-solve_alpha(omega[i]-deltomega))

# Computation of the table in e)
omega=np.linspace(0.7,1.5,1000)
alphas=[]
for i in range(len(omega)):
        alphas.append(solve_alpha(omega[i]))

table=np.hstack((np.array([omega]).T, np.array([alphas]).T))

np.savetxt("table.txt ", table)

plt.figure()
plt.plot(omega, alphas)
plt.xlabel(" $ \omega $ ")
plt.ylabel(" $\\alpha (\omega) $ ")
plt.show()

# Definition of test samples in f)

Atest=np.array([[1,0,0],[0,1,0], [0,0,1], [-1,1,0], [-1,0,1], [0,-1,1]])
btest=np.array([1237,1941, 2417, 711,1177,475])
B=Atest.copy()

# g)

n=4  #The order of the polynomial can be changed here (n=4 & 6 are shown in the report)

omega2=[]
alpha2=[]
i=0
aux=0
while(aux<1.1):
	aux=omega[i]
	omega2.append(omega[i])
	alpha2.append(alphas[i])
	i+=1

omega2=np.array(omega2)
A=np.zeros([len(omega2), n])
for i in range(len(omega2)):
	for j in range(n):
		A[i, j]=omega[i]**(2*j)

P=list(least_squares(A,alpha2))
P2=[]
for i in range(len(P)-1, -1, -1):
	P2.append(0.)
	P2.append(P[i])

plt.figure()
plt.plot(omega2, np.polyval(P2,omega2),'b--' ,markersize=3, label="fit")
plt.plot(omega2, alpha2, 'r.', markersize=2, label="data")
plt.xlabel("$ \omega $")
plt.ylabel("$\\alpha (\omega) $ ")

plt.legend(fontsize=12, loc="best")

plt.figure()
plt.plot(omega2, abs((alpha2-np.polyval(P2,omega2))/alpha2))
plt.xlabel("$ \omega $")
plt.ylabel("rel. error")
plt.yscale('log')
plt.show()

# h)

n=2 # The order of the polynomial can be changed here (n=2 & 4 are shown in the report)

A=np.zeros([len(omega), (n+1)+n])

for i in range(len(omega)):
	A[i,0]=1.
	for j in range(1,n+1):
		A[i, j]=omega[i]**j
		A[i,n+j]=-(omega[i]**j)*alphas[i]


P= list(least_squares(A,alphas))
P2=[]
P3=[]

for i in range(n, -1, -1):
	P2.append(P[i])
	if(i!=0):
		P3.append(P[n+i])
P3.append(0.)

plt.figure()
plt.plot(omega, np.polyval(P2,omega)/(1+np.polyval(P3, omega)),'b-' ,markersize=3, label="fit")
plt.plot(omega, alphas, 'r.', markersize=2, label= "data ")
plt.xlabel(" $ \omega  $ ")
plt.ylabel("$ \\alpha (\omega) $")
plt.legend(fontsize=12, loc="best")

plt.figure()
plt.plot(omega, abs(alphas-np.polyval(P2,omega)/(1+np.polyval(P3, omega))))
plt.yscale('log')
plt.xlabel("$ \omega $ ")
plt.ylabel(" rel. error ")
plt.show()
