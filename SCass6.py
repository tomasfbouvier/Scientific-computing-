# -*- coding: utf-8  -*-


import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from mpl_toolkits.mplot3d import Axes3D

Dp=1. ; Dq= 8. ; C= 4.5 ; K=7.

dt=0.01; dx2=1.; dy2=1.;

p= np.zeros([42,42])
q= np.zeros([42,42])
#pnew= np.zeros([42,42])
#qnew= np.zeros([42,42])

for i in range(11,31):
	for j in range(11,31):
		p[i,j]= C+0.1
		q[i,j]= K/C+0.2

#p[1,5]=123
#p[2,5]=124

size=np.shape(p)[0]
t=0

x=np.arange(0, size,1)
y= np.arange(0, size, 1)

x,y= np.meshgrid(x,y)


fig=plt.figure()



ax = fig.gca(projection='3d')

surf = ax.plot_surface(x, y, q , cmap=cm.coolwarm, linewidth=0, antialiased=False);  plt.pause(.01); plt.clf()

p=np.ravel(p); q=np.ravel(q)

N=size

print(p[N+5])
print(p[2*N+5])

print("N", N)

M=np.zeros([len(p),len(p)])
M1=np.zeros([len(p),len(p)])
M2=np.zeros([len(p),len(p)])
Mbound=np.identity(np.shape(M)[0])


for i in range(1,N-1):
	for j in range(1,N-1):
		#Laplaciano de X:		
		M[N*(i)+j, N*(i-1)+j]+= 1.*dt/dx2
		M[N*(i)+j, N*(i)+j]+= -2.*dt/dx2
		M[N*(i)+j, N*(i+1)+j]+= 1.*dt/dx2
		#Laplaciano de Y:
		M[N*(i)+j, N*(i)+j-1]+= 1.*dt/dy2
		M[N*(i)+j, N*(i)+j]+= -2.*dt/dy2
		M[N*(i)+j, N*(i)+j+1]+= 1.*dt/dy2	

M[0,0]=1.; M[N-1,N-1]=1.
M1=Dp*M + (-(K+1)*dt+1)*np.identity(np.shape(M)[0])

#Boundaries

for j in range(0,N):
	Mbound[N*j, N*j+1]=1.; Mbound[N*j,N*j]=0.
	Mbound[N*j+N-1, N*j+N-2]= 1.; Mbound[N*j+N-1, N*j+N-1]= 0.
	Mbound[j, N+j]=1.;Mbound[j,j]=0.
	Mbound[N*(N-1)+j,N*(N-2)+j]=1.; Mbound[N*(N-1)+j,N*(N-1)+j]=0.

M2=Dq*M+np.identity(np.shape(M)[0])

M2=Mbound@M2
M1=Mbound@M1

Cdt=Mbound@(C*dt*np.ones([len(p)]))
Kdt=Mbound*K*dt

def timestep(p,q):
	p2q=dt*Mbound@(p*p*q)

	pnew=M1@p +p2q+Cdt
	qnew=M2@q-p2q+Kdt@p

	return(pnew,qnew)

#evolucion

cont=0
ax.set_zlim(0., 6.)
	
while t<200:
	cont+=1
	t+=dt
	
#	print(t)
	ax = fig.gca(projection='3d')

	p,q= timestep(p,q)

	if(cont%10==0):
		print(t)
		ax.set_zlim(0., 6.)
		qmatr=np.reshape(q, [N,N])
		surf = ax.plot_surface(x, y, qmatr , cmap=cm.coolwarm, linewidth=0, antialiased=False);  plt.pause(.01)
		plt.clf()

plt.show()


pmatr=np.reshape(p, [N,N])
qmatr=np.reshape(q, [N,N])



plt.figure()
plt.pcolor(pmatr[1:N-1, 1:N-1], cmap='RdBu')
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

plt.figure()
plt.pcolor(qmatr[1:N-1, 1:N-1], cmap='RdBu')
plt.colorbar()
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

