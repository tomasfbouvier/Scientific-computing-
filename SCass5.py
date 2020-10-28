# -*- coding: utf-8 Project 4 part 2This project concerns reaction-diffusion equations and the approach -*-


import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from mpl_toolkits.mplot3d import Axes3D

Dp=1. ; Dq= 8. ; C= 4.5 ; K=7.

dt=0.01; dx=1.; dy=1.;

p= np.zeros([42,42])
q= np.zeros([42,42])
pnew= np.zeros([42,42])
qnew= np.zeros([42,42])

for i in range(11,31):
	for j in range(11,31):
		p[i,j]= C+0.1
		q[i,j]= K/C+0.2


size=np.shape(p)
t=0




x=np.arange(0, size[0],1)
y= np.arange(0, size[0], 1)

x,y= np.meshgrid(x,y)


fig=plt.figure()



ax = fig.gca(projection='3d')

surf = ax.plot_surface(x, y, q , cmap=cm.coolwarm, linewidth=0, antialiased=False);  plt.pause(.01); plt.clf()



cont=0
ax.set_zlim(0., 6.)

while t<10:
	cont+=1
	ax = fig.gca(projection='3d')

	t+=dt
	for i in range(1,size[0]-1):
		for j in range(1,size[1]-1):
			pnew[i,j] = p[i,j] + dt*(Dp/dx**2*(p[i+1,j]-2*p[i,j]+p[i-1,j])+Dp/dy**2*(p[i,j+1]-2*p[i,j]+p[i,j-1])+p[i,j]**2*q[i,j]+C-(K+1)*p[i,j])
			qnew[i,j] = q[i,j] + dt*(Dq/dx**2*(q[i+1,j]-2*q[i,j]+q[i-1,j])+Dq/dy**2*(q[i,j+1]-2*q[i,j]+q[i,j-1])-p[i,j]**2*q[i,j]+K*p[i,j])
	p=pnew.copy()
	q=qnew.copy()

	p[0,:]=p[1,:]; p[size[0]-1,:]=p[size[0]-2,:]; p[:,0]=p[:,1]; p[:, size[1]-1]=p[:, size[1]-2]
	q[0,:]=q[1,:]; q[size[0]-1,:]=q[size[0]-2,:]; q[:,0]=q[:,1]; q[:, size[1]-1]=q[:, size[1]-2]
"""
	if(cont%10==0):
		ax.set_zlim(0., 6.)
		print(t)
		surf = ax.plot_surface(x, y, q , cmap=cm.coolwarm, linewidth=0, antialiased=False);  plt.pause(.01)
		plt.clf()
"""
plt.show()

plt.figure()
plt.pcolor(p, cmap='RdBu')
plt.colorbar()
plt.show()

plt.figure()
plt.pcolor(q, cmap='RdBu')
plt.colorbar()
plt.show()
