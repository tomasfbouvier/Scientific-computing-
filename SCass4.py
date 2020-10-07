# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 08:49:16 2020

@author: Usuario
"""


import numpy as np
import matplotlib.pyplot as plt

x1=0.01; x2=0.; y=0; z=0;

p1=5.; p2=5.; q=100; r=100
a1=10; a2=5; b1=5; b2=1; b3=1; c1=1; c2=1; d1=1
e=15000;

r1=0.005; r2=0.005; r3=0.005; r4=0.005;

t=0; dt=0.001


ts=[]
x1s=[]
x2s=[]
ys=[]
zs=[]
while(t<0.2):
    t+=dt
    dx1= dt*((a1*x1+a2*x2)*(p1-x1)-r1*x1)
    dx2= dt*((b1*x1+b2*x2+b3*y)*(p2-x2)-r2*x2)
    dy= dt*((c1*x2+ c2*z)*(q-y)-r3*y)
    dz= dt*((d1*y*(r-z) + e*(p1-x1))-r4*z)

    x1+=dx1; x2+=dx2; y+=dy; z+=dz;
    ts.append(t); x1s.append(x1); x2s.append(x2); ys.append(y); zs.append(z)
    

plt.figure()
plt.plot(ts,x1s) 

plt.figure()
plt.plot(ts,x2s) 


plt.figure()
plt.plot(ts,ys)

plt.figure()
plt.plot(ts,zs)

plt.show()



