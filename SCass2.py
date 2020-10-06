# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from example_matrices import *
Kmat = np.load("Chladni-Kmat.npy")
from chladni_show import *


def gershgorin(A):
	centres=[]
	radii=[]
	for i in range (np.shape(A)[0]):
		c=sum(abs(A[i, :]))-abs(A[i,i])
		r=sum(abs(A[:, i]))-abs(A[i,i])
		centres.append(A[i,i])
		radii.append(min([c,r]))
	return(centres, radii)

def rayleigh_qt(A,x):
	landa= (x.T@A@x)/(x.T@x)
	return(landa)

def power_iterate(A):
	x=np.random.rand(np.shape(A)[0])
	k=0
	res=2000000
	while(res>10**(-5)):
		rq1=rayleigh_qt(A,x)
		y=A@x.T	
		x= y/max(abs(y))
		rq2=rayleigh_qt(A,x)
		res=abs(rq2-rq1)
		k+=1
	return(x,k,res)


def rayleigh_iterate(A, x0, shift0):

	x=x0
	I=np.identity(np.shape(A)[0])
	res= 1
	shift=shift0
	iter=0

	while(res>10**(-5)):
		if(np.linalg.det(A-shift*I)==0.):
			res=0
		else:
			x=np.linalg.solve(A-shift*I,x) # I used a solver from numpy	
			x=x/max(abs(x[:]))
			shift= (x.T@A@x)/(x.T@x)
			res= np.linalg.norm(A@x.T-shift*x.T)
		iter+=1
		if(iter==10**4):
			res=0
	return(x, shift, iter, res)


def find_eigenvalues(A):
	centres, radii= gershgorin(A)
	eval=[]
	evec=[]
	obj=len(centres)
	for i in range(len(centres)):
		dwshift=centres[i]- radii[i]
		upshift=centres[i] + radii[i]
		centreshift=centres[i]
		dwevec, dweval, _, _ =  rayleigh_iterate(A,10*np.random.rand(np.shape(A)[0]),dwshift)
		upvec,upval, _, _ = rayleigh_iterate(A,10*np.random.rand(np.shape(A)[0]),upshift)
		centrevec,centreval, _, _ = rayleigh_iterate(A,10*np.random.rand(np.shape(A)[0]),upshift)

		eval.append(dweval); eval.append(upval); eval.append(centreval)
		evec.append(dwevec); evec.append(upvec); evec.append(centrevec)
	evec=np.array(evec)
	eval2=[]
	evec2=[]

	for i in range(len(eval)):
		k=0
		for j in range(len(eval2)):
			if(abs(eval[i]-eval2[j])<=10**(-1)):
				k+=1
		if(k==0):
			eval2.append(eval[i])
			evec2.append(evec[i])

	return(eval2, evec2,obj)

obj=1; eval2=[];
while(len(eval2)!=obj):
	eval2,evec2, obj=find_eigenvalues(Kmat)
evec2=np.array(evec2)

minvec=evec2[np.argsort(eval2)[0],:]
show_waves(minvec, basis_set)
show_nodes(minvec, basis_set)



T=[]

for i in range(np.shape(evec2)[0]):
	T.append(evec2[np.argsort(eval2)[i], :])

T=(np.array(T)).T

lambdas=np.sort(eval2)

print(np.diag(np.linalg.inv(T)@Kmat@T))
print(lambdas)

show_all_wavefunction_nodes(T,lambdas,basis_set)



