# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 16:29:45 2018

@author: Chris
"""


from __future__ import division
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from math import pi,e




def G(x,amp):
    sig=0.75
    return amp*  np.exp(-(x)**2 / (2*sig**2))


iters = 100
val = np.zeros(iters)
val_errs = np.zeros(iters)


for i in range(iters):    
    amp_true = 18
    scatter=2
        
        
    x = np.linspace(-10,10,500)
    y_true = G(x,amp_true)
    y = y_true + scatter*np.random.randn(len(x))
    
    
    
#    N = np.identity(len(x))*scatter**2
    N = np.identity(len(x))*np.std(y)**2
    Ni = np.linalg.inv(N)
    
    A = G(x,1)
    At = np.transpose(A)
    
    Nid = np.dot(Ni,y)
    AtNid = np.dot(At,Nid)
    NiA = np.dot(Ni,A)
    AtNiA = np.dot(At,NiA)
    
    m = AtNid/AtNiA
    
    err2 = 1./AtNiA
    err = np.sqrt(err2)
    
#    print " m_true = %.3f \n m_fit  = %.3f +/- %.3f \n ratio  =  %.3f"%(amp_true,m,err,amp_true/m)
    
    val[i] = m
    val_errs[i] = err






plt.figure(num=51)
plt.clf()
plt.title("One realization of taking data on a source")

plt.plot(x,y,'wo',label="Data")
plt.plot(x,y_true,'b-',label="True")
plt.plot(x,G(x,m),'r--',lw=2,label="Fitted")

plt.legend()
plt.ylim(-10,25)






xs = np.arange(iters)+1

plt.figure(num=52)
plt.clf()
plt.title("Results from 100 realizations of taking data")

plt.subplot(2,1,1)
plt.plot(xs,val,'o',label="Amps")
plt.axhline(amp_true,color='r',label="true")
plt.legend()
plt.ylabel("Amplitude")

plt.subplot(2,1,2)
plt.plot(xs,val_errs,'o',label="errors")
plt.ylabel("Error of amplitude")
plt.xlabel("Revisit number")
plt.legend()





plt.figure(num=53)
plt.clf()
plt.title("Clear correlation between amplitude estimate and uncertainty")
plt.plot(val,val_errs,'o',label="Amps")
plt.xlabel("Amplitude")
plt.ylabel("error of amplitude")
plt.legend()










































