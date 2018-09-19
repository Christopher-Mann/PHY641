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



# shape of the signal (input has amp = 18.0)
def G(x,amp):
    sig=0.75
    return amp*  np.exp(-(x)**2 / (2*sig**2))


iters = 100
val = np.zeros(iters)
val_errs = np.zeros(iters)

# make many data runs
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





#  visualization of a single data run
plt.figure(num=51)
plt.clf()
plt.title("One realization of taking data on a source")

plt.plot(x,y,'wo',label="Data")
plt.plot(x,y_true,'b-',label="True")
plt.plot(x,G(x,m),'r--',lw=2,label="Fitted")

plt.legend()
plt.ylim(-10,25)

plt.savefig("PS1_Q5_data_example.png")



# plotting history of each data run
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

plt.savefig("PS1_Q5_history.png")






# plotting correlation between amplitude and error for multiple runs
mean_amp = np.mean(val)
std_amp = np.std(val)

plt.figure(num=53)
plt.clf()
plt.title("Clear correlation between amplitude estimate and uncertainty")
plt.plot(val,val_errs,'o',label="Amps")
plt.xlabel("Amplitude")
plt.ylabel("error of amplitude")

plt.axvline(mean_amp,color='r',linewidth=2,label="Mean = %.2f"%(mean_amp))
plt.axvline(mean_amp+std_amp,color='r',linewidth=1,linestyle='--',label="$1\sigma$ = %.2f"%(std_amp))
plt.axvline(mean_amp-std_amp,color='r',linewidth=1,linestyle='--')
plt.legend()

plt.savefig("PS1_Q5_Correlation.png")








































