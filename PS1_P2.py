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
from numpy import c_
from matplotlib import rc
from os import sys




# I independently defined the prob. distributions and their logarithmic forms
def Pp(l,n):
    return 1./math.factorial(n) * l**n * np.exp(-l)
Pp = np.vectorize(Pp) # to handle array -> factorial     
def Pg(l,n):
    return 1./np.sqrt(2.*pi*l) * np.exp(-(n-l)**2 / (2*l))



def lnPp(l,n):
    return n*( np.log(l/n) + 1 ) - l - 0.5*np.log(2*pi*n)
def lnPg(l,n):
    return  -0.5*np.log(2*pi*l) - ((n-l)**2 / (2*l))



def FN(l,n):
    return lnPp(l,n)-lnPg(l,n)
FN = np.vectorize(FN)


lamb = np.arange(1,770)

# Choose n values that are 3 or 5 sigma from mean (rounded to nearest whole number)
n3 = np.zeros(len(lamb))
n5 = np.zeros(len(lamb))
for i in range(len(lamb)):
    n3[i] = np.round(lamb[i] + np.sqrt(lamb[i])*3)
    n5[i] = np.round(lamb[i] + np.sqrt(lamb[i])*5)


    

# plot the difference in logs to see when it drops below ln(2)
    
plt.figure(num=21)
plt.clf()

plt.subplot(2,1,1)

plt.plot(lamb,FN(lamb,n3),'b.-',label="3-sigma")

plt.text(0.7,np.log(2)+0.03,"ln(2)")
plt.axhline(np.log(2),color='c',linestyle='--',linewidth=2)

plt.ylim(0,2)
plt.xlim(0,20)
plt.xlabel("Lambda")
plt.ylabel("ln(Pp) - ln(Pg)")
plt.legend()


plt.subplot(2,1,2)
plt.plot(lamb,FN(lamb,n5),'r.-',label="5-sigma")

plt.text(40,np.log(2)+0.1,"ln(2)")
plt.axhline(np.log(2),color='c',linestyle='--',linewidth=2)

plt.ylim(0,2)
plt.xlabel("Lambda")
plt.ylabel("ln(Pp) - ln(Pg)")
plt.legend()

plt.savefig("PS1_Q2.png")






