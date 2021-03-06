import pylab
import math
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy
import pickle
import pdb
import csv
import pandas as pd
import timeit
import sys

class MaxofGaussians():

    def __init__(self, RV_stats):
       self.YStats=RV_stats
       
     
    def gaussian_approx(self):
       
       X=numpy.sort(self.YStats) #sort RVs by column, mean and std
       idx  = X.shape[0]
       Q = -1*numpy.ones((2*idx, 2))
       Q [0:idx,:] = X

       
       X_g = Q[0,:]
       Q= numpy.delete( Q, 0, 0)

       idx = idx - 1

       while( not (numpy.sum ( numpy.ones((1, Q.shape[0]))* (Q[:,0] == -1) ) == Q.shape[0])):
           
           
           idx = idx + 1
           Q[idx-1,:] = X_g
           X = numpy.r_['0,2', Q[0,:],  Q[1,:] ]
           
           Q = numpy.delete ( Q,  range(0,2), 0) 
           idx = idx - 2
      
           if(X[0,1] == X[1,1] and X[0,1] == 0  and X[0,0] == X[1,0] ):
             	X_g = [X[0,0], X[0,1]]
           else:
		corr = 0
		a = math.sqrt(X[0,1]**2 + X[1,1]**2 - 2*corr*X[0,1]*X[1,1])
		alpha = (X[0,0] - X[1,0])/a

		X_g[0] = X[0,0]*(0.5*math.erfc(-alpha/math.sqrt(2))) + X[1,0]*(0.5*math.erfc(alpha/math.sqrt(2))) +a*(math.exp(-0.5 * (alpha)**2)/(math.sqrt(2*math.pi)));

		var_g = numpy.max((X[0,1]**2 + X[0,0]**2)*(0.5*math.erfc(-alpha/math.sqrt(2))) + (X[1,1]**2 + X[1,0]**2)*(0.5*math.erfc(alpha/math.sqrt(2))) + (X[0,0]+X[1,0])*a*(math.exp(-0.5 * (alpha)**2)/(math.sqrt(2*math.pi))) - X_g[0]**2, 0);
		X_g[1] = math.sqrt(var_g);
  
       
       return X_g


