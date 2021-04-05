
"""

This script computes the values of the legendre polynomials of orders 0 through N
at points given by a 1-d array mu, and stores them in a ((N+1),len(mu)) np.array.

Requires:
    None
    
Use:
    Call:
        lpAtmu(v,N)
    Return:
        p

Inputs:

    v: vector of points at which to evaluate polynomial - 1d np.array
    N: maximal order of legendre polynomials (1 less than output dimension) - nonnegative int

Outputs:

    p: Array of values of the N-th legendre polynomial over v - numpy.ndarray of shape (N+1,v)

Last Edit: 12/10/2019

"""
#1. Import packages and initialize function definition:

import numpy as np
import math
import random



def lpAtMu(v,N):
#2. Compute legendre polynomials over the nodes, through the Nth polynomial, and starting at the 0th polynomial, using the Bonnet recursion formula, (n+1)P_(n+1) = x(2n+1)P_n(x) - nP_(n-1)(x)
    numpoints = len(v)
    p = np.zeros((N+1,numpoints))
    p[0,:] = np.ones(numpoints)
    if N == 0:
        return p[0,:]
    else: 
        p[1,:] = v
        for i in range(1,N):
            p[i+1,:] = ((2*i+1)/(i+1))*np.multiply(p[i,:],v) - (i/(i+1))*p[i-1,:]
        return p

#3. Test Function:

"""
N = 5
v = np.array([-1,1,2,0,3,8,-8])
print(lpAtMu(v,N))
"""
