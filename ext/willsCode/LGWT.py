

"""
This script is for computing the definite integrals usingLegendre-Guass Quadrature.
Computes the Legendre-Gauss nodes and weights on interval [a,b] with truncation order N.

If f is a continuous function on [a,b], with a descritization induced by a vector x
of points in [a,b], evalute the definite integral of the descritization of f via
sum(f.*w).s


Requires:

    None

Use:
    Call:
        'lgwt(N,a,b)

    Return:

        [x,w] 

        

Inputs:

    N: highest order of legendre polynomials (and later the "moment order" of the PDE) - int
    a: left endpoint of interval of integration - float
    b: right endpoint of interval of definition - float

    Optional: epsilon - default to 1e-10

Outputs: [x,w]

    x:  N quadrature points within maximum error epsilon, ordered  - numpy.ndarray of shape (N,)

    w:  N corresponding quadrature weights - numpy n.darray of shape (N,)

Translated from Greg von Winckel's MATLAB 02/25/2005 Edit, extension 'lgwt.m' 

Last Edit: 11/27/2019

"""

#1. Import packages and initialize function definition:

import numpy as np
import math
import random

    #Test Variables:


eps = np.finfo(float).eps


def lgwt(N,a,b):
#2. Prepare y as the initial point guess:
    nodes_init = np.array([i for i in range(N)])
    xu = np.linspace(-1,1,N)
    y = np.cos((math.pi/(2*N))*(2*(nodes_init)+1)) + (0.27/N)*np.sin(math.pi*xu*((N-1)/(N+1)))
#3. Initialize L as matrix for Lgendre Polynomials 0 through N (N+1 in total) evaluated at y, and dL_N as d/dx(P_N) evaluated at y:
    L = np.zeros((N,N+1))
    L[:,0] = 1
    dL_N = np.zeros(N)
#4. Apply Newton method and Bonnet recursion formula to bring the difference in newton iterates below epsilon, resulting in nodes y over [-1,1]:
    y0 = 2
    i = 0
    while (max(abs(y-y0)) > eps):
        i += 1
        L[:,1] = y
        for k in range(1,N):
            L[:,k+1] = ((2*k+1)/(k+1))*np.multiply(y,L[:,k])-(k/(k+1))*L[:,k-1]
        dL_N = np.divide((N+1)*(L[:,N-1] - np.multiply(y,L[:,N])),(1-y**2))
        y0 = y
        y = y0 - np.divide(L[:,N],dL_N)
        if i > (1/eps)*1e2:
            print("LGWT fail: More than more than"+str((1/eps)*1e2)+"iterations to compute zeros via Newton")
            break
#5. Map the nodes from [-1,1] onto [a,b], yielding quadrature points x:
    x = (a*(1-y)+b*(1+y))/2
#6. Compute weights:
    w = np.zeros(N)
    for i in range(N):
        w[i] = (b-a)/((1-y[i]**2)*(dL_N[i]**2)*(N/(N+1))**2)
#7. Define function output:
    return [np.flipud(x),w]

#8. Test Function:

"""
a,b = lgwt(4,1,2)
print(a,b)
"""
