
"""
This script returns the quadrature triplet (or quadruplet for Gauss-Kronot Quadrature) of nodes, weights, and
legendre polynomials evaluated at the nodes.

Translated from 'getquad.m'

Requires: (Objects referenced outside file scope)

    LGWT
    fclencurt
    lpAtMu

Use:
    Call:
        getquad(arg1,nq,a,b,N)

    Return:
        Class with data attributes:
            {mu,w,p,wk,nq}
            

Inputs:

    arg1 = rule = [optional] string for which quadrature rule to apply, default set to 'lgwt' for 
    Gauss quadrature. Other optoins are 'clencurt' for Clenshaw-Curtis quadrature and 'gk'
    for Guass-Kronod quadrature outputs. - string (among specific options)
    nq: number of quadrature points - int or numpy.ndarray of shape (int,), list, or tuple, populated with ints
    a: left enpoint of domain of integration - float
    b: right endpoint of domain of integration - float
    N: number of moments (minus one) - int
    
Outputs:

    mu: Qudrature points - numpy.ndarray of length sum(nq)
    w: Quadrature weights - numpy.ndarray of length sum(nq)
    p: Legendre polynomials evaluted at mu - numpy.ndarray of shape ((N+1,sum(nq))
    wk: [optional; returned for Gauss-Kronod only] the weights of Gauss-Kronod quadrature 
    of order 2*nq_1. (in this case, length(w) = length(mu) = length(p[1,:]) = 2*nq +1 and
    w(1:2:end) = 0 because w holds the gauss weights for the rule of order nq. - notsure yet

"""

#1. Import packages, initialize function definition, declare formal parameters, define test variables:

import numpy as np
from fclencurt import fclencurt
from LGWT import lgwt
from lpAtMu import lpAtMu


def getquad(arg1,nq,a,b,N):

#2. Skip segment of code in 'getquad.m' corresponding to 'argin' number. Assume all arguments are specified
    
    rule = arg1
    wk = None
    
#3. Handle case where nq is numpy.ndarray of shape greater than (1,) by iterating over case where nq is an int:

    if isinstance(nq,np.ndarray) and len(nq) > 1:
        nqTotal = np.sum(nq)
        mu = np.zeros(nqTotal)
        w = np.zeros(nqTotal)
        p = np.zeros((N+1,nqTotal))
        left = a
        right = a + (b-a)/(len(nq))
        gq = getquad(rule,nq[0],left,right,N)
        for k in range(nq[0]):
            mu[k],w[k] = [gq[0][k],gq[1][k]]
        p[:,0:nq[0]] = gq[2]
        for j in range(1,len(nq)):
            nlen = nq[j] 
            lastindex = np.sum(nq[0:j])-1
            startindex = lastindex + 1 
            left = a + j * (b-a)/len(nq)
            right = a + (j+1)*(b-a)/len(nq)
            gq = getquad(rule,nq[j],left,right,N)
            for k in range(startindex,startindex + nq[j]):
                mu[k],w[k] = [gq[0][k-startindex],gq[1][k-startindex]]
            p[:,sindex:(sindex+nq[j])] = gq[2]

#4. Return the case where nq is numpy.ndarray of shape (1,) by rerouting to getquad(...,nq[0],...) which must be an int:
            
    elif isinstance(nq,np.ndarray) and len(nq) == 1:
        return getquad(rule,nq[0],a,b,N)

#3. Handle "base case" wehere nq is int and arg1 is specified:

    elif isinstance(nq,int):
    #Translation: Flip of mu handled in quadrature point files
        
        if rule == 'lgwt':
            mu,w = lgwt(nq,a,b)
            
        elif rule == 'clencurt':
            mu,w = fclencurt(nq,a,b)

        p = lpAtMu(mu,N)
        
#5. Not yet completed quadrature methods:
        
    """
    elif rule == 'cc-comp':
    elif rule == 'lobatto':
    elif rule == 'radau-comp':
    elif rule == 'gk':
    else:
        error('No valid quadrature rule given')
    """

#6. Specify return, as a class, with built-in exception for possibility of wk being specified

    class qout(object):
        def __init__(self):
            self.mu = mu
            self.w = w
            self.p = p
            self.wk = wk
            self.nq = nq
    quadsout = qout()
    
    return quadsout

#7. Test Function:

"""
q = getquad('lgwt',10,1,2,15)
print(q.mu,q.w,q.p,q.wk,q.nq)

"""

