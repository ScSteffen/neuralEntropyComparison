
"""
This script returns the quadrature points and weights for Clenshaw-Curtis quadrature on the interval [a,b]. Unlike Gauss quadratures,
Clenshaw-Curtis is only exact for polynomials up to order N, however, using the FFT algorithm, the weights and nodes are computed in
linear time. This script will calculate for N = 2^{20} +1 (10485877) points in about 5 seconds on a normal laptop computer.


Translated fom 'fclencurt.m'

Written: Greg von Winckel - 02/12/2005

Requires: Objects referenced outside file scope

    None

Use:
    Call:
            fclencurt(N1,a,b)

    Return:
            [x,w]

Inputs:

    N1: number of quadrature points - int
    a: left enpoint of domain of integration - float
    b: right endpoint of domain of integration - float
    
Outputs:

    x: Ordered quadrature points - numpy.ndarray with shape (N1,)
    w: Quadrature weights - numpy.ndarray with shape (N1,)
    
"""

#1. Import requisite modules and initialize function definition (formal parameters):

import numpy as np
import numpy.fft as fft

def fclencurt(N1,a,b):
    
#2. Compute points and weights:
    N = N1-1
    length = b-a
    c = np.zeros((N1,2))
    w = np.zeros(N1)
    c[1,1] = 1
    for k in range(0,N1,2):
        c[k,0] = 2/(1-k**2)
    indices = [k-1 for k in range(N1-1,1,-1)]
    newc = np.concatenate([c,c[indices,:]],axis = 0)
    f = np.real(fft.ifft(newc,axis = 0))
    w[0] = (1/2)*length*f[0,0]
    for k in range(1,N1-1):
        w[k] = length*f[k,0]
    w[N1-1] = (1/2)*(f[N1-1,0])
    x = (0.5)*((b+a)+N*length*f[0:N1,1])
    if x.shape != (N1,) or w.shape != (N1,):
        print('fclencurt failed')
    return [np.flipud(x),w]

#3. Test Function:

"""
N1 = 5
a = 1
b = 2
print(fclencurt(N1,a,b))
"""
