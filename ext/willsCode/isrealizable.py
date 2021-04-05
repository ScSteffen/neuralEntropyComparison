
"""
This script returns (something ...).

Translated from 'isrealizable.m'

Requires:

    None

Use:
        Call:
            isrealizable(u)
        
        Return:
            [out,minEig,A,B]
            
Inputs:
        u: moment vector - numpy.ndarray of shape (N+1,)
Outputs:

        out: True or False, indicates realizability within minEig tolerance above eps - Boolean
        minEig: minimal eigenvalue of relizability test matrices - float
        A: realizability test matrix 1 - numpy.ndarray of shape (floor(len(u)/2),floor(len(u)/2))
        B: realizability test matrix 2 - numpy.ndarray of shape (floor(len(u)/2),floor(len(u)/2))
        
"""

#1. Import packages, declare formal parameters, define test values for variables outside file scope,
# begin function definition:

import numpy as np
import numpy.linalg as LA

eps = np.finfo(float).eps

def isrealizable(u):
    
#2. Define function:
    
    Np1 = len(u)

    if (Np1%2) == 1: #N of moments is even
        Nover2 = int((Np1-1)/2 + 1e-5)
        Nover2p1 = Nover2 + 1
        A  = np.zeros((Nover2p1,Nover2p1))
        for i in range(1,Nover2p1+1):
            im1 = i-1
            for j in range(1,im1+1):
                A[i-1,j-1] = u[im1+j-1]
                A[j-1,i-1] = A[i-1,j-1]
            A[i-1,i-1] = u[i+im1-1]
        B = np.zeros((Nover2,Nover2))
        for i in range(1,Nover2+1):
            im1 = i-1
            ip1 = i + 1
            for j in range(1,im1+1):
                B[i-1,j-1] = u[im1+j-1] - u[ip1+j-1]
                B[j-1,i-1] = B[i-1,j-1]
            B[i-1,i-1] = u[i+im1-1]-u[i+ip1-1]
    else:           #N of moments is odd
        Np1over2 = int(Np1/2 + 1e-8)
        A = np.zeros((Np1over2,Np1over2))
        B = np.zeros((Np1over2,Np1over2))
        for i in range(1,Np1over2+1):
            im1 = i-1
            for j in range(1,im1+1):
                uipjm1 = u[im1+j-1]
                uipj = u[i+j-1]
                A[i-1,j-1] = uipjm1 + uipj
                A[j-1,i-1] = A[i-1,j-1]
                B[i-1,j-1] = uipjm1 - uipj
                B[j-1,i-1] = B[i-1,j-1]
            uipim1 = u[im1+i-1]
            uipi = u[i+i-1]
            #print(np.shape(A[i-1,i-1]),np.shape(uipim1),np.shape(uipi))
            A[i-1,i-1] = uipim1 + uipi
            B[i-1,i-1] = uipim1 - uipi

    minEigA = np.amin(LA.eigvals(A))
    minEigB = np.amin(LA.eigvals(B))

    minEig =  min([minEigA,minEigB])

    out = (minEig > eps)

    return [out,minEig,A,B]
            
#3. Test Function:

""" Should return True for u_0 > u_1 in N = 1 (len(u) = 2) case
testu = np.array([1.1,1.05])
print(isrealizable(testu))
"""
