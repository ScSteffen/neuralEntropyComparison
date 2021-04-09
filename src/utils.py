"""
collection of utility functions
"""

import numpy as np

def getTestData():
    return 0

def getTrainingData():
    return 0

### Basis Computation
def computeMonomialBasis1D(quadPts, polyDegree):
    """
    params: quadPts = quadrature points to evaluate
            polyDegree = maximum degree of the basis
    return: monomial basis evaluated at quadrature points
    """
    basisLen = getBasisSize(polyDegree, 1)
    nq = quadPts.shape[0]
    monomialBasis = np.zeros((basisLen, nq))

    for idx_quad in range(0, nq):
        for idx_degree in range(0, polyDegree + 1):
            monomialBasis[idx_degree, idx_quad] = np.power(quadPts[idx_quad], idx_degree)
    return monomialBasis


def getBasisSize(polyDegree, spatialDim):
    """
    params: polyDegree = maximum Degree of the basis
            spatialDIm = spatial dimension of the basis
    returns: basis size
    """

    basisLen = 0

    for idx_degree in range(0, polyDegree + 1):
        basisLen += int(
            getCurrDegreeSize(idx_degree, spatialDim))

    return basisLen


def getCurrDegreeSize(currDegree, spatialDim):
    """
    Computes the number of polynomials of the current spatial dimension
    """
    return np.math.factorial(currDegree + spatialDim - 1) / (
            np.math.factorial(currDegree) * np.math.factorial(spatialDim - 1))
