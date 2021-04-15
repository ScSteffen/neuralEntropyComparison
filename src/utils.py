"""
collection of utility functions
"""

import numpy as np
import numpy.linalg as la
import warnings 
warnings.simplefilter('error',RuntimeWarning)
from getquad import getquad



class dualityTools:
    
    def __init__(self,Closure,N,quad):
        self.N = N
        self.Np1 = N+1
        self.q = quad
        self.closure = Closure
    
    def entropy(self,alpha,tol = 1e-10):
        """
        Want to handle vectorized input for alpha
        in the shape
        (n_vals,len_alpha)
        """
        if self.N == 1:
            if len(alpha.shape) > 1:
                a_0,a_1 = alpha[:,0],alpha[:,1]
                inside = abs(a_1) < tol
                outside = 1-inside
                return inside*(2*(a_0-1)*np.exp(a_0)) + outside*(2*np.exp(a_0))*((a_0-2)*np.divide(np.sinh(a_1),a_1) + np.cosh(a_1))
            else:
                a_0,a_1 = alpha[0],alpha[1]
                if abs(a_1) < tol:
                    return 2*(a_0-1)*np.exp(a_0)
                else:
                    return 2*np.exp(a_0)*((a_0-2)*np.divide(np.sinh(a_1),a_1) + np.cosh(a_1))
                
        elif self.N >= 1:
       
            m_v = self.q.p 
            
            G_alpha = np.exp(np.dot(alpha,m_v))
            etaOfG = np.multiply(G_alpha,np.dot(alpha,m_v))
            
            integral_etaOfG = np.dot(etaOfG,self.q.w)
            
            """
            #Code for a non-vectorized version 
            m_v = q.p.T
            
            G_alpha = np.exp(np.dot(m_v,alpha))
            
            EtaofG = np.multiply(np.dot(m_v,alpha)-1,G_alpha)
            
            integral_EtaofG = np.dot(q.w.T,EtaofG)
            
            return integral_EtaofG
            """
            return integral_etaOfG
    
    def moment_vector(self,alpha,tol = 1e-10):
        """
        Want to handle vectorized input for alpha
        in the shape
        (n_vals,len_alpha)
        """
        if self.N == 1:
            if abs(alpha[1]) < tol:
                u_0 = 2*np.exp(alpha[0])
                u_1 = 0
            else:
                u_0 = 2*np.exp(alpha[0])*(np.sinh(alpha[1])/alpha[1])
                u_1 = 2*np.exp(alpha[0])*((alpha[1]*np.cosh(alpha[1]))-np.sinh(alpha[1]))/(alpha[1]**2)
            return np.array([u_0,u_1])
        
        elif self.N >= 1:
            m_v = self.q.p
            #m_v.shape = (N+1,n_v) 
            #where n_v is numquadpts
            
            G_alpha = np.exp(np.dot(alpha,m_v)) 
            #G_alpha.shape = (n_x,n_v)
            moment_set = []
            for k in range(self.N+1):
                mG = np.multiply(G_alpha,m_v[k,:])
                #Take integral via dotting with quadrature weights. 
                #this is the same as integral mG.
                
                moment_set.append(np.dot(mG,self.q.w))
            
            moments_out = np.hstack([x[:,np.newaxis] for x in moment_set])
            #Moments_out shape is: (n_x,N+1)
            
            """
            #Code for a non-vectorized version
            m_v = q.p.T
            
            G_alpha = np.exp(np.dot(m_v,alpha))
        
            mG = np.multiply(G_alpha[:,np.newaxis],m_v)
            
            integral_mG = np.dot(q.w.T,mG)
            
            return integral_mG 
            """
        
        return moments_out 
    
    def alpha0surface(self,alpha,tol = 1e-10):
        """
        Arguments: 
            alpha: Must be vector of shape (M,N+1) where k \geq 1 
            and N is moment number. We only use the 
            first N components of each vector alpha[i,:]. 
        """
        
        if self.N >= 2:
            
            m_v = self.q.p
            GoverExp =  np.exp(np.dot(alpha[:,1:],m_v))

            integral_GoverExp = np.dot(GoverExp,self.q.w)
            
            a0_out = -np.log(integral_GoverExp)
            
            """
            #Non-vectorized
            m_v = Q.p.T 
            
            Goverexp = np.exp(np.dot(m_v[:,1:],alpha[1:]))
            
            integral_Goverexp = np.dot(Q.w.T,Goverexp)
            
            a0_out = - np.log( integral_Goverexp )
            """
            
        elif self.N == 1:
            #Here the intergral has an elementary form so we can use it 
    
            a0_out = np.zeros((alpha.shape[0],),dtype = float)
            
            alpha_null = np.abs(alpha[:,1:]) < tol 
            
            a0_out[alpha_null] = -np.log(2) 
            
            a0_out[1-alpha_null] = -np.log(np.divide(2*np.sinh(alpha[1-alpha_null,1:]),alpha[1-alpha_null,1:]))
            
            """
            #Non-vectorized 
            if np.abs(alpha) < tol:
                
                a0_out = -np.log(2) 
            else:
                
                a0_out = -np.log(np.divide(2*np.sinh(alpha), alpha))
            """
            
        return a0_out 
    """
    def fobj(self,alpha,u = None,gamma = None):
        if closure == 'P_N':
    
            wG = q.w*np.dot(q.p.T,alpha)
    
            f =  np.sum(q.w*(1/2.0)*np.square(np.dot(q.p.T,alpha)))-np.dot(alpha,u)
    
            g =  np.dot(q.p,wG)-u
            
            output = [f,g,wG]
            
        elif closure == 'M_N':
            
            wG = q.w * np.exp(np.dot(q.p.T,alpha))
        
            f = np.sum(wG)-np.dot(alpha,u)
        
            g = np.dot(q.p,wG) - u 
            
        elif closure == 'M_Nreg':
            
            wG = q.w*np.exp(np.dot(q.p.T,alpha))
            
            f = np.sum(wG) - np.dot(alpha,u) + (gamma / 2)*np.dot(alpha,alpha) 
            
            g = np.dot(q.p,wG) - u + gamma*alpha
        
        output = [f,g,wG]
    
        return output 
    """
#2. Obtain integrals: produce p^{t} x a = [p^{t}_ik * a_k]_{ik} and then 
def getTestData():
    #this is defined in the modelFrame script 
    return 0

def getTrainingData():
    #defined in the modelFrame script 
    return 0

### Basis Computation
def computeMonomialBasis1D(quadPts, polyDegree):
    #to be referenced in dualityTools 
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
    #to be referenced in dualityTools 
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
