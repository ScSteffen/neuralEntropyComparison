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
        """
        Parameters: 
            Closure - (str): String like 'M_N', 'P_N', or 'M_N approx'
            
            N - (int): moment order number 
            
            quad - (obj): quadrature object which must have q.w weights, q.p basis functions evaluated, and
            # quad points, q.mu 
        """
        
        self.N = N
        self.Np1 = N+1
        self.q = quad
        self.closure = Closure
    
    def entropy(self,alpha,tol = 1e-10):
        """
        in the shape
        (n_vals,len_alpha)
        """        Want to handle vectorized input for alpha

        if self.N == 1:
            
            #If we have more than one alpha to evaluate
            if len(alpha.shape) > 1:
                inside = abs(alpha[:,1]) < tol
                outside = 1-inside
                
                h_out = np.zeros((alpha.shape[0],),dtype = float)
                
                h_out[outside] = 2*np.exp(alpha[outside,0]) * ((alpha[outside,0] - 2) * np.divide(np.sinh(alpha[outside,1]),alpha[outside,1]) + \
                      np.cosh(alpha[outside,1]) )
                
                h_out[inside] = 2*( (alpha[inside,0] -1 ) * np.exp(alpha[inside,0]) )
                
                
                #Previous return line: inside*(2*(a_0-1)*np.exp(a_0)) + outside*(2*np.exp(a_0))*((a_0-2)*np.divide(np.sinh(a_1),a_1) + np.cosh(a_1))
                
                return h_out 
            
            #If there is only one alpha to evaluate
            else:
                
                if abs(alpha[1]) < tol:
                    #if alpha_1 small just set equal to alpha_1 = 0 limit
                    return 2*(alpha[0]-1)*np.exp(alpha[0])
                
                else:
                    
                    return 2*np.exp(alpha[0])*((alpha[0]-2)*np.divide(np.sinh(alpha[1]),alpha[1]) + np.cosh(alpha[1]))
                
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
            if len(alpha.shape) == 1:
                
                if np.abs(alpha[1]) < tol:
                    u_0 = 2*np.exp(alpha[0])
                    u_1 = 0
                
                else:
                    
                    u_0 = 2*np.exp(alpha[0])*(np.divide(np.sinh(alpha[1]),alpha[1]))
                    
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
            
            if len(alpha.shape) > 1:
                
                moments_out = np.hstack([x[:,np.newaxis] for x in moment_set])
                
            else:
                
                moments_out = np.hstack([x for x in moment_set])
            
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
            
            if len(alpha.shape) > 1:
                
                m_v = self.q.p
                
                GoverExp =  np.exp(np.dot(alpha[:,1:],m_v))
    
                integral_GoverExp = np.dot(GoverExp,self.q.w)
                
                a0_out = -np.log(integral_GoverExp)
                
            elif len(alpha.shape) == 1:
                
                m_v = self.q.p
                
                GoverExp =  np.exp(np.dot(alpha[1:],m_v))
    
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
    
            if len(alpha.shape) > 1:
                
                a0_out = np.zeros((alpha.shape[0],),dtype = float)
                
                alpha_null = np.abs(alpha[:,1]) < tol 
                
                a0_out[alpha_null] = -np.log(2) 
                
                a0_out[1-alpha_null] = -np.log(np.divide(2*np.sinh(alpha[1-alpha_null,1]),alpha[1-alpha_null,1]))
            
            elif len(alpha.shape) == 1:
                
                if np.abs(alpha[1]) < tol: 
                
                    a0_out =-np.log(2) 
                    
                else:
                    a0_out = -np.log(np.divide(2*np.sinh(alpha[1]),alpha[1]))
        
        return a0_out 


#2. Obtain integrals: produce p^{t} x a = [p^{t}_ik * a_k]_{ik} and then 
"""
class TestData:
  pass 
"""

class TrainingData():
    def __init__(self,N,quad,closure,**opts):
        self.N = N
        self.quad = quad
        self.closure = closure
        self.opts = opts
        
        self.DT = dualityTools(closure,N,quad)
        
    def make_data(strat,*args,**kwargs):
        
        self.strat = strat
        
        if len(args) != (self.N):
            raise ValueError('Number of *args passed must match N, of form (N,min_alpha1,max_alpha1)')
        
        if self.N == 1:
            
            alpha1_info = args[0]
            
            self.alpha1_min,self.alpha1_max,self.num_alpha1 = alpha1_info
            
            if self.strat == 'uniform':
                
                alpha1_mesh,self.alpha1_step = np.linspace(self.alpha1_min,self.alpha1_max,\
                                                                self.num_alpha1,retstep = True)
            
                alpha0_vals = self.DT.alpha0surface(self.alpha1_mesh)
                
                alpha_data = np.hstack([alpha0_vals,alpha1_mesh])
                
                moment_data = self.DT.moment_vector(alpha_data)
                
                entropy_data = self.DT.entropy(alpha_data)
            
        elif self.N >= 1:
            
            if self.strat == 'uniform':
            
                self.param_dict = dict()
                
                linear_data = []


                for i in range(1,N+1):
                
                    self.param_dict["num_alpha"+str(i)] = args[i-1][-1]}
                    self.param_dict["alpha"+str(i)+"_min"] = args[i-1][0]}
                    self.param_dict["alpha"+str(i)+"_max"] = args[i-1][1]}

                    linear_data.append([np.linspace(self.param_dict["alpha"+str(i)+"_min"],\
                                                                    self.param_dict["alpha"+str(i)+"_max"],self.param_dict["num_alpha"+str(i)])])

                #Attempting to evaluate in vectorized manner 
                mesh = np.meshgrid2(*linear_data)
                
                alpha_data = np.vstack(map(np.ravel,mesh))
                
                alpha0_vals = self.DT.alpha0surface(self.alpha1_mesh)
                
                alpha_data = np.hstack([alpha0_vals,alpha1_mesh])
                
                moment_data = self.DT.moment_vector(alpha_data)
                
                entropy_data = self.DT.entropy(alpha_data)





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
