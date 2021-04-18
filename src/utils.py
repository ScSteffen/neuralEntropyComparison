"""
collection of utility functions
"""

import numpy as np
import numpy.linalg as la
import warnings 
import math 
warnings.simplefilter('error',RuntimeWarning)
eps = np.finfo(float).eps
#from getquad import getquad

"""
def meshgrid2(*arrs):
    arrs = tuple(reversed(arrs))
    lens = list(map(len, arrs))
    dim = len(arrs)
    sz = 1
    for s in lens:
        sz *= s
    ans = []
    for i, arr in enumerate(arrs):
        print(i,arr)
        slc = [1]*dim
        slc[i] = lens[i]
        arr2 = np.asarray(arr).reshape(slc)
        for j, sz in enumerate(lens):
            if j != i:
                arr2 = arr2.repeat(sz, axis=j)
        ans.append(arr2)
    return tuple(ans)
"""

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
        Want to handle vectorized input for alpha
        """

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
                
                GoverExp =  np.exp(np.dot(alpha[:,:],m_v[1:,:]))
    
                integral_GoverExp = np.dot(GoverExp,self.q.w)
                
                a0_out = -np.log(integral_GoverExp)
                
            elif len(alpha.shape) == 1:
                
                m_v = self.q.p
                
                GoverExp =  np.exp(np.dot(alpha[:],m_v))
    
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

class MN_Data:
    def __init__(self,N,quad,closure,**opts):
        self.N = N
        self.quad = quad
        self.closure = closure
        self.opts = opts
        
        self.DT = dualityTools(closure,N,quad)
        
    def make_train_data(self,strat,*args,**kwargs):
        
        self.train_strat = strat
        
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
                
                #total_data = np.hstack([entropy_data[:,np.newaxis],*[alpha_da+str(i) for i in range(1,N+1)]])
                
                #df_data = pd.DataFrame(total_data,columns = df_cols)
                #ta,moment_data])
                
                #data_cols = ['h',*['alpha'+str(i) for i in range(0,N+1)],*['u'+str(i) for i in range(0,N+1)]]
            
        elif self.N >= 1:
            
            if self.train_strat == 'uniform':
            
                self.train_param_dict = dict()
                
                linear_data = []

                for i in range(1,N+1):
                
                    self.train_param_dict["num_alpha"+str(i)] = args[i-1][-1]
                    self.train_param_dict["alpha"+str(i)+"_min"] = args[i-1][0]
                    self.train_param_dict["alpha"+str(i)+"_max"] = args[i-1][1]

                    linear_data.append(np.linspace(self.train_param_dict["alpha"+str(i)+"_min"],\
                                                                    self.train_param_dict["alpha"+str(i)+"_max"],\
                                                                    self.train_param_dict["num_alpha"+str(i)]))

                #Attempting to evaluate in vectorized manner 
                
                mesh = np.meshgrid(*linear_data)
                alpha_data = np.vstack(list(map(np.ravel,mesh)))
                alpha_data = alpha_data.T
                
                #mesh = meshgrid2(*linear_data)
                #alpha_data = np.vstack([mesh[0].ravel(),mesh[1].ravel()])
                #alpha_data = np.vstack(list(map(np.ravel,mesh)))
                
                alpha0_vals = self.DT.alpha0surface(alpha_data)
                
                alpha_data = np.hstack([alpha0_vals,alpha_data])
                
                moment_data = self.DT.moment_vector(alpha_data)
                
                entropy_data = self.DT.entropy(alpha_data)
                
                entropy_data = entropy_data[:,np.newaxis]
                
                total_data = np.hstack([entropy_data[:,np.newaxis],alpha_data,moment_data])
                
                data_cols = ['h',*['alpha'+str(i) for i in range(0,N+1)],*['u'+str(i) for i in range(1,N+1)]]
                
                df_data = pd.DataFrame(total_data,columns = data_cols)
                
                #df_data.to_csv() 
    def make_test_data(self,strat,*args,**kwargs):
        
        self.test_strat = strat 
        
        if self.N == 1:
            
            pass
        
        elif self.N >= 1:
            
            if self.test_strat == 'uniform':
                
                self.test_param_dict = dict()
                
                linear_data = []
                
                
                self.test_param_dict["num_u0"] = args[0][-1]
                self.test_param_dict["u0_min"] = args[0][0]
                self.test_param_dict["u0_max"] = args[0][1] 
                linear_data.append(np.linspace(self.test_param_dict["u0_min"],\
                                               self.test_param_dict["u0_max"],\
                                               self.test_param_dict["num_u0"]))
                for i in range(1,N+1):
                    
                    self.test_param_dict["num_alpha"+str(i)] = args[i][-1]
                    self.test_param_dict["alpha"+str(i)+"_min"] = args[i][0]
                    self.test_param_dict["alpha"+str(i)+"_max"] = args[i][1]
        
                    linear_data.append([np.linspace(self.test_param_dict["alpha"+str(i)+"_min"],\
                                                                    self.test_param_dict["alpha"+str(i)+"_max"],\
                                                                    self.test_param_dict["num_alpha"+str(i)])])
        
                #Attempting to evaluate in vectorized manner 
                
                u0_mesh = np.linspace(self.test_param_dict["u0_min"],\
                                      self.test_param_dict["u0_max"],\
                                      self.test_param_dict["num_u0"])
                
                alpha_mesh = np.meshgrid2(*linear_data[1:])
            
                alpha_data = np.vstack(map(np.ravel,alpha_mesh))
                
                alpha0_vals = self.DT.alpha0surface(self.alpha_data)
                
                alpha0_vals = np.vstack([alpha0_vals +  u0_mesh[i] for i in range(len(u0_mesh))])
                
                alpha_data = np.vstack([alpha_data for i in range(len(u0_mesh))])
                
                alpha_data = np.hstack([alpha0_vals,alpha_data])
                
                moment_data = self.DT.moment_vector(alpha_data)
                
                entropy_data = self.DT.entropy(alpha_data)
                
                entropy_data = entropy_data[:,np.newaxis]
                
                total_data = np.hstack([entropy_data[:,np.newaxis],alpha_data,moment_data])
                
                data_cols = ['h',*['alpha'+str(i) for i in range(0,N+1)],*['u'+str(i) for i in range(1,N+1)]]
                
                df_data = pd.DataFrame(total_data,columns = df_cols)
        
    
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
    
if __name__ == "__main__":
    N = 2
    Q = getquad('lgwt',10,-1,1,N)
    
    DataClass = MN_Data(N,Q,'M_N')
    DataClass.make_train_data('uniform',[-1,1,10],[-2,2,10])