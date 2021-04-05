"""
This script gives functions fobj and momentvector_quad which computes the dual objective function value (for aribitrary alpha, 
fixed u)), and a quadrature based moment-reconstruction, for the dual map alpha --> hat(u)(alpha)

First function translated from 'fobj.m':
    alphaIn formal parrameter is rewritten as alpha_init
    uScaled rewritten as u_s

Requires: (Objects defined outside file scope)
    getquad; q = getquads(rule,nq,a,b,N)
    
Use:
    Call:
        fobj(alpha,u,q)
    Return:
        [f,g,wG]

Inputs:
    alpha: vector of dual variables - numpy ndarray of shape (N+1,)
    u: moment vector - numpy nd array of shape (N+1,)
    q: quadrature class obtained from getquad(arg1,nq,a,b,N) - class with data attributes
Outputs:
    f: value of the objective function at alpha given u - float
    g: gradient of the objective function with respect to alpha - np.ndarray of shape (N+1,)
    wG:  vector ranging over velocities of weight at that velocity times function value at that velocity.
    np.sum(wG) is therefore a velocity integral.  
"""

#1. Import packages and initialize function definition, define default and within-file-scope-only variables

import numpy as np
import numpy.linalg as la
import warnings 
warnings.simplefilter('error',RuntimeWarning)
from getquad import getquad

#Assumes q = getquad( ... )

def fobj(closure,alpha,u,q,gamma = None):

#2. Obtain integrals: produce p^{t} x a = [p^{t}_ik * a_k]_{ik} and then 
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


def alpha0surface(alpha,N = None,Q = None,tol = 1e-10):
    """
    Arguments: 
        alpha: Must be vector of shape (M,N+1) where k \geq 1 
        and N is moment number. We only use the 
        first N components of each vector alpha[i,:]. 
    """
    
    if N == 2:
        
        m_v = Q.p.T 
        
        Goverexp = np.exp(np.dot(m_v[:,1:],alpha[1:]))
        
        integral_Goverexp = np.dot(Q.w.T,Goverexp)
        
        a0_out = - np.log( integral_Goverexp )
        
    if N == 1:
        #Here the intergral has an elementary expression
        #so we use it 
        """
        This is for vectorized evaluation. The code in makeM1data is not vectorized currently.
        
        a0_out = np.zeros((alpha.shape[0],),dtype = float)
        
        #alpha_null is the set of values where we might get overflow from sinh(alpha1)/alpha1
        #I did not take the time to make sinh(alpha1)/alphanull more stable. 
        
        alpha_null = np.abs(alpha[:,1:]) < tol 
        
        a0_out[alpha_null] = -np.log(2) 
        
        a0_out[1-alpha_null] = -np.log( np.divide(2*np.sinh(alpha[1-alpha_null,1:]), alpha[1-alpha_null,1:]) )
        """
        if np.abs(alpha) < tol:
            
            a0_out = -np.log(2) 
        else:
            
            a0_out = -np.log(np.divide(2*np.sinh(alpha), alpha))
            
    return a0_out 


def momentvector_quad(alpha,q,closure = 'M_N'):

    m_v = q.p.T
    
    G_alpha = np.exp(np.dot(m_v,alpha))

    mG = np.multiply(G_alpha[:,np.newaxis],m_v)
    
    integral_mG = np.dot(q.w.T,mG)
    
    return integral_mG 


def entropy_quad(alpha,q,closure = 'M_N'):
    
    m_v = q.p.T
    
    G_alpha = np.exp(np.dot(m_v,alpha))
    
    EtaofG = np.multiply(np.dot(m_v,alpha)-1,G_alpha)
    
    integral_EtaofG = np.dot(q.w.T,EtaofG)
    
    return integral_EtaofG


def IsRealizableM2(x,q,mode = 'alpha'):
    if mode == 'alpha':
        u_hat = momentvector_quad(x,q)
        u_0,u_1,u_2 = u_hat[:]
        if (3*(u_1**2) - (u_0**2) < 2*u_2*u_0) and (u_0 > 0) and (u_2 < u_0):
            return True
        else:
            return False
    elif mode == 'u':
        u_hat = x
        u_0,u_1,u_2 = u_hat[:]
        if (3*(u_1**2) - (u_0**2) < 2*u_2*u_0) and (u_0 > 0) and (u_2 < u_0):
            return True
        else:
            return False

#4. Test function:

if __name__ == "__main__":
    Q = getquad('lgwt',20,-1,1,1)
    alpha = np.array([-10,-10,])
    m_v = Q.p.T
    print(np.multiply(np.dot(m_v,alpha),np.dot(m_v,alpha)),np.dot(m_v,alpha)**2)
    """
    alpha0_pred = alpha0surface(alpha,Q)
    alpha_new = np.array([alpha0_pred,alpha[1],alpha[2]],dtype = float)
    moment_pred = momentvector_quad(alpha_new,Q)
    print(IsRealizableM2(moment_pred,Q))
    print(moment_pred)
    print(momentvector_quad(alpha,Q))
    """
    
