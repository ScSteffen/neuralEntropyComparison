"""
This script is used to compare the L1 error in various moment --> multipiler duality maps using the M_N cloure map 

Requires:
    Variables defined outside file scope:
        getquad: q - ouptut of getquads
        normgLast
        optstuffset: optstuff - output of optstuffset
        
Use:
    Call:
        dualfixedpoly1Q(u,alpha_init,optstuff,q)

    Return:
        [alpha,outs,uClosed]
        
Inputs:
    u: moments to obtain dual for in legendre basis - numpy.ndarray of shape (N_1,)
    alpha_init: initial multipliers for argmin computation - numpy.ndarray of shape (N1,)
    opstuff: a class containing several string data attributes which specify information for the solver setup
    - instance of a class called structinfo
    
Outputs:
    alpha: optimal multipliesrs in legendre basis - numpy.ndarray of shape (N1,)
    outs: a class containing information on the optimization process
    - an instance of a class called structinfo
    uClosed: the moments closed by the algorithm - numpy.ndarray 

"""

#1. Import packages and initialize function definition, define default and within-file-scope-only variables
#declare formal parameters:

import numpy as np
import pandas as pd 
import pickle
import math
from getquad import getquad
from optstuffset import optstuffset
from dualfixedpoly1Q_opts import dualfixedpoly1Q_opts
from Simplical_Approx_Tools import cartproduct, meshdata_2d
#from scipy import linalg 
from numpy import linalg as LA
#from scipy.optimize import minimize
#from scipy.optimize import Bounds
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.patches as mpatches
def truncate(number, digits):
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def momentcone(data):
    if 'ratio' in data.keys(): 
        ratio = data['ratio']
    else:
        ratio = 1
    numx = data['nums'][0]
    a = data['starts'][0]
    b = data['u0max']
    shape = data['shape']
    if shape == 'log linear cone':
        u0min = data['starts'][0]
        u0max = data['u0max']
        c = data['logstep']
        xnum = data['nums'][0]
        ynums = data['nums'][1] 
        graph = np.array([[a,a]])
        xdom = np.array([u0min*(c**i) for i in range(xnum)],dtype = float)
        if abs(xdom[-1] - u0max) < 1e-8:
            pass
        else:
            print('\n\n','momentcone failed in log linear cone geometry; u0 domain final element mismatch','\n\n')
        for i in range(numx):
            z = ratio*abs(xdom[i])
            ydom,ystep = np.linspace(-z,z,ynums[i],retstep = True)
            for j in range(ynums[i]):
                if i == 0:
                    point = np.array([[xdom[i],0]],dtype = float)
                else:
                    point = np.array([[xdom[i],ydom[j]]],dtype = float)
                graph = np.concatenate([graph,point],axis = 0)
        graph = np.delete(graph,axis = 0,obj = 0)
    elif shape == 'uniform cone':
        numy = data['nums'][1]
        xdom,xstep = np.linspace(a,b,numx,retstep = True)
        ystep = data['steps'][1]
        graph = np.array([[a,a]])
        for j in range(numx):
            z = ratio*abs(xdom[j])
            if z <= floor:
                z = 0
            ydom,ystep[j] = np.linspace(-z,z,numy,retstep = True)
            for i in range(numy):
                point = np.array([[xdom[j],ydom[i]]])
                graph = np.concatenate([graph,point],axis = 0)
        graph = np.delete(graph,axis = 0,obj = 0)
    elif shape == 'linear cone' or shape == 'nested cone':
        xdom,xstep = np.linspace(a,b,numx,retstep = True)
        ynums = data['nums'][1] 
        graph = np.array([[a,a]])
        for i in range(numx):
            z = ratio*abs(xdom[i])
            ydom,ystep = np.linspace(-z,z,ynums[i],retstep = True)
            for j in range(ynums[i]):
                if i == 0:
                    point = np.array([[xdom[i],0]],dtype = float)
                else:
                    point = np.array([[xdom[i],ydom[j]]],dtype = float)
                graph = np.concatenate([graph,point],axis = 0)
        graph = np.delete(graph,axis = 0,obj = 0)
    return graph


def makealphaoptim(**args):    
    """ 
    ###
    Parameters (optional keyword):
        N: 
        reg:
    Returns: 
        if reg == False: return alpha_optim,target_optim
            
            alpha_optim: u --> dualfixedpoly1Q_opts(u)[0]
            target_optim: u --> Entropy(alpha_optim(u))
            
        if reg != False: return False, False
    ###
    """
    
    closure = 'M_N'
    
    if 'N' in args:
        N = args['N']
    else:
        N = int(input('Define the N for moment system (called by MN_Duality_Tools.makealphaoptim):'))
        
    if 'reg' in args:
        reg = args['reg']
    else:
        reg = False
    
    
    """
    If no regularization, i.e. 
    reg == False, define MN Duality map 
    (u --> alpha) via dualfixedpoly1Q_opts 
    """ 
    
    if reg == False:
        
        class likestruct(object):
            def __init__(self):
                self.a = []
        moreopts = likestruct()
        
        #Define quadrature rule: clencurt or lgwt
        
        moreopts.qrule = 'lgwt'
        
        #Define number of spatial scells
        
        moreopts.n = 20
        
        #Define number of quadrature points:
        
        moreopts.nq0 = 10
        
        optstuff = optstuffset(N,moreopts)
        
        q = getquad(moreopts.qrule,moreopts.nq0,-1,1,N)
        
        def alpha_optim(u):
            alpha = dualfixedpoly1Q_opts(closure,u,0,optstuff,q)[0]
            return alpha
        def target_optim(u):
            return Entropy(alpha_optim(u))
    
    elif reg == 'full':
        """ 
        If reg == 'full' set alpha_optim,target_optim = False,False
        """
        alpha_optim = False
        target_optim = False
        
    elif reg == 'partial': 
        """
        If reg == 'partial' set alpha_optim,target_optim = False,False
        """
        alpha_optim = False
        target_optim = False
        
    return alpha_optim,target_optim


def makeandsavealphadomain(shape,**opts):
    """ Requires:
            shape - 'projection','urect' : choose shape of multiplier domain
            interactive - True/False (bool) : choose whether or not to use default parameters (False), or to specify each parameter individually (6 total)
    """
    
    if 'saveappend' in opts:
        
        saveappend = opts['saveappend']
        
    else:
        
        saveappend = str()
    
    if 'reg' in opts:
        
        reg = opts['reg']
        
    else:
        
        reg = False
    
    if 'interactive' in opts:
        
        interactive = opts['interactive']
        
    else:
        
        interactive = False
        
    if shape == 'projection':
    
        if interactive == False:
            
            if 'a1triplet' in opts and 'u0triplet' in opts:
                
                a1min,a1max,numa1 = opts['a1triplet']
                
                u0min,u0max,numa0 = opts['u0triplet']
                
            else:
                
                """tightest binding default values"""
                
                a1min,a1max,numa1 = -36,36,144
        
                u0min,u0max,numa0 = 1e-2,8,160
            
        else:
    
            a1min,a1max,numa1 = input("Recall that the sample lines will be determined by slopes r = coth(a1)-(1/a1). Enter list of [a1min,a1max,numa1]:").split(",")
            
            a1min,a1max,numa1 = float(a1min),float(a1max),int(numa1)
    
            u0min,u0max,numa0 = input("Recall that a0 will interpolate the u0 = 1 line with u1 = r(a_1) across this range of u0. Enter list of [u0min,u0max,numa0]:").split(",")
        
            u0min,u0max,numa0 = float(u0min),float(u0max),int(numa0)
            
        a1dom,a1step = np.linspace(a1min,a1max,numa1,retstep = True)
        
        hull_ratio = 1/(np.tanh(a1max)) - 1/(a1max)
    
        a0min_vec = np.zeros((numa1,))
    
        a0max_vec = np.zeros((numa1,))
    
        for i in range(numa1):
    
            if abs(a1dom[i]) < 1e-8:
    
                a0min_vec[i] = np.log((u0min/2)*1)
    
                a0max_vec[i] = np.log((u0max/2)*1)
    
            else:
    
                a0min_vec[i] = np.log((u0min/2)*abs((a1dom[i]/np.sinh(a1dom[i]))))
    
                a0max_vec[i] = np.log((u0max/2)*abs((a1dom[i]/np.sinh(a1dom[i]))))
    
        multiplier_domain= np.zeros((1,2),dtype = float)
    
        for i in range(numa1):
            
            m = a0min_vec[i]
            
            M = a0max_vec[i] 
    
            if m >= M:
                
                print('a0min_vec failed at',i)
    
            a0dom_temp = np.linspace(m,M,numa0)
    
            for j in range(numa0):
    
                point = np.array([[a0dom_temp[j],a1dom[i]]])
    
                multiplier_domain = np.concatenate([multiplier_domain,point],axis = 0)
                
        #Generate Sample Domain from Multiplier Domain
                
        multiplier_domain = np.delete(multiplier_domain,obj = 0,axis = 0)
        
        #u1min = min(moment_domain[:,1])
        
        #u1max = min(moment_domain[:,1])

        #Document Multiplier Domain Info: 
        
        multiplier_info = [('proj',u0min,u0max,a1min,a1max,numa0,numa1,truncate(hull_ratio,4))]
    
        multiplier_cols = ['Geometry','u0min','u0max','a1min','a1max','numa0','numa1','r']
    
        multiplier_table = pd.DataFrame(multiplier_info,columns = multiplier_cols)
        
        with open('hull_ratio_data'+saveappend+'.pickle','wb') as fileratio:
             
             pickle.dump(hull_ratio,fileratio)
        
        with open('multiplier_domain_data'+saveappend+'.pickle','wb') as filedom:
                
            pickle.dump(multiplier_domain,filedom)
                    
        with open('multiplier_table_data'+saveappend+'.pickle','wb') as filemult:
                
            pickle.dump(multiplier_table,filemult)
        
        return [multiplier_domain,hull_ratio,multiplier_table]
        
    elif shape == 'urect':
            
        if reg != False:
                
            if interactive == False:
                
                if 'a0triplet' and 'a1triplet' in opts:
            
                    a0min,a0max,numa0 = opts['a0triplet']
                    
                    a1min,a1max,numa1 = opts['a1triplet']
                    
                else:
                    """ tightest-bound default setting for regularized """
                                                    
            elif interactive == True:
                 
                raise ValueError('unspecified logic sequence in "makeandsavealphadomain" from MN_Duality_Tools: interactive urect reg')
                
                    
        elif reg == False:
            
            if interactive == False:
                    
                if 'a0triplet' and 'a1triplet' in opts:
                    
                    a0min,a0max,numa0 = opts['a0triplet']
                    
                    a1min,a1max,numa1 = opts['a1triplet']
                        
                else:
                    
                    raise ValueError('unspecified logic sequence in "makeandsavealphadomain" in MN_Duality_Tools: defaults false reg no interactive')
                    
                    """tightest-bound default setting for non-regularized"""
                
            elif interactive == True:
                
                raise ValueError('unspecified logic sequence in "makeandsavealphadomain" from MN_Duality_Tools: true interactive urect false reg')
                    
        a0dom,a0step = np.linspace(a0min,a0max,numa0,retstep = True)
        
        a1dom,a1step = np.linspace(a1min,a1max,numa1,retstep = True)
        
        if reg == False:
        
            hull_ratio = 1/(np.tanh(a1max)) - 1/(a1max)
            
        elif reg == 'partial':

            pass
            
        elif reg == 'full':
            
            pass
        
        multiplier_domain = cartproduct(a0dom,a1dom)
        
        multiplier_info = [('urect',a0min,a0max,a1min,a1max,numa0,numa1,truncate(a0step,4),truncate(a1step,4))]

        multiplier_cols = ['Geometry','a0min','a0max','a1min','a1max','numa0','numa1','a0step','a1step']

        multipliertable = pd.DataFrame(multiplier_info,columns = multiplier_cols)
        
        if reg == False:
        
            with open('hull_ratio_data'+saveappend+'.pickle','wb') as fileratio:
                
                pickle.dump(hull_ratio,fileratio)

        with open('multiplier_domain_data'+saveappend+'.pickle','wb') as filedom:
        
            pickle.dump(multiplier_domain,filedom)
            
        with open('multiplier_table_data'+saveappend+'.pickle','wb') as filemult:
        
            pickle.dump(multipliertable,filemult)
                        
        return [multiplier_domain,multipliertable]


def makeandsavetestdomain(interactive,cutoff_ratio,**opts):
    
    if 'saveappend' in opts:
        
        saveappend  = opts['saveappend']
        
    else:
        
        saveappend = str()

    if interactive == False:

        test_shape = 'linear cone'

    else:
        
        test_shape = input("Enter a test domain shape. Options are ['linear cone',...]")

    if test_shape == 'linear cone':

        if interactive == False:
                
            u0min_test,u0max_test,numu0_test = 1e-3,7,40
    
            test_increment = 1
            
            test_ratio = cutoff_ratio-0.05
    
        elif interactive == True:
                
            u0min_test,u0max_test,numu0_test,test_increment = input("Enter a commma-sep list u0min_test,u0max_test,numu0_test,test_increment:").split(",")
            
            test_ratio = float(input("Enter a test domain ratio below ("+str(cutoff_ratio)+"):"))
    
        u0min_test,u0max_test,numu0_test,test_increment = float(u0min_test),float(u0max_test),int(numu0_test),int(test_increment)


        while test_ratio >= cutoff_ratio:
            
            test_ratio = float(input("Test Ratio Too Large. Enter Test Ratio Below ("+str(cutoff_ratio)+") :"))

        test_data = meshdata_2d(u0min_test,u0max_test,numu0_test,ratio = test_ratio,m = test_increment,shape = test_shape)
    
        test_domain = momentcone(test_data)
    
        #Document Test domain Info for Table 
    
        testinfo = [('lin cone',u0min_test,u0max_test,numu0_test,test_increment,test_ratio,test_data["size"])]
        test_cols = ['Geometry','u0 Min','u0 Max','Num u0','k','r','NumPts']
        testtable = pd.DataFrame(testinfo,columns = test_cols)
        with open('testdata_data'+saveappend+'.pickle','wb') as filetestdata:
            pickle.dump(test_data,filetestdata)
        with open('testdomain_data'+saveappend+'.pickle','wb') as filetestdom:
            pickle.dump(test_domain,filetestdom)
        with open('testtable_data'+saveappend+'.pickle','wb') as filetesttable:
            pickle.dump(testtable,filetesttable)
        
        return testtable,test_domain


def moment_vector(a,tol = 1e-10):
    """
    Call:
           momentmapvec
           
    Return: 
            u
    
    Input: 
           a: vector of moment multiplier variable, alpha - np.ndarray of shape (2,)
    Output: 
            u: moment vector associated to multipliers a_0 and a_1 by the MN closure map, computed via analytic formula (not quadrature) - np.ndarray of shape (2,)
    """
    if abs(a[1]) < tol:
        u_0 = 2*np.exp(a[0])
        u_1 = 0
    else:
        u_0 = 2*np.exp(a[0])*(np.sinh(a[1])/a[1])
        u_1 = 2*np.exp(a[0])*((a[1]*np.cosh(a[1]))-np.sinh(a[1]))/(a[1]**2)
    return np.array([u_0,u_1])

def Entropy(a,tol = 1e-10):
    if len(a.shape) > 1:
        a_0,a_1 = a[:,0],a[:,1]
        inside = abs(a_1) < tol
        outside = 1-inside
        return inside*(2*(a_0-1)*np.exp(a_0)) + outside*(2*np.exp(a_0))*((a_0-2)*np.divide(np.sinh(a_1),a_1) + np.cosh(a_1))
    else:
        a_0,a_1 = a[0],a[1]
        if abs(a_1) < tol:
            return 2*(a_0-1)*np.exp(a_0)
        else:
            return 2*np.exp(a_0)*((a_0-2)*np.divide(np.sinh(a_1),a_1) + np.cosh(a_1))

def moment_vector_fullreg(a,gamma = 1e-4,tol = 1e-8):
    """floor case for a[1]"""
    if abs(a[1]) < tol:
        return np.array([2*np.exp(a[0]),0]) + gamma*np.array([a[0],0])
    else:
        return 2*np.exp(a[0])*np.array([np.sinh(a[1])/a[1],(a[1]*np.cosh(a[1]) - np.sinh(a[1]))/(a[1]**2)]) + gamma*a
    
def entropy_fullreg(a,gamma = 1e-4,tol = 1e-8):
    if abs(a[1]) < tol:
        return 2*(a[0]-1)*np.exp(a[0]) + gamma*abs(a[0])
    else:
        return 2*np.exp(a[0])*((a[0]-2)*(np.sinh(a[1])/a[1]) + np.cosh(a[1])) + gamma*LA.norm(a)
    
    
def moment_vector_partialreg(a,gamma = 1e-4,tol = 1e-8):
    if abs(a[1]) < tol:
        return np.array([2*np.exp(a[0]),0])
    else:
        return 2*np.exp(a[0])*np.array([np.sinh(a[1])/a[1],(a[1]*np.cosh(a[1]) - np.sinh(a[1]))/(a[1]**2)]) + gamma*np.array([0,a[1]])
    
def entropy_partialreg(a,gamma= 1e-4,tol = 1e-8):
    if abs(a[1]) < tol:
        return 2*(a[0]-1)*np.exp(a[0]) 
    else:
        return 2*np.exp(a[0])*((a[0]-2)*(np.sinh(a[1])/a[1]) + np.cosh(a[1])) + gamma*abs(a[1])
    

def dual_vector(u):
    if np.abs(u[1]/u[0]) < 1/3:
        a_1 = 3*u[1]/u[0]
    else:
        epsilon = 1/((3*(u[1]/u[0]))**3)
        a_1 = np.arctanh(1/((3*u[1]/u[0])+epsilon))
    a_0 = np.log(np.abs(((u[0]*(a_1+0.1))/np.sinh((a_1+0.1)))))
    a_init = np.array([a_0,a_1],dtype = float)
    def lstsq(a):
        return (moment_vector(a)[0]-u[0])**2 + (moment_vector(a)[1]-u[1])**2
    """
    def obj_grad(a):
        obj_grad_1 = 2*f(a[1],0)*g(a[0])*(f(a[1],0)*g(a[0])-u_0) + 2*f(a[1],1)*g(a[0])*(f(a[1],1)*g(a[0])-u_1)
        obj_grad_2 = 2*f(a[1],1)*g(a[0])*(f(a[1],0)*g(a[0])-u_0) + 2*f(a[1],2)*g(a[0])*(f(a[1],1)*g(a[0])-u_1)
        return np.array([obj_grad_1,obj_grad_2],dtype = float)
    #def obj_hess(a):
        #a_1 = a[1]
        a_0 = a[0]
        h_11 = 4*squared(f(a_0,0))*squared(g(a_0))-2*f(a_1,0)*g(a_0)*u_0 + 4*squared(f(a_1,1))*squared(g(a_0)) - 2*f(a_1,1)*g(a_0)*u_1
        h_12 = 4*f(a_1,0)*f(a_1,1)*squared(g(a_0)) - 2*f(a_1,1)*g(a_0)*u_0 + 4*f(a_1,1)*f(a_1,2)*squared(g(a_0)) - 2*f(a_1,2)*g(a_0)*u_1
        h_21 = 4*f(a_1,1)*f(a_1,0)*squared(g(a_0)) - 2*f(a_1,1)*g(a_0)*u_0 + 4*f(a_1,2)*f(a_1,1)*squared(g(a_0)) - 2*f(a_1,2)*g(a_0)*u_1
        h_22 = 2*(f(a_1,2)*f(a_1,1) + squared(f(a_1,1)))*squared(g(a_0)) - 2*f(a_1,2)*g(a_0)*u_0 + 2*(f(a_1,3)*f(a_1,1) + squared(f(a_1,2)))*squared(g(a_0)) - 2*(f(a_1,3)*g(a_0)*u_1)
        return np.array([[h_11,h_12],[h_21,h_22]],dtype = float)
    """
    moment = minimize(lstsq,x0 = a_init,method = 'CG',jac = '2-point', tol = 1e-5, options = {'gtol':1e-5})
    val = moment.x
    return val


def ratiofunc(x):
    if abs(x) < 1e-6:
        val = 0
    else:
        val = 1/(np.tanh(x))-1/(x)
    return val 


def entropygraph(entropy_function,moment_vector,multiplier_domain,n):
    graph = np.zeros((1,n+1))
    for a in multiplier_domain:
        value = entropy_function(a)
        position = moment_vector(a)
        element = position.tolist()
        element.append(value)
        point = np.array([element],dtype = float)
        graph = np.concatenate([graph,point],axis = 0)
    graph = np.delete(graph,axis = 0, obj = 0)
    return graph


def approxplot(approx_moments,moment_domain):
    fig = plt.figure()
    fig.show()
    ax = fig.add_subplot(111)
    ax.scatter(moment_domain[:,0],moment_domain[:,1],color = 'b', label = 'Moment Data')
    ax.scatter(approx_moments[:,0],approx_moments[:,1],color = 'r', label = 'Moment Approximations')
    ax.set_title('$u_0 = \exp{(a_0)}\cdot(\frac{sinh(a_1)}{a_1})$'+" || "+r'$ u_1 =\exp{(a_0)}\cdot(\frac{\cosh(a_1)-\frac{\sinh(a_1)}{a_1}}{a_1})$')
    plt.xlabel('$u_0$')
    plt.ylabel('$u_1$')
    ax.legend()
    plt.show()

