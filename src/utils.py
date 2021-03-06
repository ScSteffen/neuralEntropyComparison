"""
collection of utility functions
"""

import numpy as np
import pandas as pd
import pickle 
import warnings
import math
from tabulate import tabulate
import os 
from os import path 

warnings.simplefilter('error', RuntimeWarning)
eps = np.finfo(float).eps


# from getquad import getquad

def mono_basis_1D(quadPts, polyDegree):
    # to be referenced in dualityTools
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
    # to be referenced in dualityTools
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


def fclencurt(N1, a, b):
    # 2. Compute points and weights:
    N = N1 - 1
    length = b - a
    c = np.zeros((N1, 2))
    w = np.zeros(N1)
    c[1, 1] = 1
    for k in range(0, N1, 2):
        c[k, 0] = 2 / (1 - k ** 2)
    indices = [k - 1 for k in range(N1 - 1, 1, -1)]
    newc = np.concatenate([c, c[indices, :]], axis=0)
    f = np.real(fft.ifft(newc, axis=0))
    w[0] = (1 / 2) * length * f[0, 0]
    for k in range(1, N1 - 1):
        w[k] = length * f[k, 0]
    w[N1 - 1] = (1 / 2) * (f[N1 - 1, 0])
    x = (0.5) * ((b + a) + N * length * f[0:N1, 1])
    if x.shape != (N1,) or w.shape != (N1,):
        print('fclencurt failed')
    return [np.flipud(x), w]


def lgwt(N, a, b):
    # 2. Prepare y as the initial point guess:
    nodes_init = np.array([i for i in range(N)])
    xu = np.linspace(-1, 1, N)
    y = np.cos((math.pi / (2 * N)) * (2 * (nodes_init) + 1)) + (0.27 / N) * np.sin(math.pi * xu * ((N - 1) / (N + 1)))
    # 3. Initialize L as matrix for Lgendre Polynomials 0 through N (N+1 in total) evaluated at y, and dL_N as d/dx(P_N) evaluated at y:
    L = np.zeros((N, N + 1))
    L[:, 0] = 1
    dL_N = np.zeros(N)
    # 4. Apply Newton method and Bonnet recursion formula to bring the difference in newton iterates below epsilon, resulting in nodes y over [-1,1]:
    y0 = 2
    i = 0
    while (max(abs(y - y0)) > eps):
        i += 1
        L[:, 1] = y
        for k in range(1, N):
            L[:, k + 1] = ((2 * k + 1) / (k + 1)) * np.multiply(y, L[:, k]) - (k / (k + 1)) * L[:, k - 1]
        dL_N = np.divide((N + 1) * (L[:, N - 1] - np.multiply(y, L[:, N])), (1 - y ** 2))
        y0 = y
        y = y0 - np.divide(L[:, N], dL_N)
        if i > (1 / eps) * 1e2:
            print("LGWT fail: More than more than" + str((1 / eps) * 1e2) + "iterations to compute zeros via Newton")
            break
    # 5. Map the nodes from [-1,1] onto [a,b], yielding quadrature points x:
    x = (a * (1 - y) + b * (1 + y)) / 2
    # 6. Compute weights:
    w = np.zeros(N)
    for i in range(N):
        w[i] = (b - a) / ((1 - y[i] ** 2) * (dL_N[i] ** 2) * (N / (N + 1)) ** 2)
    # 7. Define function output:
    return [np.flipud(x), w]


def lpAtMu(v, N):
    # 2. Compute legendre polynomials over the nodes, through the Nth polynomial, and starting at the 0th polynomial, using the Bonnet recursion formula, (n+1)P_(n+1) = x(2n+1)P_n(x) - nP_(n-1)(x)
    numpoints = len(v)
    p = np.zeros((N + 1, numpoints))
    p[0, :] = np.ones(numpoints)
    if N == 0:
        return p[0, :]
    else:
        p[1, :] = v
        for i in range(1, N):
            p[i + 1, :] = ((2 * i + 1) / (i + 1)) * np.multiply(p[i, :], v) - (i / (i + 1)) * p[i - 1, :]
        return p


def getquad(arg1, nq, a, b, N, mono=True):
    # 2. Skip segment of code in 'getquad.m' corresponding to 'argin' number. Assume all arguments are specified

    rule = arg1
    wk = None

    # 3. Handle case where nq is numpy.ndarray of shape greater than (1,) by iterating over case where nq is an int:

    if isinstance(nq, np.ndarray) and len(nq) > 1:
        nqTotal = np.sum(nq)
        mu = np.zeros(nqTotal)
        w = np.zeros(nqTotal)
        p = np.zeros((N + 1, nqTotal))
        left = a
        right = a + (b - a) / (len(nq))
        gq = getquad(rule, nq[0], left, right, N)
        for k in range(nq[0]):
            mu[k], w[k] = [gq[0][k], gq[1][k]]
        p[:, 0:nq[0]] = gq[2]
        for j in range(1, len(nq)):
            nlen = nq[j]
            lastindex = np.sum(nq[0:j]) - 1
            startindex = lastindex + 1
            left = a + j * (b - a) / len(nq)
            right = a + (j + 1) * (b - a) / len(nq)
            gq = getquad(rule, nq[j], left, right, N)
            for k in range(startindex, startindex + nq[j]):
                mu[k], w[k] = [gq[0][k - startindex], gq[1][k - startindex]]
            p[:, sindex:(sindex + nq[j])] = gq[2]

    # 4. Return the case where nq is numpy.ndarray of shape (1,) by rerouting to getquad(...,nq[0],...) which must be an int:

    elif isinstance(nq, np.ndarray) and len(nq) == 1:
        return getquad(rule, nq[0], a, b, N)

    # 3. Handle "base case" wehere nq is int and arg1 is specified:

    elif isinstance(nq, int):
        # Translation: Flip of mu handled in quadrature point files

        if rule == 'lgwt':
            mu, w = lgwt(nq, a, b)

        elif rule == 'clencurt':
            mu, w = fclencurt(nq, a, b)

        if mono:
            p = mono_basis_1D(mu, N)
        else:
            p = lpAtMu(mu, N)

    # 5. Not yet completed quadrature methods:

    """
    elif rule == 'cc-comp':
    elif rule == 'lobatto':
    elif rule == 'radau-comp':
    elif rule == 'gk':
    else:
        error('No valid quadrature rule given')
    """

    # 6. Specify return, as a class, with built-in exception for possibility of wk being specified

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

    def __init__(self, Closure, N, quad):
        """
        Parameters: 
            Closure - (str): String like 'M_N', 'P_N', or 'M_N approx'
            
            N - (int): moment order number 
            
            quad - (obj): quadrature object which must have q.w weights, q.p basis functions evaluated, and
            # quad points, q.mu 
        """

        self.N = N
        self.Np1 = N + 1
        self.q = quad
        self.closure = Closure

    def entropy(self, alpha, tol=1e-10):
        """
        in the shape
        (n_vals,len_alpha)
        Want to handle vectorized input for alpha
        """

        if self.N == 1:

            # If we have more than one alpha to evaluate
            if alpha.shape[0] > 1:

                inside = abs(alpha[:, 1]) < tol
                outside = (1 - inside).astype(bool)

                h_out = np.zeros((alpha.shape[0],), dtype=float)

                h_out[outside] = 2 * np.exp(alpha[outside, 0]) * ( \
                            (alpha[outside, 0] - 2) * np.divide(np.sinh(alpha[outside, 1]), alpha[outside, 1]) + \
                            np.cosh(alpha[outside, 1]))

                h_out[inside] = 2 * ((alpha[inside, 0] - 1) * np.exp(alpha[inside, 0]))

                # Previous return line: inside*(2*(a_0-1)*np.exp(a_0)) + outside*(2*np.exp(a_0))*((a_0-2)*np.divide(np.sinh(a_1),a_1) + np.cosh(a_1))

                # If there is only one alpha to evaluate
            elif alpha.shape[0] == 1:

                if abs(alpha[1]) < tol:
                    # if alpha_1 small just set equal to alpha_1 = 0 limit
                    h_out = 2 * (alpha[0] - 1) * np.exp(alpha[0])

                else:

                    h_out = 2 * np.exp(alpha[0]) * ( \
                                (alpha[0] - 2) * np.divide(np.sinh(alpha[1]), alpha[1]) + np.cosh(alpha[1]))

            return h_out

        elif self.N >= 1:

            m_v = self.q.p

            G_alpha = np.exp(np.dot(alpha, m_v))

            etaOfG = np.multiply(G_alpha, np.dot(alpha, m_v))

            integral_etaOfG = np.dot(etaOfG, self.q.w)

            """
            #Code for a non-vectorized version 
            m_v = q.p.T
            
            G_alpha = np.exp(np.dot(m_v,alpha))
            
            EtaofG = np.multiply(np.dot(m_v,alpha)-1,G_alpha)
            
            integral_EtaofG = np.dot(q.w.T,EtaofG)
            
            return integral_EtaofG
            """
            return integral_etaOfG

    def moment_vector(self, alpha, tol=1e-10):
        """
        Want to handle vectorized input for alpha
        in the shape
        (n_vals,len_alpha)
        """
        if self.N == 1:
            if alpha.shape[0] == 1:
                #               #If only 1 sample
                if np.abs(alpha[1]) < tol:
                    u_0 = 2 * np.exp(alpha[0])
                    u_1 = 0

                    moments_out = np.array([u_0, u_1])
                else:

                    u_0 = 2 * np.exp(alpha[0]) * (np.divide(np.sinh(alpha[1]), alpha[1]))

                    u_1 = 2 * np.exp(alpha[0]) * ((alpha[1] * np.cosh(alpha[1])) - np.sinh(alpha[1])) / (alpha[1] ** 2)

                    moments_out = np.array([u_0, u_1])

            elif alpha.shape[0] > 1:
                # If more than 1 sample

                inside = np.abs(alpha[:, 1]) < tol

                outside = (1 - inside).astype(bool)

                moments_out = np.zeros((alpha.shape[0], 2))

                moments_out[inside, 0] = 2 * np.exp(alpha[inside, 0])
                moments_out[inside, 1] = 0

                moments_out[outside, 0] = 2 * np.exp(alpha[outside, 0]) * (
                    np.divide(np.sinh(alpha[outside, 1]), alpha[outside, 1]))
                moments_out[outside, 1] = 2 * np.exp(alpha[outside, 0]) * np.divide(
                    (alpha[outside, 1] * np.cosh(alpha[outside, 1]) - \
                     np.sinh(alpha[outside, 1])), np.power(alpha[outside, 1], 2))

            return moments_out

        elif self.N >= 1:

            m_v = self.q.p
            # m_v.shape = (N+1,n_v)
            # where n_v is numquadpts

            G_alpha = np.exp(np.dot(alpha, m_v))
            # G_alpha.shape = (n_x,n_v)
            moment_set = []
            for k in range(self.N + 1):
                mG = np.multiply(G_alpha, m_v[k, :])
                # Take integral via dotting with quadrature weights.
                # this is the same as integral mG.

                moment_set.append(np.dot(mG, self.q.w))

            if len(alpha.shape) > 1:

                moments_out = np.hstack([x[:, np.newaxis] for x in moment_set])

            else:

                moments_out = np.hstack([x for x in moment_set])

            # Moments_out shape is: (n_x,N+1)

            """
            #Code for a non-vectorized version
            m_v = q.p.T
            
            G_alpha = np.exp(np.dot(m_v,alpha))
        
            mG = np.multiply(G_alpha[:,np.newaxis],m_v)
            
            integral_mG = np.dot(q.w.T,mG)
            
            return integral_mG 
            """
            return moments_out

    def alpha0surface(self, alpha, tol=1e-10):
        """
        Arguments: 
            alpha: Must be vector of shape (M,N+1) where k \geq 1 
            and N is moment number. We only use the 
            first N components of each vector alpha[i,:]. 
        """

        if self.N >= 2:

            if len(alpha.shape) > 1:

                m_v = self.q.p

                # Removed the 1 from alpha[:,1:] since we are passing
                # an alpha of length N, not N+1 with 0th coordinate useless

                GoverExp = np.exp(np.dot(alpha[:, :], m_v[1:, :]))

                integral_GoverExp = np.dot(GoverExp, self.q.w)

                a0_out = -np.log(integral_GoverExp)

            elif len(alpha.shape) == 1:

                m_v = self.q.p

                GoverExp = np.exp(np.dot(alpha[:], m_v))

                integral_GoverExp = np.dot(GoverExp, self.q.w)

                a0_out = -np.log(integral_GoverExp)

            """
            #Non-vectorized
            m_v = Q.p.T 
            
            Goverexp = np.exp(np.dot(m_v[:,1:],alpha[1:]))
            
            integral_Goverexp = np.dot(Q.w.T,Goverexp)
            
            a0_out = - np.log( integral_Goverexp )
            """

        elif self.N == 1:
            # Here the constraint can be expressed with elementary functions 
            # so we use them.

            if alpha.shape[0] > 1:
                # Conditinal should select more than 1 sample

                a0_out = np.zeros((alpha.shape[0],), dtype=float)

                alpha_in = np.abs(alpha[:]) < tol

                alpha_out = (1 - alpha_in).astype(bool)

                a0_out[alpha_out] = -np.log(2)

                a0_out[alpha_out] = -np.log(np.divide(2 * np.sinh(alpha[alpha_out]), alpha[alpha_out]))

            elif alpha.shape[0] == 1:

                if np.abs(alpha[1]) < tol:

                    a0_out = -np.log(2)

                else:
                    a0_out = -np.log(np.divide(2 * np.sinh(alpha), alpha))

        return a0_out

    # 2. Obtain integrals: produce p^{t} x a = [p^{t}_ik * a_k]_{ik} and then

"""
class TestData:
  pass 
"""

class MN_Data:
    def __init__(self, N, quad, closure, **opts):
        self.N = N
        self.quad = quad
        self.closure = closure
        self.opts = opts

        self.DT = dualityTools(closure, N, quad)

    def make_train_data_wrapper(self, epsilon, alphaMax, nS):
        """
        :param epsilon: distance to the boundary
        :param alpha: maximium value of alpha in one dimension
        :param nS: number of sampling points
        :return: [u,alpha,h]
        """

        if (self.N == 1):
            return self.make_train_data('uniform', epsilon, [-alphaMax, alphaMax, nS])
        elif (self.N == 2):
            return self.make_train_data('uniform', epsilon, [-alphaMax, alphaMax, nS], [-alphaMax, alphaMax, nS])
        elif (self.N == 3):
            return self.make_train_data('uniform', epsilon, [-alphaMax, alphaMax, nS], [-alphaMax, alphaMax, nS],
                                        [-alphaMax, alphaMax, nS])
        elif (self.N == 4):
            return self.make_train_data('uniform', epsilon, [-alphaMax, alphaMax, nS], [-alphaMax, alphaMax, nS],
                                        [-alphaMax, alphaMax, nS], [-alphaMax, alphaMax, nS])

        return 0
    
    def make_test_data_wrapper(self,strat,uParams,*alphaParams):
        if self.N==1:
            return self.make_test_data(strat,uParams,*alphaParams)
        else:
            pass
        

    def make_train_data(self, strat, epsilon, *args, **kwargs):
        """
        Params:
            - strat (type: str): used as selector for sampling strategy. 
            'uniform' is only currently valid value to pass 
            
            - epsilon (type: float): 
                
            - args (type(s): list): each element passed in order must be 
                                list(min_alpha_i,max_alpha_i,num_alpha_i)
                                for   1 \leq i \leq N
            - kwargs:
                - 'savedir' (type: str): full or relative 
                path to filename for saved training data 
        """

        self.train_strat = strat

        if len(args) != (self.N):
            raise ValueError('Number of *args passed must match N, of form (N,min_alpha1,max_alpha1)')

        if self.N == 1:

            if self.train_strat == 'uniform':
                alpha1_info = args[0]

                self.train_param_dict = dict()

                self.train_param_dict['alpha1_min'] = alpha1_info[0]
                self.train_param_dict['alpha1_max'] = alpha1_info[1]
                self.train_param_dict['num_alpha1'] = alpha1_info[-1]

                linear_data = [np.linspace(self.train_param_dict['alpha1_min'], \
                                           self.train_param_dict['alpha1_max'], \
                                           self.train_param_dict['num_alpha1'])]

                alpha1_mesh = np.linspace(self.train_param_dict['alpha1_min'], \
                                          self.train_param_dict['alpha1_max'], \
                                          self.train_param_dict['num_alpha1'])

                alpha0_vals = self.DT.alpha0surface(alpha1_mesh)

                alpha0_vals = np.reshape(alpha0_vals, (alpha0_vals.shape[0], 1))
                alpha1_mesh = np.reshape(alpha1_mesh, (alpha1_mesh.shape[0], 1))

                alpha_data = np.hstack([alpha0_vals, alpha1_mesh])

                moment_data = self.DT.moment_vector(alpha_data)

                entropy_data = self.DT.entropy(alpha_data)

                total_data = np.hstack([moment_data, alpha_data])

                total_data = np.hstack([total_data, entropy_data[:, np.newaxis]])

                data_cols = [*['u' + str(i) for i in range(0, self.N + 1)], \
                             *['alpha' + str(i) for i in range(0, self.N + 1)], 'h']

                # Copied here from N >= 1 case
                del_indices = self.check_realizable(moment_data, epsilon)
                total_data = total_data[del_indices]

                df_data = pd.DataFrame(total_data, columns=data_cols)

                # print(tabulate(df_data, headers='keys', tablefmt='psql'))

        elif self.N >= 1:

            if self.train_strat == 'uniform':

                self.train_param_dict = dict()

                linear_data = []

                for i in range(1, self.N + 1):
                    self.train_param_dict["num_alpha" + str(i)] = args[i - 1][-1]
                    self.train_param_dict["alpha" + str(i) + "_min"] = args[i - 1][0]
                    self.train_param_dict["alpha" + str(i) + "_max"] = args[i - 1][1]

                    linear_data.append(np.linspace(self.train_param_dict["alpha" + str(i) + "_min"], \
                                                   self.train_param_dict["alpha" + str(i) + "_max"], \
                                                   self.train_param_dict["num_alpha" + str(i)]))

                # Evaluate alpha mesh in vectorized manner

                mesh = np.meshgrid(*linear_data)
                alpha_data = np.vstack(list(map(np.ravel, mesh)))
                alpha_data = alpha_data.T

                alpha0_vals = self.DT.alpha0surface(alpha_data)

                alpha_data = np.hstack([alpha0_vals[:, np.newaxis], alpha_data])

                moment_data = self.DT.moment_vector(alpha_data)

                entropy_data = self.DT.entropy(alpha_data)

                total_data = np.hstack([moment_data, alpha_data])

                total_data = np.hstack([total_data, entropy_data[:, np.newaxis]])

                data_cols = [*['u' + str(i) for i in range(0, self.N + 1)], \
                             *['alpha' + str(i) for i in range(0, self.N + 1)], 'h']

                ## remove elements too close to the boundary
                del_indices = self.check_realizable(moment_data, epsilon)

                total_data = total_data[del_indices]
                # print to dataframe
                df_data = pd.DataFrame(total_data, columns=data_cols)

                # print(tabulate(df_data, headers='keys', tablefmt='psql'))

                if 'savedir' in kwargs:
                    self.train_data_path = kwargs['savedir']
                    df_data.to_csv(self.train_data_path, index=False)

                #return [moment_data, alpha_data, entropy_data[:, np.newaxis]]

        # print to file; Will commented this out 12:02 pm CDT since folders not here
        #df_data.to_csv("data/1D/Monomial_M" + str(self.N) + ".csv", index=False)
        return [total_data[:, 0:self.N + 1], total_data[:, self.N + 1:2 * self.N + 2], total_data[:, 2 * self.N + 2:]]

    def make_test_data(self, strat, *args, **kwargs):

        self.test_strat = strat

        if self.N == 1:

            self.test_param_dict = dict()

            linear_data = []

            # To do (Will): only nuance here is the shape of the arrays

            self.test_param_dict["num_u0"] = args[0][-1]
            self.test_param_dict["u0_min"] = args[0][0]
            self.test_param_dict["u0_max"] = args[0][1]
            linear_data.append(np.linspace(self.test_param_dict["u0_min"], \
                                           self.test_param_dict["u0_max"], \
                                           self.test_param_dict["num_u0"]))

            self.test_param_dict['num_alpha1'] = args[1][-1]
            self.test_param_dict['alpha1_min'] = args[1][0]
            self.test_param_dict['alpha1_max'] = args[1][1]
            linear_data.append(np.linspace(self.test_param_dict['alpha1_min'], \
                                           self.test_param_dict['alpha1_max'], \
                                           self.test_param_dict['num_alpha1']))

            u0_mesh = np.linspace(self.test_param_dict["u0_min"], \
                                  self.test_param_dict["u0_max"], \
                                  self.test_param_dict["num_u0"])

            alpha1_mesh = np.linspace(self.test_param_dict['alpha1_min'], \
                                      self.test_param_dict['alpha1_max'], \
                                      self.test_param_dict['num_alpha1'])
            
            alpha0_vals = self.DT.alpha0surface(alpha1_mesh)

            alpha0_vals = np.hstack([alpha0_vals + np.log(u0_mesh[i]) for i in range(len(u0_mesh))])
            alpha1_data = np.hstack([alpha1_mesh for i in range(len(u0_mesh))])

            alpha0_vals = np.reshape(alpha0_vals, (alpha0_vals.shape[0], 1))
            alpha1_data = np.reshape(alpha1_data, (alpha1_data.shape[0], 1))

            alpha_data = np.hstack([alpha0_vals, alpha1_data])

            moment_data = self.DT.moment_vector(alpha_data)

            entropy_data = self.DT.entropy(alpha_data)

            total_data = np.hstack([moment_data, alpha_data])

            total_data = np.hstack([total_data, entropy_data[:, np.newaxis]])

            data_cols = [*['u' + str(i) for i in range(0, self.N + 1)], \
                         *['alpha' + str(i) for i in range(0, self.N + 1)], 'h']

            # Copied here from N >= 1 case
            """
            del_indices = self.check_realizable(moment_data, epsilon)
            total_data = total_data[del_indices]
            """

            df_data = pd.DataFrame(total_data, columns=data_cols)

            print(tabulate(df_data, headers='keys', tablefmt='psql'))

            #return [total_data[:,0:2],total_data[:,3:5],total_data[:,6]]
        
        elif self.N >= 1:

            if self.test_strat == 'uniform':

                self.test_param_dict = dict()

                linear_data = []

                self.test_param_dict["num_u0"] = args[0][-1]
                self.test_param_dict["u0_min"] = args[0][0]
                self.test_param_dict["u0_max"] = args[0][1]
                linear_data.append(np.linspace(self.test_param_dict["u0_min"], \
                                               self.test_param_dict["u0_max"], \
                                               self.test_param_dict["num_u0"]))
                for i in range(1, N + 1):
                    self.test_param_dict["num_alpha" + str(i)] = args[i][-1]
                    self.test_param_dict["alpha" + str(i) + "_min"] = args[i][0]
                    self.test_param_dict["alpha" + str(i) + "_max"] = args[i][1]

                    linear_data.append([np.linspace(self.test_param_dict["alpha" + str(i) + "_min"], \
                                                    self.test_param_dict["alpha" + str(i) + "_max"], \
                                                    self.test_param_dict["num_alpha" + str(i)])])

                # Attempting to evaluate in vectorized manner

                u0_mesh = np.linspace(self.test_param_dict["u0_min"], \
                                      self.test_param_dict["u0_max"], \
                                      self.test_param_dict["num_u0"])

                mesh = np.meshgrid(*linear_data[1:])
                alpha_data = np.vstack(list(map(np.ravel, mesh)))
                alpha_data = alpha_data.T

                alpha0_vals = self.DT.alpha0surface(alpha_data)

                # Want n_alpha_samples \times n_u0_samples to be the size of alpha0_vals

                # We need to use hstack here to sequence these vectors since they are 1-d
                alpha0_vals = np.hstack([alpha0_vals + np.log(u0_mesh[i]) for i in range(len(u0_mesh))])

                # Have to reshape as 2d array in order to hstack with another 2d array
                alpha0_vals = np.reshape(alpha0_vals, (alpha0_vals.shape[0], 1))

                alpha_data = np.vstack([alpha_data for i in range(len(u0_mesh))])
                # Might need to reshape alpha0_vals for this
                alpha_data = np.hstack([alpha0_vals, alpha_data])

                moment_data = self.DT.moment_vector(alpha_data)

                entropy_data = self.DT.entropy(alpha_data)

                # Must make the entropy data fully 2d in order to hstack with other 2d arrays
                entropy_data = entropy_data[:, np.newaxis]

                total_data = np.hstack([moment_data, alpha_data])

                total_data = np.hstack([total_data, entropy_data])

                data_cols = [*['u' + str(i) for i in range(0, N + 1)], \
                             *['alpha' + str(i) for i in range(0, N + 1)], 'h']

                df_data = pd.DataFrame(total_data, columns=data_cols)

                if 'savedir' in kwargs:
                    self.test_data_path = kwargs['savedir']
                    df_data.to_csv(self.test_data_path, index=False)
                    
        return [total_data[:, 0:self.N + 1], total_data[:, self.N + 1:2 * self.N + 2], total_data[:, 2 * self.N + 2:]]
            
                # Sample name for data: Monomial_M2_1d.csv or Monomial_M2_1d_normal.csv

    def check_realizable(self, u, epsilon=1e-5):
        """
        :param u: dim(u) = n_x times (N+1) , where n_x = ns =  number of samples, N =  order of moment system ( N+1) entries
        :return:
        """

        nSys = self.N + 1
        ns = u.shape[0]

        violate_rules = np.array(np.ones((ns,)), dtype=bool)

        if (self.N >= 1):  # M1 closure and upwards
            for i in range(ns):
                if u[i, 1] > 1 - epsilon or u[i, 1] < -1 + epsilon:
                    violate_rules[i] = False

        if (self.N >= 2):  # M2 closure and upwards
            for i in range(ns):
                upper_bound = 1
                lower_bound = u[i, 1] * u[i, 1]
                if u[i, 2] > upper_bound - epsilon or u[i, 2] < lower_bound + epsilon:
                    violate_rules[i] = False

        if (self.N >= 3):  # M3 closure and upwards
            for i in range(ns):
                upper_bound = u[i, 2] - (u[i, 1] - u[i, 2]) * (
                        u[i, 1] - u[i, 2]) / (1 - u[i, 1])
                lower_bound = - u[i, 2] + (u[i, 1] + u[i, 2]) * (
                        u[i, 1] + u[i, 2]) / (1 + u[i, 1])

                if u[i, 3] > upper_bound - epsilon or u[i, 3] < lower_bound + epsilon:
                    violate_rules[i] = False

        if (self.N >= 4):  # M4 closure and upwards
            for i in range(ns):

                upper_bound = (u[i, 2] * u[i, 2] * u[i, 2] - u[i, 3] * u[i, 3] + 2 * u[i, 1] * u[i, 2] * u[i, 3]) / (
                        u[i, 2] - u[i, 1] * u[i, 1])

                lower_bound = u[i, 2] - (u[i, 1] - u[i, 3]) * (u[i, 1] - u[i, 3]) / (1 - u[i, 2])

                if u[i, 4] > upper_bound - epsilon or u[i, 4] < lower_bound + epsilon:
                    violate_rules[i] = 0

        return violate_rules
    
class AnalysisTools():
    def __init__(self,closure,**opts):
        """
        Closure is string like 'M_N'
        """
        self.closure = closure
        pass
    
    def newDF(self,N,domain,datID,method,saveNames):
        """
        domain: <str> must be one of [train', 'test']
        
        datID: <str> - must be one of ['xx','yy']
        
        method: <str> - must be one of ['icnn','ecnn']
        
        saveNames: <list> - must be list of strings 
                            containing checkpoint or shortcut model 
                            names
        """
         #Example: analysis/raw/results_ecnn_train_10.pickle
        filePath = 'analysis/raw/' +'results_'+method+'_'+domain+'_'+datID+'.pickle'
        if path.exists(filePath):
            inpt = input('\n The dataframe specified already exists; override with empty? Type [y/n]: ')
            if 'y' in inpt:
                make = True 
            else:
                print('\n y not found in user input; no new dataframe at path location \n')
                make = False
        else:
            make = True
            
        if make:
            print('\n Making new dataframe at path:',filePath,'\n')
            cols = ['NetID','Size','RMSE h','RMSE u','RMSE u0','RMSE u1','RMSE alpha','Num NegDef']
            
            baseList = []
            
            for name in saveNames:
                newRow = [name,*[0 for x in cols[1:]]]
                baseList.append(newRow)
            
            baseFrame = pd.DataFrame(baseList,columns = cols)
            
            baseFrame = baseFrame.transpose()
            baseFrame.columns = baseFrame.iloc[0]
            baseFrame = baseFrame.drop(baseFrame.index[0])
            with open(filePath,'wb') as handle:
                pickle.dump(baseFrame,handle)
        else:
            print('No new frame')


if __name__ == "__main__":
    N = 1
    Q = getquad('lgwt', 10, -1, 1, N)
    epsilon = 0.03
    DataClass = MN_Data(N, Q, 'M_N')
    # DataClass.make_train_data('uniform', epsilon, [-100, 100, 10], [-100, 100, 20])
    #DataClass.make_test_data('uniform', [1, 3, 10], [-10, 10, 100])
    DataClass.make_test_data_wrapper('uniform',[1e-8,8,100],[-65,65,int(1e+04)])
    