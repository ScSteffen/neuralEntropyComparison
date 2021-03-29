# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:32:57 2020

@author: Will
"""

"""
1. Define a uniform mesh of alpha_1, and of u_0, and use the following approach to obtain a 
sampling of the realizable domain:

Each value of alpha_1 determines a ray in the relizable domain with slope r = (1/tanh(alpha1))-alpha1. Define a uniform sequence of
alpha_0 which causes u_0 to interpolate between u_0 min and u_0 max along the ray determined by slope r.

                        D = { (u_0^{i,j},u_1^{i,j}) | where ... }
"""


import numpy as np 
from MN_Duality_Tools import Entropy, moment_vector
import pickle 
import pandas as pd
from tabulate import tabulate
from fobj import alpha0surface
from dualfixedpoly1Q_opts import dualfixedpoly1Q_opts
from optstuffset import optstuffset
from getquad import getquad 


savedata = True 

saveid = 'Spline162'

#For splines, saveid is one of {'spline10216','splinetest'} for train and test respectively;
#For networks, saveid is one of ('2','2') for train and test respectively 

#Mode = 'Train' or '1dTest' or '2dTest'
mode = 'Train'
#Standard is M1_set_A

datafile = 'M1'+mode+'Data'+saveid

tablefile = 'M1'+mode+'DataTable'+saveid

if mode == 'Train':
    
    alpha1_min = -65
    alpha1_max = 65
    N_alpha1 = int(162)
    #For: M1_set_A int(1*(10**4))

    alpha1_mesh,alpha1_step = np.linspace(alpha1_min,alpha1_max,N_alpha1,retstep = True)
    
    bdratio = 1/(np.tanh(alpha1_max)) - 1/(alpha1_max)
    print(bdratio)
    
if mode == '1dTest':
    """
    We take the same smallest/largest u1-moment components on the 1d Training domain, 
    and sample with a uniform distribution of moments between these two points. 
    
    This will give a somewhat unbiased test domain within the convex-body of the training points
    
    We choose the density to be at least 10-times that of the training domain (should be fine enough!)
    
    Note this requires us to compute the corresponding alpha and entropy values via 
    dualfixedpoly1Q_opts. 
    """
    """
    nQuadpts = 10
    Quad = getquad('lgwt',nQuadpts,-1,1,1)
    """
    
    alpha1_min = -65
    alpha1_max = 65
    N_alpha1 = int(1*(10**5))
    alpha1_mesh = np.random.uniform(alpha1_min,alpha1_max,N_alpha1)

    #N_u1 = int(5.20*(10**4))
    
    """
    alpha0_min_pair = alpha0surface(alpha1_min,1)
    alpha0_max_pair = alpha0surface(alpha1_max,1)
    
    alpha_lower_bd = np.array([alpha0_min_pair,alpha1_min])
    alpha_upper_bd = np.array([alpha0_max_pair,alpha1_max])

    moment_lower_bd= moment_vector(alpha_lower_bd)
    moment_upper_bd= moment_vector(alpha_upper_bd) 

    u1_lower_bd = moment_lower_bd[1]
    u1_upper_bd = moment_upper_bd[1] 
    print('U1 Test bounds: ',u1_lower_bd,u1_upper_bd)

    u1_test_mesh = np.random.uniform(u1_lower_bd,u1_upper_bd,size = N_u1)
    """
    
    bdratio = 1/(np.tanh(alpha1_max)) - 1/(alpha1_max)
    print(bdratio)
    #u1_test_mesh = np.random.uniform(u1_lower_bd,u1_upper_bd,N_1dtest)
    

#If we want to test moments, we want a fully 2d set 
if mode == '2dTest':
    
    alpha1_min = -65
    alpha1_max = 65
    N_alpha1 = int(5.20*(10**4))
    
    u0_min = 1e-8
    u0_max = 8
    N_u0 = 160
    
    """
    #Alternate test-domain strategy
    alpha0_min = np.zeros((N_alpha1,))
    alpha0_max = np.zeros((N_alpha1,))
    """
    
    #Set for test domain near zero
    test_tol = 1e-8
    
    alpha1_mesh = np.random.uniform(alpha1_min,alpha1_max,size = N_alpha1)
    
    bdratio = 1/(np.tanh(alpha1_max)) - 1/(alpha1_max)
    print(bdratio)
    

alpha_list = []
entropy_list = []
moment_list = []

#Generate the data 

if mode == 'Train':
    
    for i in range(N_alpha1):
        
        alpha1 = alpha1_mesh[i]
        
        alpha0 = alpha0surface(alpha1,1)
        
        alpha_point = np.array([alpha0,alpha1])
        
        alpha_list.append([alpha0,alpha1])
        
        moment_point = moment_vector(alpha_point)
        
        moment_list.append([moment_point[0],moment_point[1]])
        
        entropy_list.append([Entropy(alpha_point)])
        
    #The data should have u_0 = 1 for all points, so is essentially 1 dimensional 
        
    alpha_data  = np.array(alpha_list,    dtype = float)
    moment_data = np.array(moment_list,   dtype= float)
    entropy_data = np.array(entropy_list, dtype =float)
    
    datalist = [alpha_data,moment_data,entropy_data]
    
    alpha1_edges  = [alpha1_min,alpha1_max]
    
    domain_cols= ['N_alpha1','alpha_1 edges','Max-Ratio']
    domain_info = [[N_alpha1,tuple(alpha1_edges),'{:.2e}'.format(bdratio)]]
    domain_table = pd.DataFrame(domain_info,columns = domain_cols)
    
    print('\n\n',mode+' Domain Error Summary (excluding validation) in Latex','\n\n',domain_table.to_latex())
    #mpltable(error_table,width = 0.12,savedir = savefolder + method + 'firsterrtable' + saveid + '.pdf')
    print('\n\n',mode +' Domain Info: \n\n',\
          tabulate(domain_table,headers = 'keys',tablefmt = 'psql'))
    
    if savedata == True:
        with open(datafile + '.pickle','wb') as file_handle:
            pickle.dump(datalist,file_handle)
        with open(tablefile + '.pickle','wb') as tablefile_handle:
            pickle.dump(domain_table,tablefile_handle)
            
            
if mode == '1dTest':
    
    """
    For: uinform in u1 sampling strategy 
    
    moment_list_closed = []
    
    optstuff = optstuffset(1) 
    """
    
    for i in range(N_alpha1):
        
        alpha1 = alpha1_mesh[i]
        
        alpha0 = alpha0surface(alpha1,1)
        
        alpha_point = np.array([alpha0,alpha1])
        
        alpha_list.append([alpha0,alpha1])
        
        moment_point = moment_vector(alpha_point)
        
        moment_list.append([moment_point[0],moment_point[1]])
        
        entropy_list.append([Entropy(alpha_point)])
        
        """
        For: uniform in u1 sampling strategy
        
        moment_point = np.array([1.,u1_test_mesh[i]],dtype= float)
        
        moment_list.append([moment_point[0],moment_point[1]])
        
        alpha_point,opt_info,toss = dualfixedpoly1Q_opts('M_N',moment_point,None,optstuff,Quad)
    
        moment_point_closed = moment_vector(alpha_point)
        
        alpha_point[0] -= np.log(moment_point_closed[0])
        
        moment_point_closed = moment_vector(alpha_point)
        
        alpha_list.append([alpha_point[0],alpha_point[1]])
        
        moment_list_closed.append([moment_point_closed[0],moment_point_closed[1]])
        
        entropy_list.append(Entropy(alpha_point))
        """
        
    alpha_data  = np.array(alpha_list,    dtype = float)
    moment_data = np.array(moment_list,   dtype= float)
    entropy_data = np.array(entropy_list, dtype =float)
    """
    For: uniform in u1 sampling strategy
    moment_data_closed = np.array(moment_list_closed, dtype = float)
    
    datalist = [alpha_data,moment_data_closed,entropy_data]
    """
    datalist = [alpha_data,moment_data,entropy_data]
    
    #here we store the closed moments only- this way the data is self-consistent to macihne precision
    """
    For: uniform in u1 sampling strategy
    u1_upper_closed = np.max(moment_data_closed[:,1])
    u1_lower_closed = np.min(moment_data_closed[:,1])
    
    #num_invalid = np.sum((moment_data_closed[:,0] > 1) + (moment_data_closed[:,0] < 1))
    #print(num_invalid)
    validity = np.allclose(moment_data_closed[:,0],np.ones((moment_data_closed.shape[0],)))
    print("u_0 equiv 1 ? ",validity)
    """
    
    """
    For: uniform in u1 sampling strategy
    
    domain_cols= ['N_u1','u_1 Edges','Max-Ratio']
    domain_info = [[N_u1,('{:.2e}'.format(u1_lower_closed),'{:.2e}'.format(u1_upper_closed)),\
                    '{:.2e}'.format(max([u1_upper_closed,abs(u1_lower_closed)]))]]
    """
    
    domain_cols = ['N_alpha1','alpha1 Edges','Max Ratio']
    domain_info = [[N_alpha1,tuple([alpha1_min,alpha1_max]),bdratio]]
    domain_table = pd.DataFrame(domain_info,columns = domain_cols)
    
    
    """
    domain_cols = ['N_alpha1',r'$alpha_1$ Interior Edges',r'$u_1$ Interior Edges',\
                   r'$alpha_1$ Boundary Edges',r'$u_1$ Symmetric Boundary Edges','Max ratio']
    domain_info = [[N_alpha1,('{:.2e}'.format(alpha1_min),'{:.2e}'.format(alpha1_max)),bdratio]]
    domain_table = pd.DataFrame(domain_info,columns = domain_cols)
    """
    
    print('\n\n',mode+' 1d Test Domain Info (Latex)','\n\n',domain_table.to_latex())
    #mpltable(error_table,width = 0.12,savedir = savefolder + method + 'firsterrtable' + saveid + '.pdf')
    print('\n\n',mode +' 1d Test Domain Info: \n\n',\
          tabulate(domain_table,headers = 'keys',tablefmt = 'psql'))
    
    if savedata == True:
        with open(datafile + '.pickle','wb') as file_handle:
            pickle.dump(datalist,file_handle)
        with open(tablefile + '.pickle','wb') as tablefile_handle:
            pickle.dump(domain_table,tablefile_handle)
    
if mode == '2dTest':
    
    u0_vals = np.linspace(u0_min,u0_max,N_u0)
    
    for i in range(N_alpha1):
        #For each value of alpha_1, alpha_0 is monotonically increasing in u_0. 
        #This can be seen clearly from the duality relation.
        
        alpha0 = alpha0surface(alpha1_mesh[i],1)
        
        alpha_projected = np.array([alpha0,alpha1_mesh[i]],dtype = float)
        
        u0_scale_add = np.array([0,0],dtype = float)
        
        print(i)
        
        for j in range(N_u0):
            
            u0_scale_add[0] = np.log(u0_vals[j])
            
            alpha_scaledup = alpha_projected + u0_scale_add
            
            alpha_list.append([alpha_scaledup[0],alpha_scaledup[1]])
            
            moment_scaledup = moment_vector(alpha_scaledup)
            
            moment_list.append([moment_scaledup[0],moment_scaledup[1]])
            
            entropy_list.append(Entropy(alpha_scaledup))
            
        """
        if abs(alpha1_mesh[i]) < test_tol:
        
            alpha0_min[i] = -np.log(2/u0_min)
        
            alpha0_max[i] = -np.log(2/u0_max)
        
        else:
    
            alpha0_min[i] = -np.log(np.divide(2*np.sinh(alpha1_mesh[i]),u0_min*alpha1_mesh[i]))
        
            alpha0_max[i] = -np.log(np.divide(2*np.sinh(alpha1_mesh[i]), u0_max*alpha1_mesh[i]))
            

        -np.log( np.divide(2*np.sinh(alpha[1-alpha_null,1:]), alpha[1-alpha_null,1:]) )
        
            a0_out = -np.log(2) 
        else:
            
            a0_out = -np.log(np.divide(2*np.sinh(alpha), alpha))

        alpha0_mesh = np.linspace(alpha0_min[i],alpha0_max[i],N_u0)
        
        m = alpha0_min[i]
        M = alpha0_max[i] 
        
        if m >= M:
            
            print('\n\n','makeM1data.py data attribute alpha0_min vec failed at index: ',i,'\n\n')
        """
 
    alpha_data  = np.array(alpha_list,    dtype = float)
    moment_data = np.array(moment_list,   dtype= float)
    entropy_data = np.array(entropy_list, dtype =float)
    
    datalist = [alpha_data,moment_data,entropy_data]
    
    alpha1_edges  = ['{:.2e}'.format(alpha1_min),'{:.2e}'.format(alpha1_max)]
    u0_edges = ['{:.2e}'.format(u0_min),'{:.2e}'.format(u0_max)]
    
    domain_cols= ['(N_u0,N_alpha1)','alpha_1 edges','u0 edges','Max-Ratio']
    domain_info = [[(N_u0,N_alpha1),tuple(alpha1_edges),tuple(u0_edges),'{:.3e}'.format(bdratio)]]
    domain_table = pd.DataFrame(domain_info,columns = domain_cols)
    
    print('\n\n',mode+' Domain Error Summary (excluding validation) in Latex','\n\n',domain_table.to_latex())
    #mpltable(error_table,width = 0.12,savedir = savefolder + method + 'firsterrtable' + saveid + '.pdf')
    print('\n\n',mode +' Domain Info: \n\n',\
          tabulate(domain_table,headers = 'keys',tablefmt = 'psql'))
    
    if savedata == True:
        with open(datafile + '.pickle','wb') as file_handle:
            pickle.dump(datalist,file_handle)
        with open(tablefile + '.pickle','wb') as tablefile_handle:
            pickle.dump(domain_table,tablefile_handle)
