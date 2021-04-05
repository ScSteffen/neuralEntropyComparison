# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:32:57 2020

@author: Will
"""


import numpy as np 
from MN_Duality_Tools import Entropy, moment_vector
import pickle 
import pandas as pd
from tabulate import tabulate
from fobj import alpha0surface
#dualfixedpoly1Q_opts is the optimization which solves for \hat{\alpha}(u)
#from dual objective function 
from dualfixedpoly1Q_opts import dualfixedpoly1Q_opts
from optstuffset import optstuffset
from getquad import getquad 
import os 


#Determine whether or not to save data to filepath 
savedata = True 

#The 'name' you want to give the datafile; it will be automnatically appended to a filepath

#Most used is saveid = 'M1_set_A'

saveid = 'Spline502'

#For splines, the data is train only, and saveid is of the form Spline%N where N is number of points 
#for train and test respectively;

#For networks, saveid is of the form 'M1_set_A'

#Mode can be: 'Train' or '1dTest' or '2dTest'

mode = 'Train'

#The filepath for the datafile, and the 'tablefile' (we save a small table to summarize)
#the data instead of a pandas header & dataframe. This avoids extra value of dataframe. 
datafile = 'M1'+mode+'Data'+saveid

tablefile = 'M1'+mode+'DataTable'+saveid

if mode == 'Train':
    
    alpha1_min = -65
    alpha1_max = 65
    N_alpha1 = int(502)
    #For saveid = M1_set_A, N = int(1*(10**4))

    alpha1_mesh,alpha1_step = np.linspace(alpha1_min,alpha1_max,N_alpha1,retstep = True)
    
    bdratio = 1/(np.tanh(alpha1_max)) - 1/(alpha1_max)
    print(bdratio)
    
if mode == '1dTest':
    """
    We take the same boundary for alpha_1 components on the 1d Training domain as 
    with the test data, and sample with a uniform distribution between
    these two points. 
    
    This will give a somewhat 'unbiased' test domain within the convex-body of 
    the training points.
    
    We choose the density to be at least 10-times that of the training domain. While 
    this provides some density of points in the region where \hat{u}(\alpha) behavior
    is truly exponential, we need to strongly 'check' the model behavior in this region
    for error. 
    """
    
    
    alpha1_min = -65
    alpha1_max = 65
    N_alpha1 = int(1*(10**5))
    alpha1_mesh = np.random.uniform(alpha1_min,alpha1_max,N_alpha1)
    
    #bdratio is the maximum ratio of (u_1 / u_0) produced by this sampling
    bdratio = 1/(np.tanh(alpha1_max)) - 1/(alpha1_max)
    print(bdratio)
    

#If we want to test moments, we want a fully 2d set 
if mode == '2dTest':
    
    alpha1_min = -65
    alpha1_max = 65
    N_alpha1 = int(5.20*(10**4))
    
    u0_min = 1e-8
    u0_max = 8
    N_u0 = 160
    
    #Set tolerance for alpha_1 near zero calculation, since formula sinh(\alpha_1)/alpha_1 
    #needs to be directly convergted to its limiting value of 1
    
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
    
    print('\n\n',mode+' Domain Error Summary (excluding validation) in Latex','\n\n',\
          domain_table.to_latex())
    print('\n\n',mode +' Domain Info: \n\n',\
          tabulate(domain_table,headers = 'keys',tablefmt = 'psql'))
    
    if savedata == True:
        with open(datafile + '.pickle','wb') as file_handle:
            pickle.dump(datalist,file_handle)
        with open(tablefile + '.pickle','wb') as tablefile_handle:
            pickle.dump(domain_table,tablefile_handle)
            
            
if mode == '1dTest':
    
    for i in range(N_alpha1):
        
        alpha1 = alpha1_mesh[i]
        
        alpha0 = alpha0surface(alpha1,1)
        
        alpha_point = np.array([alpha0,alpha1])
        
        alpha_list.append([alpha0,alpha1])
        
        moment_point = moment_vector(alpha_point)
        
        moment_list.append([moment_point[0],moment_point[1]])
        
        entropy_list.append([Entropy(alpha_point)])

        
    alpha_data  = np.array(alpha_list,    dtype = float)
    moment_data = np.array(moment_list,   dtype= float)
    entropy_data = np.array(entropy_list, dtype =float)
   
    datalist = [alpha_data,moment_data,entropy_data]

    domain_cols = ['N_alpha1','alpha1 Edges','Max Ratio']
    domain_info = [[N_alpha1,tuple([alpha1_min,alpha1_max]),bdratio]]
    domain_table = pd.DataFrame(domain_info,columns = domain_cols)
        
    print('\n\n',mode+' 1d Test Domain Info (In Latex)','\n\n',domain_table.to_latex())
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
        #This can be seen from the N = 1,d = 1 elementary expression for the reconstruction map
        
        alpha0 = alpha0surface(alpha1_mesh[i],1)
        
        alpha_projected = np.array([alpha0,alpha1_mesh[i]],dtype = float)
        
        u0_scale_add = np.array([0,0],dtype = float)
        
        #Print i just to give us a picture of how far along a large iteration is
        if (i % 100) == 0:
            print('\n Make 2d Test Has Elapsed', i)
        
        for j in range(N_u0):
            
            u0_scale_add[0] = np.log(u0_vals[j])
            
            alpha_scaledup = alpha_projected + u0_scale_add
            
            alpha_list.append([alpha_scaledup[0],alpha_scaledup[1]])
            
            moment_scaledup = moment_vector(alpha_scaledup)
            
            moment_list.append([moment_scaledup[0],moment_scaledup[1]])
            
            entropy_list.append(Entropy(alpha_scaledup))
 
    alpha_data  = np.array(alpha_list,    dtype = float)
    moment_data = np.array(moment_list,   dtype= float)
    entropy_data = np.array(entropy_list, dtype =float)
    
    datalist = [alpha_data,moment_data,entropy_data]
    
    alpha1_edges  = ['{:.2e}'.format(alpha1_min),'{:.2e}'.format(alpha1_max)]
    u0_edges = ['{:.2e}'.format(u0_min),'{:.2e}'.format(u0_max)]
    
    domain_cols= ['(N_u0,N_alpha1)','alpha_1 edges','u0 edges','Max-Ratio']
    domain_info = [[(N_u0,N_alpha1),tuple(alpha1_edges),tuple(u0_edges),'{:.3e}'.format(bdratio)]]
    domain_table = pd.DataFrame(domain_info,columns = domain_cols)
    
    print('\n\n',mode+' Domain Error Summary (excluding validation) in Latex','\n\n',\
          domain_table.to_latex())
    print('\n\n',mode +' Domain Info: \n\n',\
          tabulate(domain_table,headers = 'keys',tablefmt = 'psql'))
    
    if savedata == True:
        with open(datafile + '.pickle','wb') as file_handle:
            pickle.dump(datalist,file_handle)
        with open(tablefile + '.pickle','wb') as tablefile_handle:
            pickle.dump(domain_table,tablefile_handle)
