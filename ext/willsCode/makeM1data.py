# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:32:57 2020

@author: Will
"""


import numpy as np 
#from MN_Duality_Tools import Entropy, moment_vector
import pickle 
import pandas as pd
from tabulate import tabulate
from fobj import alpha0surface
import pickle 
import os 
from DualityTools import DualityTools 


pd.set_option('display.float_format', '{:.2e}'.format)

#dualfixedpoly1Q_opts is the optimization which solves for \hat{\alpha}(u)
#from dual objective function 

#from dualfixedpoly1Q_opts import dualfixedpoly1Q_opts
#from optstuffset import optstuffset
#from getquad import getquad 

class make_MN_data:
    
    def moment_vector(a,tol = 1e-10):
        """

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
    
    def entropy(a,tol = 1e-10):
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
    
    def __init__(self,N,Quad = None,savedata = False):
        #Create emtpy containers for the data we will make. Starting with lists. 
        #We will vectorize operations and do in numpy later. 
        
        #getDualityTools needs to be build 
        
        self.N = N
        self.Quad = Quad
        
        self.dualityTools = getDualityTools(N,Quad)
            
        pass
        
    def make_trainData(self,trainParams,N = self.N):
        
        alpha_list = []
        entropy_list = []
        moment_list = []
        
        if N ==1:
            
            #UnPack the Training Pararmeters 
            self.train_N, self.train_alpha1_min, self.train_alpha1_max = trainParams
            
            #Make uniform domain to sample over 
            self.alpha1_mesh = np.random.uniform(self.train_alpha1_min,self.train_alpha1_max,self.train_N)
            
            for i in range(self.train_N):
                #Draft in loop format and vectorize later 
                alpha1 = self.alpha1_meshalpha1_mesh[i]
                
                alpha0 = self.dualityTools.alpha0surface(alpha1,1)
                
                alpha_point = np.array([alpha0,alpha1])
                
                alpha_list.append([alpha0,alpha1])
                
                moment_point = self.moment_vector(alpha_point)
                
                moment_list.append([moment_point[0],moment_point[1]])
                
                entropy_list.append([self.entropy(alpha_point)])
                
            alpha_data  = np.array(alpha_list,    dtype = float)
            moment_data = np.array(moment_list,   dtype= float)
            entropy_data = np.array(entropy_list, dtype =float)
                
        elif N  > 1:
            
            pass 
        
    def make_testData(self,testParams,N = self.N):
        
        alpha_list = []
        entropy_list = []
        moment_list = []
        
        if N == 1:
            
            self.test_N_alpha1, self.test_alpha1_min, self.test_alpha1_max = testParams[0]
            self.test_N_u0, self.test_u0_min, self.test_alpha1_max = testParams[1]
            
            for i in range(N_alpha1):
                #For each value of alpha_1, alpha_0 is monotonically increasing in u_0. 
                #This can be seen from the N = 1,d = 1 elementary expression for the reconstruction map
                
                alpha0 = self.dualityTools.alpha0surface(alpha1_mesh[i],1)
                
                alpha_projected = np.array([alpha0,alpha1_mesh[i]],dtype = float)
                
                u0_scale_add = np.array([0,0],dtype = float)
                
                #Print i just to give us a picture of how far along a large iteration is
                if (i % 100) == 0:
                    print('\n Make 2d Test Has Elapsed', i)
                
                for j in range(N_u0):
                    
                    u0_scale_add[0] = np.log(u0_vals[j])
                    
                    alpha_scaledup = alpha_projected + u0_scale_add
                    
                    alpha_list.append([alpha_scaledup[0],alpha_scaledup[1]])
                    
                    moment_scaledup = self.moment_vector(alpha_scaledup)
                    
                    moment_list.append([moment_scaledup[0],moment_scaledup[1]])
                    
                    entropy_list.append(self.entropy(alpha_scaledup))
                    
            alpha_data  = np.array(alpha_list,    dtype = float)
            moment_data = np.array(moment_list,   dtype= float)
            entropy_data = np.array(entropy_list, dtype =float)
        
                    
        elif N > 1:
            
            
            pass


#Determine whether or not to save data to filepath 
if __name__ == "__main__":
    
    savedata = True 
    
    #The 'name' you want to give the datafile; it will be automnatically appended to a filepath
    
    #Most used network = 'M1_set_A'
    
    saveid = 'NoUse500'
    
    #The 'name' you want to give the datafile; it will be automnatically appended to a filepath
    #For networks, the important already used saveid is 'M1_set_A'
    
    
    #For splines, the data is train only, and saveid is of the form Spline%N where N is number of points 
    #for train and test respectively;
    
    #For networks, saveid is of the form 'M1_set_A'
    
    #Mode can be: 'Train' or '1dTest' or '2dTest'
    
    mode = 'Train'
    
    #Any way to store this data in a list or group together, and not in separate .csv? Or should it be separate csv?
    
     #Use os.path to get your parent_dir and the data folder in an OS-free way
    alpha_filename = 'M1' + mode+'Data'+saveid  + '_gradient.csv'
    entropy_filename = 'M1'+mode+'Data'+saveid + '_entropy.csv'
    moment_filename = 'M1'+mode + 'Data'+saveid + '_moment.csv'
    table_filename = 'M1'+mode+'DataTable'+saveid  + '.pickle'

    #Use os.path to get your parent_dir and the data folder in an OS-free way
    parent_dir = os.path.abspath('../..')
    data_folder = os.path.join(parent_dir,'data')
    
    alpha_datafile = os.path.join(data_folder,alpha_filename)
    entropy_datafile = os.path.join(data_folder,entropy_filename)
    moment_datafile = os.path.join(data_folder,moment_filename)
    table_file= os.path.join(data_folder,table_filename)
    
    
    if mode == 'Train':
        
        alpha1_min = -65
        alpha1_max = 65
        N_alpha1 = int(500)
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
        
        #datalist = [alpha_data,moment_data,entropy_data]
        
        alpha1_edges  = [alpha1_min,alpha1_max]
        
        domain_cols= ['N_alpha1','alpha_1 edges','Max-Ratio']
        domain_info = [[N_alpha1,tuple(alpha1_edges),bdratio]]
        domain_table = pd.DataFrame(domain_info,columns = domain_cols)
        
        print('\n\n',mode+' Domain Error Summary (excluding validation) in Latex','\n\n',\
              domain_table.to_latex())
        print('\n\n',mode +' Domain Info: \n\n',\
              tabulate(domain_table,headers = 'keys',tablefmt = 'psql'))
        
        if savedata == True:
            #Add file extensions for separate data files after determining path and filename 
            
            np.savetxt(alpha_datafile,alpha_data, delimiter = ',')
            np.savetxt(moment_datafile,moment_data, delimiter = ',')
            np.savetxt(entropy_datafile,entropy_data, delimiter = ',')
            with open(table_file,'wb') as handle:
                pickle.dump(domain_table,handle)
            
            """
            #Previous way I did this on my device using .pickle files. Inefficient probably.
                
            with open(datafile + '.pickle','wb') as file_handle:
                pickle.dump(datalist,file_handle)
            with open(tablefile + '.pickle','wb') as tablefile_handle:
                pickle.dump(domain_table,tablefile_handle)
            """
                
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
       
        #datalist = [alpha_data,moment_data,entropy_data]
    
        domain_cols = ['N_alpha1','alpha1 Edges','Max Ratio']
        domain_info = [[N_alpha1,tuple([alpha1_min,alpha1_max]),bdratio]]
        domain_table = pd.DataFrame(domain_info,columns = domain_cols)
            
        print('\n\n',mode+' 1d Test Domain Info (In Latex)','\n\n',domain_table.to_latex())
        print('\n\n',mode +' 1d Test Domain Info: \n\n',\
              tabulate(domain_table,headers = 'keys',tablefmt = 'psql'))
        
        if savedata == True:
            
            np.savetxt(alpha_datafile,alpha_data, delimiter = ',')
            np.savetxt(moment_datafile,moment_data, delimiter = ',')
            np.savetxt(entropy_datafile,entropy_data, delimiter = ',')
            with open(table_file,'wb') as handle:
                pickle.dump(domain_table,handle)
            
            """
            with open(datafile + '.pickle','wb') as file_handle:
                pickle.dump(datalist,file_handle)
            with open(tablefile + '.pickle','wb') as tablefile_handle:
                pickle.dump(domain_table,tablefile_handle)
            """
            
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
            
            np.savetxt(alpha_datafile,alpha_data, delimiter = ',')
            np.savetxt(moment_datafile,moment_data, delimiter = ',')
            np.savetxt(entropy_datafile,entropy_data, delimiter = ',')
            with open(table_file,'wb') as handle:
                pickle.dump(domain_table,handle)
            
            """
            with open(datafile + '.pickle','wb') as file_handle:
                pickle.dump(datalist,file_handle)
            with open(tablefile + '.pickle','wb') as tablefile_handle:
                pickle.dump(domain_table,tablefile_handle)
            """