# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 18:02:41 2020

@author: Will
"""

import pickle 
import os 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from tabulate import tabulate 
from getquad import getquad
from fobj import fobj, momentvector_quad,alpha0surface,IsRealizableM2, entropy_quad


"""
1. Choose training / test data info
"""

if __name__ == "__main__":

    savedata = True
    
    mode = 'Train'
    
    saveid = 'NoUse500'
    
    
    alpha_filename =   'M2'  +  mode + 'Data'      +saveid  + '_gradient.csv'
    entropy_filename = 'M2'  +  mode + 'Data'      +saveid + '_entropy.csv'
    moment_filename =  'M2'  +  mode + 'Data'     +saveid + '_moment.csv'
    table_filename =   'M2'  +  mode + 'DataTable' +saveid  + '.pickle'
    
    #Use os.path to get your parent_dir and the data folder in an OS-free way
    parent_dir = os.path.abspath('../..')
    data_folder = os.path.join(parent_dir,'data')
    
    alpha_datafile = os.path.join(data_folder,alpha_filename)
    entropy_datafile = os.path.join(data_folder,entropy_filename)
    moment_datafile = os.path.join(data_folder,moment_filename)
    table_file= os.path.join(data_folder,table_filename)
    
    
    Q = getquad('lgwt',20,-1,1,2)
    
    if mode  == 'Test':
        
        u0_edges = [1e-8,8]
        
        N_u0 = 16
        
    alpha1_edges = [-10,10]
    
    alpha2_edges = [-10,10]
    
    N_alpha1 = 100
    
    N_alpha2 = 50
    
    """
    2. Generate a set of 'alpha' values, and map them to the moment values. 
    Strategy: Choose uniform sampling of alpha_1, and alpha_2 (starting with alpha_1 = 0 ?) 
    and solve for alpha_0 via the u_0 = 1 constraint:
        
        1 = integral_{[-1,1]} [ exp(alpha_0 + alpha_1 v + (alpha_2)/2 (3v^2 -1)) ] 
        <====> 
        alpha_0 = - log (integral_{[-1,1]} [ exp(alpha_1 v + (alpha_2 / 2)(3v^2 -1) ) ] )
    """ 
    
    alpha1_mesh = np.linspace(alpha1_edges[0],alpha1_edges[-1],N_alpha1)
    alpha2_mesh = np.linspace(alpha2_edges[0],alpha2_edges[-1],N_alpha2)
    
    alpha1_vals,alpha2_vals = np.meshgrid(alpha1_mesh,alpha2_mesh)
    alpha0_vals = np.zeros((N_alpha1,N_alpha2))
    
    #Realizable = np.zeros((N_alpha1,N_alpha2))
    
    moment_list = []
    alpha_list = []
    entropy_list = []
    
    if mode == 'Train':
        
        for i in range(N_alpha1):
            for j in range(N_alpha2):
                
                alpha_input = np.array([0,alpha1_vals[j,i],alpha2_vals[j,i]])
                alpha0_vals[i,j] = alpha0surface(alpha_input,2,Q)
                alpha_calc = np.array([alpha0_vals[i,j],alpha1_vals[j,i],alpha2_vals[j,i]])
                
                if IsRealizableM2(alpha_calc,Q):
                    pass
                else:
                    print(alpha_calc,' not realizable at index ',(i,j))
                    
                entropy_list.append(entropy_quad(alpha_calc,Q))
                alpha_list.append([alpha_calc[0],alpha_calc[1],alpha_calc[2]])
                u0,u1,u2 = momentvector_quad(alpha_calc,Q)[:]
                moment_list.append([u0,u1,u2])
        
        moment_data = np.array(moment_list,dtype = float)    
        alpha_data  = np.array(alpha_list,dtype = float)
        entropy_data = np.array(entropy_list,dtype = float)
        
        boundary_dist = ['{:.2e}'.format(min([np.linalg.norm(x - np.array([0,-1/2])) for x in moment_data[:,1:]])),\
                         '{:.2e}'.format(min([np.linalg.norm(x - np.array([1/np.sqrt(3),0])) for x in moment_data[:,1:]])),\
                         '{:.2e}'.format(min([np.linalg.norm(x - np.array([1,1])) for x in moment_data[:,1:]])),\
                         '{:.2e}'.format(min([np.linalg.norm(x - np.array([0,1])) for x in moment_data[:,1:]]))]
        
        table_cols = ['(N-alpha1,N-alpha2)','alpha-1 Edges','alpha-2 Edges','BD-Dist: [0,-1/2],[1/sqrt(3)],[1,1],[0,1]']
        table_vals = [[(N_alpha1,N_alpha2),tuple(alpha1_edges),tuple(alpha2_edges),tuple(boundary_dist)]]
        
        data_table = pd.DataFrame(table_vals,columns = table_cols)
        
        print('\n\n',mode+' Domain Info Summary LaTeX: \n', data_table.to_latex(),'\n')
        print('\n\n',mode+' Domain Info Summary:','\n', tabulate(data_table,headers = 'keys',tablefmt = 'psql'))
        
    if mode == 'Test':
        
        u0_vals = np.linspace(u0_edges[0],u0_edges[-1],N_u0)
        
        for i in range(N_alpha1):
            for j in range(N_alpha2):
                
                alpha_start = np.array([0,alpha1_vals[j,i],alpha2_vals[j,i]])
                alpha0_vals[i,j] = alpha0surface(alpha_input,2,Q)
                alpha_projected = np.array([alpha0_vals[i,j],alpha1_vals[j,i],alpha2_vals[j,i]])
                
                for k in range(N_u0):
                
                    alpha_scaledup = alpha_projected + np.array([np.log(u0_vals[k]),0,0])
                    
                    if IsRealizableM2(alpha_scaledup,Q):
                        pass 
                    else: 
                        print(alpha_calc,' not realizable at index ',(i,j,k))
                        
                    entropy_list.append(entropy_quad(alpha_scaledup,Q))
                    alpha_list.append([alpha_scaledup[0],alpha_scaledup[1],alpha_scaledup[2]])
                    u0,u1,u2 = momentvector_quad(alpha_scaledup,Q)[:]
                    moment_list.append([u0,u1,u2])
                    
        moment_data = np.array(moment_list,dtype = float)
        alpha_data = np.array(alpha_list,dtype = float)
        entropy_data =  np.array(entropy_list,dtype = float)
        
       
        table_cols = ['(N u0,N alpha1,N alpha2)','alpha-1 Edges','alpha-2 Edges',]
        table_vals = [[(N_u0,N_alpha1,N_alpha2),tuple(alpha1_edges),tuple(alpha2_edges)]]
        
        data_table = pd.DataFrame(table_vals,columns = table_cols)
        
        print('\n\n',mode+' Domain Info Summary LaTeX: \n', data_table.to_latex(),'\n')
        print('\n\n',mode+' Domain Info Summary:','\n', tabulate(data_table,headers = 'keys',tablefmt = 'psql'))
    
    plt.scatter(moment_data[:,1],moment_data[:,2])
    plt.xlabel('u_1')
    plt.ylabel('u_2')
    plt.title('Moment Data at {u_0 = 1}')
    plt.show()
    
    plt.scatter(alpha_data[:,1],alpha_data[:,2])
    plt.xlabel('alpha_1')
    plt.ylabel('alpha_2')
    plt.title('Alpha Data for {u_0 = 1}')
    plt.show()

    
    if savedata == True:
        np.savetxt(alpha_datafile,alpha_data, delimiter = ',')
        np.savetxt(moment_datafile,moment_data, delimiter = ',')
        np.savetxt(entropy_datafile,entropy_data, delimiter = ',')
        with open(table_file,'wb') as handle:
            pickle.dump(data_table,handle)
            
            


