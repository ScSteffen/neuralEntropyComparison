# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 11:28:30 2021

@author: Will
"""

import numpy as np 
import pandas as pd 
pd.set_option('display.float_format', '{:.2e}'.format)
import pickle 
from tabulate import tabulate 
import matplotlib.pyplot as plt


"""
1. Table for Training Data Errors of Interest: RMSE
"""


def make_M1_summary_fromdata(net_file_dict,spline_file_dict,net_select_ids,\
                spline_select_ids,savedir,**kwargs):
    
    #Produce a single final dataframe for the networks which contains
    #the relevant information for table 1 - we will fix precise formatting in LaTeX
    #but want a correct data block here
    
    with open(net_file_dict["Train"],'rb') as handle:
        net_train_list = pickle.load(handle)
    with open(net_file_dict["1dTest"],'rb') as handle:
        net_1d_list = pickle.load(handle)
    with open(net_file_dict["2dTest"],'rb') as handle:
        net_2d_list = pickle.load(handle)
      
    
    net_columns = ['SaveId','Size','h','u','u0','u1','alpha']
    df_net_train = pd.DataFrame(net_train_list,\
                             columns = [*net_columns[0:2],*['Train '+x for x in net_columns[2:]]],dtype = str)
    df_net_1d = pd.DataFrame(net_1d_list,\
                             columns = [*net_columns[0:2],*['1dTest '+x for x in net_columns[2:]]],dtype = str)
    df_net_2d = pd.DataFrame(net_2d_list,\
                             columns = [*net_columns[0:2],*['2dTest '+x for x in net_columns[2:]]],dtype = str)
    
    for x in net_columns[2:]:
        df_net_train['Train '+x] = df_net_train['Train '+x].astype(float)
        df_net_1d['1dTest '+x] = df_net_1d['1dTest '+x].astype(float)
        df_net_2d['2dTest '+x] = df_net_2d['2dTest '+x].astype(float)

    df_net_train_select = df_net_train[df_net_train['SaveId'].isin(net_select_ids)]
    df_net_train_select = df_net_train_select.drop_duplicates()
    df_net_train_select = df_net_train_select.transpose()
    df_net_train_select.columns = df_net_train_select.iloc[0] 
    df_net_train_select = df_net_train_select.drop(df_net_train_select.index[0])
    
    df_net_1d_select = df_net_1d[df_net_1d['SaveId'].isin(net_select_ids)]
    df_net_1d_select = df_net_1d_select.drop(columns = ['Size'])
    df_net_1d_select = df_net_1d_select.drop_duplicates()
    df_net_1d_select = df_net_1d_select.transpose()
    df_net_1d_select.columns = df_net_1d_select.iloc[0]
    df_net_1d_select = df_net_1d_select.drop(df_net_1d_select.index[0])
    
    df_net_2d_select = df_net_2d[df_net_2d['SaveId'].isin(net_select_ids)]
    df_net_2d_select = df_net_2d_select.drop(columns = ['Size'])
    df_net_2d_select = df_net_2d_select.drop_duplicates()
    df_net_2d_select = df_net_2d_select.transpose()
    df_net_2d_select.columns = df_net_2d_select.iloc[0]
    df_net_2d_select = df_net_2d_select.drop(df_net_2d_select.index[0])

    #print(df_net_2d_select,df_net_1d_select,df_net_train_select)
    #print(df_net_train_select,df_net_1d_select,df_net_2d_select)
    match_names = ['A','B','C','D','E','F',\
                   'G','H','I','K','L','M','N']
    df_net_all = pd.concat([df_net_train_select,df_net_1d_select,df_net_2d_select],ignore_index = False)
#    df_net_all.columns = match_names
    
    """
    for x in df_net_all.columns:
        df_net_all[x] = df_net_all[x].astype(float)
    """
#    print(df_net_all)
    if kwargs['mode'] == 'table':
        print(df_net_all.to_latex())
        print(tabulate(df_net_all,headers = 'keys',tablefmt = 'psql',floatfmt = '.2e'))
    

    """
    #Splines: roduce a single final dataframe for the splines which contains
    #the relevant information for table 1 - we will fix precise formatting in LaTeX
    #but want a correct data block here
    
    #np_spline_data = np.array(spline_data_list,dtype = str)
    
    train_net_data = pd.DataFrame(np_net_data, columns = column_labels)
    
    #df_net_data.loc[df_net_data['SaveId'].isin(net_select)]
    #pd_net_select 
    
    #pd_spline 
    """
    
    with open(spline_file_dict["Train"],'rb') as handle:
        spline_train_list = pickle.load(handle)
    with open(spline_file_dict["1dTest"],'rb') as handle:
        spline_1d_list = pickle.load(handle)
    with open(spline_file_dict["2dTest"],'rb') as handle:
        spline_2d_list = pickle.load(handle)
      
    """
    with open(spline_results_filename,'rb') as nhandle:
        spline_data_list = pickle.load(spline_results_filename,nhandle)
    """
    
    spline_columns = ['SaveId','Size','h','u','u0','u1','alpha']
    df_spline_train = pd.DataFrame(spline_train_list,\
                             columns = [*spline_columns[0:2],*['Train '+x for x in spline_columns[2:]]],dtype = str)
    df_spline_1d = pd.DataFrame(spline_1d_list,\
                             columns = [*spline_columns[0:2],*['1dTest '+x for x in spline_columns[2:]]],dtype = str)
    df_spline_2d = pd.DataFrame(spline_2d_list,\
                             columns = [*spline_columns[0:2],*['2dTest '+x for x in spline_columns[2:]]],dtype = str)
    
    for x in spline_columns[2:]:
        df_spline_train['Train '+x] = df_spline_train['Train '+x].astype(float)
        df_spline_1d['1dTest '+x] = df_spline_1d['1dTest '+x].astype(float)
        df_spline_2d['2dTest '+x] = df_spline_2d['2dTest '+x].astype(float)

    df_spline_train_select = df_spline_train[df_spline_train['SaveId'].isin(spline_select_ids)]
    df_spline_train_select = df_spline_train_select.drop_duplicates()
    df_spline_train_select = df_spline_train_select.transpose()
    df_spline_train_select.columns = df_spline_train_select.iloc[0] 
    df_spline_train_select = df_spline_train_select.drop(df_spline_train_select.index[0])
    
    df_spline_1d_select = df_spline_1d[df_spline_1d['SaveId'].isin(spline_select_ids)]
    df_spline_1d_select = df_spline_1d_select.drop(columns = ['Size'])
    df_spline_1d_select = df_spline_1d_select.drop_duplicates()
    df_spline_1d_select = df_spline_1d_select.transpose()
    df_spline_1d_select.columns = df_spline_1d_select.iloc[0]
    df_spline_1d_select = df_spline_1d_select.drop(df_spline_1d_select.index[0])
    
    df_spline_2d_select = df_spline_2d[df_spline_2d['SaveId'].isin(spline_select_ids)]
    df_spline_2d_select = df_spline_2d_select.drop(columns = ['Size'])
    df_spline_2d_select = df_spline_2d_select.drop_duplicates()
    df_spline_2d_select = df_spline_2d_select.transpose()
    df_spline_2d_select.columns = df_spline_2d_select.iloc[0]
    df_spline_2d_select = df_spline_2d_select.drop(df_spline_2d_select.index[0])

    #print(df_net_2d_select,df_net_1d_select,df_net_train_select)
    #print(df_net_train_select,df_net_1d_select,df_net_2d_select)
    #match_names = ['A','B','C','D','E','F',\
    #               'G','H','I','K','L','M']
    df_spline_all = pd.concat([df_spline_train_select,df_spline_1d_select,df_spline_2d_select],ignore_index = False)
    #df_spline_all.columns = match_names
    """
    for x in df_net_all.columns:
        df_net_all[x] = df_net_all[x].astype(float)
    """
#    print(df_net_all)
    if kwargs['mode'] == 'table':
        print(df_spline_all.to_latex())
        print(tabulate(df_spline_all,headers = 'keys',tablefmt = 'psql',floatfmt = '.2e'))
        
    if kwargs['mode'] == 'table':
        df_total  = pd.concat([df_net_all,df_spline_all],axis = 1)
        print(df_total.to_latex())
        print(tabulate(df_total,headers = 'keys',tablefmt = 'pqsl',floatfmt = '.2e'))
        
    if kwargs['mode'] == 'plot':
        
        
        net_sizes = df_net_all.loc['Size'].to_numpy(dtype = int)
        spline_sizes = df_spline_all.loc['Size'].to_numpy(dtype = int)
        
        error_keys = ['2dTest '+'u','2dTest '+'h','2dTest '+'alpha']
        error_dict = {}
        for x in error_keys:
            error_dict[x] = [df_spline_all.loc[x].to_numpy(dtype = float),\
                              df_net_all.loc[x].to_numpy(dtype = float)]
        print(error_dict)
        """
        spline_u_errors = df_spline_all.loc['2dTest '+'u'].to_numpy(dtype = float)
        net_u_errors = df_net_all.loc['2dTest '+'u'].to_numpy(dtype = float)
        
        spline_h_errors = df_spline_all.loc['2dTest '+'h'].to_numpy(dtype = float)
        net_h_errors = df_net_all.loc['2dTest '+'h'].to_numpy(dtype = float)
        
        spline_h_errors = df_spline_all.loc['2dTest '+'alpha'].to_numpy(dtype = float)
        net_h_errors = df_net_all.loc['2dTest '+'alpha'].to_numpy(dtype = float)
        """
        
        #print(spline_errors,net_errors)
        fig,ax = plt.subplots(1)
        
        for key in error_keys:
            fig,ax = plt.subplots(1)
            ax.set_xlabel('log_10 N')
            if 'alpha' in key:
                ax.set_ylabel('log_{10} RMSE alpha')
                ax.set_title('log_{10} RMSE alpha' + ' Against log_{10} Number of Parameters (N)')
            else:
                ax.set_ylabel('log_{10} RMSE '+key[-1])
                ax.set_title('log_{10} RMSE '+ key[-1] + ' Against log_{10} Number of Parameters (N)')
                
            ax.scatter(np.log10(net_sizes),\
                       np.log10(error_dict[key][1]),label = 'Network',color = 'C0')
            ax.scatter(np.log10(spline_sizes),np.log10(error_dict[key][0]),label = 'Spline',color = 'C1')
        
            ax.grid(b = True)
            plt.savefig(savedir+'/M1_data_driven_errbysize_'+['alpha' if 'alpha' in key else key[-1]][0]+'.eps')
            plt.show()
            plt.clf()
        
#            plt.savefig(savedir)
        
#        net_errors = 
    
new_train_table = False
new_1dtest_table = False
new_2dtest_table = False

make_table1 = True
view_1dtest_table = False
view_2dtest_table = False

if new_train_table:
    #Method for networks is just empty string: ''
    
    domain = 'Train'
    method = 'Spline'
    
    if method == 'Spline':
        data_set_id = 'Spline'
    elif method == '':
        data_set_id = 'M1_set_A'

    name_train_table = domain+'_All_Results_'+method + data_set_id+'.pickle'
    
    columns = ['SaveId','Size','RMSE h','RMSE u','RMSE u0','RMSE u1','RMSE alpha']
    
    with open(name_train_table,'wb') as handle:
        
        pickle.dump([],handle)
        
        
if new_1dtest_table:
    #Method for networks is just empty string: ''
    method = 'Spline'
    data_set_id = 'M1_set_A'
    
    domain = '1dTest'
    
    name_1dtest_table = domain+'_All_Results_'+method + data_set_id+'.pickle'

    columns = ['SaveId','Size','RMSE h','RMSE u','RMSE u0','RMSE u1','RMSE alpha']
    
    with open(name_1dtest_table,'wb') as handle:
        
        pickle.dump([],handle)
        
if new_2dtest_table:
    method = 'Spline'
    data_set_id = 'M1_set_A'
    domain = '2dTest'
    name_2dtest_table = domain+'_All_Results_'+method+data_set_id +'.pickle'
    
    columns = ['SaveId','Size','RMSE h','RMSE u','RMSE u0','RMSE u1','RMSE alpha']

    with open(name_2dtest_table,'wb') as handle:
        
        pickle.dump([],handle)
        
if make_table1:
    
    data_set_id = 'M1_set_A'
    domain = 'Train'
    
    paper_directory = '/Users/Will/ml_closures/paper/figures'
    
    net_filename_train_data = domain + '_All_Results_' + data_set_id + '.pickle'
    net_filename_1dtest_data = '1dTest'+'_All_Results_' + data_set_id +'.pickle'
    net_filename_2dtest_data = '2dTest'+'_All_Results_'+data_set_id +'.pickle'
    
    spline_filename_train_data = domain+'_All_Results_'+'SplineSpline'+'.pickle'
    spline_filename_1dtest_data = '1dTest'+'_All_Results_' +'Spline' + data_set_id+'.pickle'
    spline_filename_2dtest_data = '2dTest'+'_All_Results_'+'Spline'+data_set_id+'.pickle'
    
    net_file_dict = {'Train':net_filename_train_data,'1dTest':net_filename_1dtest_data,\
                     '2dTest':net_filename_2dtest_data}
    
    spline_file_dict = {'Train':spline_filename_train_data,'1dTest':spline_filename_1dtest_data,\
                        '2dTest':spline_filename_2dtest_data}
    
    all_nets = ['0by15_new_1','0by30_new_1','0by45_new_2','1by15_new_2','1by30_new_2','1by45_new_1',\
                     '2by15_new_2','2by30_new_3','2by45_new_3','3by15_new_3','3by30_new_3','3by45_new_3']
    
    #selected_nets = ['0by15_new_1','2by45_new_3','3by45_new_3']
    selected_nets = all_nets
    selected_nets = ['M1_'+x+'m' for x in selected_nets]
    
    all_splines = ['Spline22','Spline42','Spline62','Spline82','Spline102','Spline122','Spline142','Spline162']
    
    #selected_splines = ['Spline22','Spline42','Spline122']
    selected_splines = all_splines
    
    selected_splines = [x+'m' for x in selected_splines]

    make_M1_summary_fromdata(net_file_dict,spline_file_dict,\
                         selected_nets[:],selected_splines[:],\
                         paper_directory,mode = 'plot')
    #Could remove m here from the names
    
    
    
    """
#   This section is for prototyping my code for make_table1_fromdata
    with open(table_filename,'rb') as handle:
        train_results_list = pickle.load(handle)
    
    
    columns = ['SaveId','Size','RMSE h','RMSE u','RMSE u0','RMSE u1','RMSE alpha']
    
    pd_results = pd.DataFrame(train_results_list,columns = [*columns[0:2],*['Train '+x for x in columns[2:]]])
    
    saveid_select = ['M1_0by15_new_3m','M1_0by15_new_2m'] 
    
    select_results = pd_results[pd_results['SaveId'].isin(saveid_select)]
    
    select_results = select_results.drop_duplicates()
    
    select_results['SaveId'] = select_results['SaveId'].astype(str)
    
    select_results = select_results.transpose()
    
    select_results.columns = select_results.iloc[0]
    select_results = select_results.drop(select_results.index[0])

    #select_results.options.display.float_format = '{:.2f}'.format
    
    print(select_results)
    """
    
    """
    results_data = results_numpy[:,1:]
    
    saveid_list = results_numpy[:,0].astype(str)
    
    column_labels= ['Size','RMSE h','RMSE u','RMSE u0','RMSE u1','RMSE alpha']
    
    df = pd.DataFrame(results_data,columns = column_labels,index = saveid_list)

    print(df[saveid_list[4]])
    """