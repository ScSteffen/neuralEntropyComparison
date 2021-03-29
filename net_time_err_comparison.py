# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 20:15:17 2021

@author: Will
"""
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np 
import pickle 
from tabulate import tabulate 
from runandsave import runandsave_call


def makeandsave_tasi(cpu,wall,**kwargs):
    plt.plot(cpu,color = 'orange',label = 'CPU')
    plt.plot(wall,color = 'blue',label = 'Wall')
    plt.title('Computation Time at Each Iteration ')
    plt.xlabel('Iteration Number')
    plt.ylabel('Time Elapsed (s)')
    plt.legend()
    plt.show()
    if 'savedir' in kwargs:
        plt.savefig(kwargs['savedir'])
        
def makeandsave_hist(data,name,**kwargs):
    
    data_arr = np.array(data)
    spread = np.std(data_arr)
    avg = np.mean(data_arr)
    #q25,q50,q75 = np.percentile(data,[0.25,0.50,0.75])
    q50 = np.median(data)
    
    q = 0.05
    low,med,high = np.quantile(data_arr,[q,0.5,(1-q)])
    refinement = (data[:] <= high) * (data[:] >= low)
    clean_dat = data_arr[refinement]
    clean_mean = np.mean(clean_dat)
    clean_stddev = np.std(clean_dat)
    """
    print('med,max,min',med,max(data),min(data))
    data_cols = ['Inner 90% Mean','Inner 90% Stddev','Min','Max','Median']
    data_info = [['{:.2e}'.format(clean_mean),'{:.2e}'.format(clean_stddev),'{:.2e}'.format(min(data)),\
                  '{:.2e}'.format(max(data)),'{:.2e}'.format(med)]]
    data_table = pd.DataFrame(data_info,columns = data_cols)
    """
    
    data_cols = ['Inner 90% Mean','Inner 90% StdDev','Min','Max','Median']
    data_info = [['{:.2e}'.format(clean_mean),'{:.2e}'.format(clean_stddev),'{:.2e}'.format(min(data)),\
                  '{:.2e}'.format(max(data)),'{:.2e}'.format(q50)]]
    data_table = pd.DataFrame(data_info,columns = data_cols)
    print('\n\n',name+'Time Summary:','\n\n',data_table.to_latex())
    #mpltable(error_table,width = 0.12,savedir = savefolder + 'firsterrtable'+saveid+'.pdf')
    print('\n\n',name + 'Times Summary: \n\n',\
          tabulate(data_table,headers = 'keys',tablefmt = 'psql'))
    
    
    nbins = 10
    print('Number of bins:',nbins)
    counts,bins = np.histogram(data,bins = nbins)
    
    """
    q25, q75 = np.percentile(data,[.25,.75])
    bin_width = 2*(q75 - q25)*len(data)**(-1/3)
    nbins = int((max(data) - min(data))/bin_width)
    print("Freedman–Diaconis number of bins:", nbins)
    """
    
    clean_bin_edges = [round(x,1) for x in bins][0:-1]
    fig,ax = plt.subplots()
    bin_width = (bins[1]-bins[0])
    ax.bar(clean_bin_edges,counts,width = bin_width, align ='edge')
    ax.set(xticks = clean_bin_edges,
       xlim = [round(bins[0],1),round(bins[-1],1)])
    
    print('Counts and Bins for' +name,counts,bins)
    ax.set_xlabel(name + ' Times')
    ax.set_ylabel('Counts')
    ax.set_title('Histogram of '+name+' Times')
    if 'savedir' in kwargs:
        plt.savefig(kwargs['savedir'])
        
    fig.show()
    
    """
    #q25, q75 = np.percentile(data,[.25,.75])
    #bin_width = 2*(q75 - q25)*len(data)**(-1/3)
    #nbins = int((max(data) - min(data))/bin_width)
    #print("Freedman–Diaconis number of bins:", nbins)
    
    
    plt.hist(data, bins = nbins)
    plt.title('Histogram of '+name+' Times')
    plt.xlabel(name+' Time')
    plt.ylabel('Counts')
    plt.legend()
    if 'savedir' in kwargs:
        plt.savefig(kwargs['savedir'])
    plt.show()
    """

def plot_times(saveid,savefolder):
    """
    Saveid is of same form as usual
    """
    saveplot_times= savefolder + saveid_analyze + 'timesasiter.pdf'
    saveplot_cpu_hist = savefolder + saveid_analyze + 'cpuhist.pdf'
    saveplot_wall_hist = savefolder + saveid_analyze + 'wallhist.pdf'
    
    file_analyze = saveid_analyze + 'timings.pickle'
    with open(file_analyze,'rb') as anlyzfile:
        cpu_times_analyze,wall_times_analyze = pickle.load(anlyzfile)
        
    fixed_cpu = [round(x,3) for x in cpu_times_analyze]
    
    fixed_wall = [round(x,3) for x in wall_times_analyze]
    
    #Plot both timings at once
    makeandsave_tasi(fixed_cpu,fixed_wall,savedir = saveplot_times)
    
    #Histogram each timings separately
    makeandsave_hist(fixed_cpu,name = 'CPU',savedir = saveplot_cpu_hist)
    
    makeandsave_hist(fixed_wall,name = 'Wall',savedir =  saveplot_wall_hist)
        

def pickle_times(saveid):
    """
    Saveid are of the form:
        M2_2by30_2symm 
        or 
        M2opt 
    """
    filename = saveid + '_current_timings.pickle'
    #Create the empty pickle file with the saveid 
    
    type1_times = [] 
    type2_times = [] 
    
    with open(filename,'wb') as newfile:
        pickle.dump([type1_times,type2_times],newfile)

def get_total_time(saveid,closure,method,symm,N,**kwargs):
    filename = saveid + '_current_timings.pickle'
    #Create the empty pickle file with the saveid 
    type1_times = [] 
    type2_times = [] 
    
    with open(filename,'wb') as newfile:
        pickle.dump([type1_times,type2_times],newfile)
        
    runandsave_call(doreturn = False,Closure = closure,N = N,test = 'plane',\
                    nx = 100,display = False,saveid= saveid,method = method)


def get_final_state(saveid,closure,method,symm = False,N,**kwargs):
    """
    Savedir should be of form
    """
    filename = saveid + '_final_state.pickle'

    #newconvtest = True just causes runandsave_call to 
    #return the solution at the final time as well as other info
    #This is not the same as running convergence test. 
    """
    if 'method' in kwargs:
        method = kwargs['method']
    else:
        method = 'net'
    """
    
    #For symmetric networks, we need to go and manually input saveid, and change to symm, inside runandsave_call
    uFinal,x,dx,dt,tf =  runandsave_call(doreturn = True,Closure = closure,N = N, test = 'plane',\
                                                  nx = 100,display = True,saveid = saveid,method = method)
    with open(filename,'wb') as handle:
        pickle.dump(uFinal,handle)
        
def compare_error_with_time(savedir,N,**kwargs):
    
    if kwargs['mode'] == 'get':
        saveid_list = kwargs['saveid_list'] 
        #Create an empty timing sample file
        for saveid in saveid_list:
            filename = saveid + '_current_timings.pickle'
            #Create the empty pickle file with the saveid 
            type1_times = [] 
            type2_times = [] 
            with open(filename,'wb') as newfile:
                pickle.dump([type1_times,type2_times],newfile)
        #List-comprehension determines the closure (can change this to a list comp or regex)
        closures_dict = {}
        methods_dict = {}
        symm_bools_dict = {}
        for saveid in saveid_list:
            if 'by' in saveid or 'spline' in saveid:
                closure = 'M_N approx'
                if 'by' in saveid:
                    method = 'net'
                if 'spline' in saveid:
                    method = 'spline'
            elif 'P1' in saveid:
                closure = 'P_N'
                method = None
            elif 'opt' in saveid:
                closure = 'M_N'
                method = None
            if 'symm' in saveid:
                symm = True
            else:
                symm = False
                
            closures_dict[saveid] = closure
            methods_dict[saveid] = method
            symm_bools_dict[saveid] = symm
            
        #Save final state of solver:
        for saveid in saveid_list:
            get_final_state(saveid,closures_dict[saveid],methods_dict[saveid],\
                       symm_bools_dict[saveid],N)
            
        
        
    
    if kwargs['mode'] == 'analyze':
        saveid_list = kwargs['saveid_list']
        
        if N == 2:
            saveid_list = ['M2','P2','M2_0by15_3','M2_0by15_3symm','M2_2by30_2','M2_2by30_2symm',\
                           'M2_4by50_2','M2_4by50_2symm','M2_4by50_4','M2_4by50_4symm'] 
            
            with open('M2opt_final_state.pickle','rb') as statehandle:
                uFinal_ref = pickle.load(statehandle)
            with open('MNopt_current_timings.pickle','rb') as timehandle:
                toss_times,wall_times = pickle.load(timehandle)
                
        elif N ==1:
            
            saveid_list = ['M1opt','P1','M1_1by30_match_1','M1_2by45_match_1','M1_5by45_match_3','spline960','spline4140','spline10215']
        
            with open('M1opt_final_state.pickle','rb') as statehandle:
                uFinal_ref = pickle.load(statehandle)
            with open('M1opt_current_timings.pickle','rb') as timehandle:
                toss_times,wall_times = pickle.load(timehandle)
            
        time_err_data = np.zeros((2,len(saveid_list)),dtype = float)
        
        with open('dx100.pickle','rb') as filehandle:
            dx = pickle.load(filehandle) 
            print('\n\n here is dx',dx,'\n\n')
            
            
        q1,q2,q3 = np.quantile(wall_times,[0.25,0.5,0.75])
        """
        time_err_data[0,0] = min(wall_times)
        time_err_data[1,0] = q1
        """
        time_err_data[0,0] = q2
        """
        time_err_data[3,0] = q3
        time_err_data[4,0] = max(wall_times)
        #Slice 0th index for u0 error.
        """
        time_err_data[1,0] = 0
        
        i = int(1)
        for saveid in saveid_list[1:]:
            
            with open(saveid + '_current_timings.pickle','rb') as timehandle:
    
                toss_times,wall_times = pickle.load(timehandle) 
        
            q1,q2,q3 = np.quantile(wall_times,[0.25,0.5,0.75])
            
            with open(saveid + '_final_state.pickle','rb') as statehandle:
                
                uFinal = pickle.load(statehandle) 
                
            """
            time_err_data[0,i] = min(wall_times)
            time_err_data[1,i] = q1
            """
            time_err_data[0,i] = q2
            """
            time_err_data[3,i] = q3
            time_err_data[4,i] = max(wall_times)
            #Slice 0th index for u0 error.
            """
            
            time_err_data[1,i] = np.sqrt( dx*np.sum(np.square(uFinal-uFinal_ref),axis = 0) [0] / dx*(np.sum(np.square(uFinal_ref),axis = 0)[0]) )
            
            i += int(1)
            
        pd.set_option('display.float_format','{:.2e}'.format)
        
        df = pd.DataFrame(time_err_data,columns = saveid_list, index = ['Median Time (s)','L2 u0 Error'])
        
        df.to_latex()
        
        print(tabulate(df))
        
        print(df.to_latex())
    
    
if __name__ == "__main__":
    
    #'M2opt' is optimization saveid for thhe create_state function; must retrievew for timings
    
    #saveid = 'M2_0by15_3symm'
    #M1_5by45_match_3
    
    #saveid = 'P2'
    
    N = 1
    
    no = 'spline10215'
    
    Closure = 'M_N approx'
    
    create_times = False
    
    analyze_times = False
    
    analyze_errors = False
    
    analyze_both = False
    
    create_state = False
    
    saveid_list = ['M2opt','M1_4by10','P1','abc','spline22']

    
    if create_times == True:
        
        pickle_times(saveid)
        
    elif create_state == True:
        
        save_state(saveid,Closure,symm = False)
        
    elif analyze_times == True:
        
        plot_times(saveid) 
        
    elif analyze_both == True:
        
        compare_error_times(None,N)
    

        
        