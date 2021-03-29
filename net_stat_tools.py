# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 20:55:29 2020

@author: Will
"""
import numpy as np 
import matplotlib.pyplot as plt 
from vizualization_tools import mesh_hist,superplot,mpltable,plot_mm,barhist
from tabulate import tabulate
import math 
import pandas as pd 
import warnings 
from MN_Duality_Tools import moment_vector
import pickle
    
warnings.simplefilter('error',RuntimeWarning)

def truncate(number, digits):
    stepper = 10.0 ** digits
    newval = math.trunc(stepper * number) / stepper
    if newval > 100:
        newval = int(newval)
        return newval
    else:
        return newval
    
def plot_heatmap(points,nbins,**opts):
    x_vals = points[:,0]
    y_vals = points[:,1] 
    bincounts,xbins,ybins = np.histogram2d(x_vals,y_vals,bins = nbins)
    xinds,yinds = np.digitize(x_vals,xbins) - 1,np.digitize(y_vals,ybins) - 1
    bounds = [xbins[0]-1,xbins[-1]+1,ybins[0]-1,ybins[-1]+1]
    heat = np.zeros((nbins+1,nbins+1),dtype = float)
    newbincount = np.zeros((nbins+1,nbins+1),dtype = float)
    for i in range(len(xinds)):
        newbincount[xinds[i],yinds[i]] += 1
        heat[xinds[i],yinds[i]] += points[i,2]
        
    hot = newbincount > 0
    cold = 1-hot  
    temp_weights = newbincount + 1*cold 
    logavg_heat = np.log10(np.divide(heat,temp_weights) + \
                           np.min(np.divide(heat,temp_weights)[heat > 0])*cold)
    
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(xbins,ybins,logavg_heat.T,cmap = 'jet')
    
    cbar = fig.colorbar(heatmap,ax = ax)
    #cbar.set_label('Log10 Avg Error', rotation = 270)
    
    #plt.imshow(heat.T,extent = bounds,cmap = 'jet',origin = 'lower')
    #plt.colorbar()
    if 'xlabel' in opts:
        ax.set_xlabel(opts['xlabel'])
    if 'ylabel' in opts:
        ax.set_ylabel(opts['ylabel'])
    if 'title' in opts:
        ax.set_title(opts['title'])
    if 'savedir' in opts:
        plt.savefig(opts['savedir'])
        
    plt.show()
    
def plotlosscurve(history,bools,regularization,savefolder,saveid):
    #Get the saveid and savefolder
    
    
    enforce_func,enforce_grad,enforce_moment,enforce_conv = bools
    
    if type(enforce_grad) == float:
        if enforce_grad > 0:
            enforce_grad = True
    if type(enforce_moment) == float:
        if enforce_moment > 0:
            enforce_moment = True
    if type(enforce_func) == float:
        if enforce_func > 0:
            enforce_moment = True      
    if type(enforce_grad) == float:
        if enforce_conv > 0:
            enforce_conv = True
      
    
    plt.plot(np.log10(history.history['loss']))
    plt.plot(np.log10(history.history['val_loss']))
    plt.legend(['training','validation'],loc = 'upper right')
    plt.title('Log10 of Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Log10 Loss Value')
    plt.savefig(savefolder+'losscurve'+saveid+'.eps')
    plt.show()
    
    #plt.plot(np.log10(history.history['lr']))
    #plt.xlabel('Epochs')
    #plt.ylabel('Log10 of Learning Rate')
    #plt.title('Log10 of Base Learning Rate Value')
    #plt.savefig(savefolder+'learningrate'+saveid+'.eps')
    #plt.show()
    
    #Make and modify loss keys according to weights
    legend_train = []
    legend_validate = []
    components_train = []
    components_validate = []
    if enforce_func == True:
        legend_train.append('function')
        legend_validate.append('function')
        components_train.append('output_1_loss')
        components_validate.append('val_output_1_loss')
    if enforce_grad == True:
        legend_train.append('gradient')
        legend_validate.append('gradient')
        components_train.append('output_2_loss')
        components_validate.append('val_output_2_loss')
    if enforce_moment == True:
        legend_train.append('moment')
        legend_validate.append('moment')
        components_train.append('output_3_loss')
        components_validate.append('val_output_3_loss')
    if enforce_conv == True:
        legend_train.append('conv')
        legend_validate.append('conv')
        components_train.append('output_4_loss')
        components_validate.append('val_output_4_loss')
    for key in components_train:
        if key == 'output_4_loss':
            try: 
                plt.plot(np.log10(history.history[key]),label = key)
            except RuntimeWarning:  
                legend_train.remove('conv')
        else:
            plt.plot(np.log10(history.history[key]),label = key)
    if regularization == True:
        reg_vals = np.array(history.history['loss'])-\
        (float(bools[0])*np.array(history.history['output_1_loss']) + float(bools[1])*np.array(history.history['output_2_loss']) + 
         float(bools[2])*np.array(history.history['output_3_loss'])+float(bools[3])*np.array(history.history['output_4_loss']) )
        try:
            plt.plot(np.log10(reg_vals),label = 'regularization')
        except RuntimeWarning:
            print(reg_vals)
        legend_train.append('regularization')
    plt.legend(legend_train,loc = 'upper right')
    plt.title('Log10 of Training Loss Components')
    plt.xlabel('Epochs')
    plt.ylabel('Log10 Loss Value')
    plt.savefig(savefolder+'complosscurvetrain'+saveid+'.eps')
    plt.show()
    
    for key in components_validate:
        if key == 'val_output_4_loss':
            try: 
                plt.plot(np.log10(history.history[key]),label = key)
            except RuntimeWarning:  
                legend_validate.remove('conv')
        else:
            plt.plot(np.log10(history.history[key]),label = key)
    plt.legend(legend_validate,loc = 'upper right')
    plt.title('Log10 of Validation-Loss Components')
    plt.xlabel('Epochs')
    plt.ylabel('Log10 Loss Value')
    plt.savefig(savefolder+'complosscurveval'+saveid+'.eps')
    plt.show()  

    if enforce_conv == True and ('conv' not in legend_train):
        plt.plot(history.history['output_4_loss'])
        plt.title('Training Conv-Loss Value (Achieves Zero)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.savefig(savefolder+'convlosstrain'+saveid+'.eps')
    if enforce_conv == True and ('conv' not in legend_validate):
        plt.plot(history.history['val_output_4_loss'])
        plt.title('Validation Conv-Loss Value (Achieves Zero)')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.savefig(savefolder+'convlossval'+saveid+'.eps')
    
def quickstats(f,alpha,u,conv,savefolder,saveid):
    
    h_pred,h_train = f
    alpha_pred,alpha_train = alpha
    u_pred,u_train = u
    conv_pred = conv
    
    h_pred_graph = np.hstack((u_train,h_pred))
        
    h_true = h_train.reshape((h_train.shape[0],1))
    h_true_graph = np.hstack((u_train,h_true))
    
    MSE = np.mean(np.square(h_pred-h_true),axis = 0)[0]
    MSE_rel = np.sum(np.square(h_pred-h_true),axis = 0)[0]/np.sum(np.square(h_true),axis = 0)[0]
    
    u_MSE = np.mean(np.sum(np.square(u_pred-u_train),axis = 1))
    u_norm = np.mean(np.sum(np.square(u_train),axis =1))
    u_MSE_rel = u_MSE/u_norm
    
    alpha_train_norm = np.mean(np.sum(np.square(alpha_train),axis = 1))
    final_MSE_alpha = np.mean(np.sum(np.square(alpha_pred-alpha_train),axis =1))
    final_MSE_alpha_rel = final_MSE_alpha / alpha_train_norm
    
    det_pred_train = conv_pred[:,0]
    a_pred_train = conv_pred[:,1]
    
    #moment_loss_curve = moment_history.history['output_3_loss']
    
    num_nonconv_train = np.sum((det_pred_train < 0) + (a_pred_train < 0))
    
    error_cols = ['MSE-f','MSEr-f','MSEr-alpha','MSE-u','MSEr-u','# NegDef']
    error_info = [['{:.2e}'.format(MSE),'{:.2e}'.format(MSE_rel),'{:.2e}'.format(final_MSE_alpha_rel),\
                   '{:.2e}'.format(u_MSE),'{:.2e}'.format(u_MSE_rel),num_nonconv_train]]
    
    
    error_table = pd.DataFrame(error_info,columns = error_cols)
    mpltable(error_table,width = 0.13,savedir = savefolder + 'firsterrtable'+saveid+'.pdf')
    print('\n\n','Network Final Error on Training Data (exclude validation): \n\n',\
          tabulate(error_table,headers = 'keys',tablefmt = 'psql'))
    
    #Plot network function 
    superplot(approx_vals = h_pred_graph,targetvals = h_true_graph,\
              view = [10,20],title = 'Network and Target at Training Pts',\
              savedir = savefolder + 'viewnet'+saveid+'.pdf')
    
    
def runerranalysis(testdom_object,compute_conv,f,alpha,u,conv,savefolder,saveid):
    testinfo = testdom_object
    h_pred_test = f
    grad_pred_test = alpha
    u_predict = u
    conv_test_pred  = conv
            
    det_pred = conv_test_pred[:,0]
    a_pred = conv_test_pred[:,1]
    
    
    moment_pred = np.zeros((testinfo.moment_domain.shape[0],2))
    h_true_test = testinfo.graph[:,2]
    for i in range(testinfo.moment_domain.shape[0]):
        moment_pred[i] = moment_vector(grad_pred_test[i])
    h_true_test = h_true_test.reshape((h_true_test.shape[0],1))
    
    if np.allclose(moment_pred,0) == True:
        raise ValueError('moment prediction is still zero within tolerance of np.allclose')
        
    ### Function Error ###
    
        #Compute function error over the test domain:
    
    funcerr_vals = np.square(h_pred_test - h_true_test)
    MSE_test = np.mean(np.square(h_true_test-h_pred_test))
    l2_test = np.mean(np.square(h_pred_test))
    MSErel_test = MSE_test / l2_test
    
        #Plot function error over the test domain 
    
    funcerr_graph = np.hstack((testinfo.moment_domain,funcerr_vals))
    #superplot(errorpoints = funcerr_graph,targetvalsEarly  = testinfo.points,title = 'Function Error and Hull',\
              #savedir = savefolder+'funcerr'+saveid+'.pdf')
    plot_heatmap(funcerr_graph,50,title = 'Squared Function Error: Log10 Average',\
                 xlabel = 'u0',ylabel = 'u1',savedir = savefolder+'funcheat'+saveid+'.eps')

    ### Moment Error ###
        #Compute moment error over finer domain

    momenterr_vals = np.sum(np.square(testinfo.moment_domain - moment_pred),axis = 1)
    MSE_moment = np.sum(np.mean((np.square(testinfo.moment_domain - moment_pred)),axis = 0))
    l2_moment = np.mean(np.sum(np.square(testinfo.moment_domain),axis =1))
    MSErel_moment = MSE_moment / l2_moment
    
        #Plot the moment error over the test domain 
    
    momenterr_plot = momenterr_vals.reshape((momenterr_vals.shape[0],1))
    momenterr_graph = np.hstack((testinfo.moment_domain,momenterr_plot))
    
    #superplot(errorpoints = momenterr_graph,targetvals = testinfo.points,title = 'Moment Error and Hull',\
              #savedir = savefolder+'momenterr'+saveid+'.pdf')
    plot_heatmap(momenterr_graph,50,title = 'Squared Moment Error: Log10 Average',\
                 xlabel = 'u0',ylabel = 'u1',savedir = savefolder+'momentheat'+saveid+'.eps')
    
    ### Gradient (Alpha) Error:
        #Compute alpha errorr on the domain 

    alphaerr_vals = np.sum(np.square(grad_pred_test - testinfo.multiplier_domain),axis = 1)
    MSE_alpha= np.mean(alphaerr_vals)
    l2_alpha = np.mean(np.sum(np.square(testinfo.multiplier_domain),axis = 1))
    MSErel_alpha = MSE_alpha / l2_alpha 
    
        #Plot the alpha errors over the test domain 
    
    alphaerr_plot = alphaerr_vals.reshape(alphaerr_vals.shape[0],1)
    alphaerr_graph = np.hstack((testinfo.moment_domain,alphaerr_plot))
    #superplot(errorpoints = alphaerr_graph,targetvals = testinfo.points,title = 'Alpha Error and Hull',\
              #savedir = savefolder +'alphaerr'+saveid+'.pdf')
    plot_heatmap(alphaerr_graph,50,title  = 'Squared Alpha Error: Log10 Average',\
                 xlabel = 'u0',ylabel = 'u1',savedir = savefolder+'alphaheat'+saveid+'.eps')
    
    ### Determinant Values: 
    if compute_conv == True:
        detneg = det_pred < 0
        aneg = a_pred < 0
        num_negdef_points = np.sum(aneg + detneg)
        print('Number of negative definite points','\n', num_negdef_points,'\n\n')
            #Pretty-print the error values
    else:
        num_negdef_points = np.nan

    test_cols = ['h','rel-h','u','rel-u','rel-alpha','#NegDef']
    test_errors = [[truncate(MSE_test,4),truncate(MSErel_test,4),truncate(MSE_moment,4),truncate(MSErel_moment,4),\
                    truncate(MSErel_alpha,4),num_negdef_points]]
    test_table = pd.DataFrame(test_errors,columns = test_cols)
    print('\n\n','Network Error Metrics Over Test Domain:','\n',tabulate(test_table,headers = 'keys',tablefmt = 'psql'))
    mpltable(test_table,width = 0.12,savedir = savefolder+'metricstable'+saveid+'.pdf')
    
def plot_M1_1d(truevals,predvals,**kwargs):
    
    h_true,u_true = truevals
    h_pred,u_pred = predvals
    
    u0_errors = np.log10(  np.divide(np.abs(u_true[:,0] - u_pred[:,0]),np.abs(u_true[:,0]))+1e-10 )
    u1_errors = np.log10(  np.divide(np.abs(u_true[:,1] - u_pred[:,1]), np.abs(u_true[:,1]))+1e-10 )
    f_errors =  np.log10(  np.divide(np.abs(h_true-h_pred),np.abs(h_true)   ) + 1e-10            )
    
    fig, ax = plt.subplots(1)
    """
    if 'xlabel' in kwargs:
        ax.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    """
        
    ax.scatter(u_true[:,1],u0_errors,color = 'blue')
    ax.set_xlabel('u_1')
    ax.set_ylabel('log_10 u_0 Error') 
    ax.set_title('log_10 of Relative u_0 Error')
    plt.show()
    if 'savedir' in kwargs:
        plt.savefig(kwargs['savedir']+'1dtest_u0')
    plt.clf()
    
    fig,ax = plt.subplots(1)
    ax.scatter(u_true[:,1],u1_errors,color = 'orange')
    ax.set_xlabel('u_1')
    ax.set_ylabel('log_10 u_1 Error') 
    ax.set_title('log_10 of Relative u_1 Error')
    plt.show()
    if 'savedir' in kwargs:
        plt.savefig(kwargs['savedir']+'1dtest_u1')
    plt.clf()
    
    fig,ax = plt.subplots(1)
    ax.scatter(u_true[:,1],f_errors,color = 'green')
    ax.set_xlabel('u_1')
    ax.set_ylabel('log_10 h Error') 
    ax.set_title('log_10 of Relative h Error')
    plt.show()
    if 'savedir' in kwargs:
        plt.savefig(kwargs['savedir']+'1dtest_h')
    plt.clf()

#    ax.legend()
  
    
def quickstats_scaled(f,alpha,u,conv,savefolder,saveid,data_id,size,method,domain,N = 1,append_results= False,**kwargs):

    h_pred,h_train = f
    alpha_pred,alpha_train = alpha
    u_pred,u_train = u
    
    if method == 'net':
        
        conv_pred = conv
    
    h_pred = h_pred.reshape((h_pred.shape[0],1))
    h_true = h_train.reshape((h_train.shape[0],1))
    
    L2_h = np.mean(np.square(h_pred-h_true),axis = 0)[0]
    L2_hrel = np.sum(np.square(h_pred-h_true),axis = 0)[0] / np.sum(np.square(h_true),axis = 0)[0]
    
    L2_u = np.mean(np.sum(np.square(u_pred-u_train),axis = 1))
    L2norm_u = np.mean(np.sum(np.square(u_train),axis =1))
    L2_urel = L2_u / L2norm_u
    
    L2_u0 = np.mean(np.square(u_pred[:,0]-u_train[:,0]))
    u0_norm = np.mean(np.square(u_train[:,0]))
    L2_u0rel = L2_u0 / u0_norm
    
    L2norm_alpha = np.mean(np.sum(np.square(alpha_train),axis = 1))
    L2_alpha = np.mean(np.sum(np.square(alpha_pred-alpha_train),axis =1))
    L2_alpharel = L2_alpha / L2norm_alpha
    
    L2_u0_spec = np.mean(np.square(u_train[:,0] - u_pred[:,0])) / np.mean(np.square(u_train[:,0])) 
    L2_u1_spec = np.mean(np.square(u_train[:,1] - u_pred[:,1])) / np.mean(np.square(u_train[:,1]))
    if N == 2:   
        L2_u2_spec = np.mean(np.square(u_train[:,2] - u_pred[:,2])) / np.mean(np.square(u_train[:,2]))
    
    if method == 'net':
        
        if N == 1:
            
            num_nonconv_train = np.sum((conv_pred < 0))
            
        if N == 2:
            
            num_nonconv_train = np.sum((conv_pred[:,0] < 0) + (conv_pred[:,1] < 0))
            
            
        error_cols = ['MSE f','MSE r-f','MSE alpha','MSE r-alpha','MSE u','MSE r-u','MSE r-u0','# NegDef']
        error_info = [['{:.2e}'.format(L2_h),'{:.2e}'.format(L2_hrel),'{:.2e}'.format(L2_alpha),'{:.2e}'.format(L2_alpharel),\
                       '{:.2e}'.format(L2_u),'{:.2e}'.format(L2_urel),'{:.2e}'.format(L2_u0rel),num_nonconv_train]]
        error_table = pd.DataFrame(error_info,columns = error_cols)
        print('\n ------------------------------------------------------------------- \n ')
        print(domain+' Data Error Summary (excluding validation) in LaTeX','\n\n',error_table.to_latex())
        
        #mpltable(error_table,width = 0.12,savedir = savefolder + 'firsterrtable'+saveid+'.pdf')
        print('\n\n',domain + ' Data Error Summary (exclude validation): \n\n',\
              tabulate(error_table,headers = 'keys',tablefmt = 'psql'))
        
    elif method == 'spline':
        
        error_cols = ['MSE f','MSE r-f','MSEalpha','MSEr-alpha','MSE u','MSE r-u','MSE r-u0']
        error_info = [['{:.2e}'.format(L2_h),'{:.2e}'.format(L2_hrel),'{:.2e}'.format(L2_alpha),'{:.2e}'.format(L2_alpharel),\
                       '{:.2e}'.format(L2_u),'{:.2e}'.format(L2_urel),'{:.2e}'.format(L2_u0rel)]]
        error_table = pd.DataFrame(error_info,columns = error_cols)
        print('\n -------------------------------------------------------------------- \n')
        print(domain+' Domain Error Summary (excluding validation) in Latex','\n\n',error_table.to_latex())
        #mpltable(error_table,width = 0.12,savedir = savefolder + method + 'firsterrtable' + saveid + '.pdf')
        print('\n\n',method+' '+domain+' Domain Error Summary (excluding validation): \n\n',\
              tabulate(error_table,headers = 'keys',tablefmt = 'psql'))
    
    
    if N ==2:

        Select_MSE_Cols = ['MSE r-f','MSE r-u','MSE r-u0','MSE r-u1','MSE r-u2', 'MSE r-alpha']
            
        Select_RMSE_Cols = ['RMSE r-f','RMSE r-u','RMSE r-u0','RMSE r-u1','RMSE r-u2', 'RMSE r-alpha']
        
        RMSE_vals = [[np.sqrt(L2_hrel),np.sqrt(L2_urel),np.sqrt(L2_u0_spec),\
                      np.sqrt(L2_u1_spec),np.sqrt(L2_u2_spec),np.sqrt(L2_alpharel)]]
        
        MSE_vals = [[L2_hrel,L2_urel,L2_u0_spec,L2_u1_spec,L2_u2_spec,L2_alpharel]]
        
        Format_RMSE_vals = [['{:.2e}'.format(x) for x in RMSE_vals[0]]]
        
        Format_MSE_vals = [['{:.2e}'.format(x) for x in MSE_vals[0]]]

        Format_MSE_Table = pd.DataFrame(Format_MSE_vals,columns = Select_MSE_Cols)
        
        Format_RMSE_Table = pd.DataFrame(Format_RMSE_vals,columns = Select_RMSE_Cols)
        
        print('\n\n Select ' + domain +' RMSE Table in Latex:\n\n ',Format_RMSE_Table.to_latex(),'\n\n')
        
    if N == 1:
        
        Select_MSE_Cols = ['MSE r-f','MSE r-u','MSE r-u0','MSE r-u1', 'MSE r-alpha']
            
        Select_RMSE_Cols = ['RMSE r-f','RMSE r-u','RMSE r-u0','RMSE r-u1', 'RMSE r-alpha']
        
        RMSE_vals = [[np.sqrt(L2_hrel),np.sqrt(L2_urel),np.sqrt(L2_u0_spec),\
                      np.sqrt(L2_u1_spec),np.sqrt(L2_alpharel)]]
        
        MSE_vals = [[L2_hrel,L2_urel,L2_u0_spec,L2_u1_spec,L2_alpharel]]
    
        Format_RMSE_vals = [['{:.2e}'.format(x) for x in RMSE_vals[0]]]
        
        Format_MSE_vals = [['{:.2e}'.format(x) for x in MSE_vals[0]]]
        
        Format_RMSE_Table = pd.DataFrame(Format_RMSE_vals,columns = Select_RMSE_Cols)
        
        Format_MSE_Table = pd.DataFrame(Format_MSE_vals,columns = Select_MSE_Cols)
        
        if append_results:
            
            results_to_append = [saveid,size,np.sqrt(L2_hrel),np.sqrt(L2_urel),np.sqrt(L2_u0_spec),\
                                 np.sqrt(L2_u1_spec),np.sqrt(L2_alpharel)]
            
            if (method == 'spline') and (domain == 'Train'):
                
                with open(domain+'_All_Results_'+'Spline'+'Spline'+'.pickle','rb') as handle:
                    
                    result_list = pickle.load(handle)
                    
                result_list.append(results_to_append)
                
                with open(domain+'_All_Results_'+'Spline'+'Spline'+'.pickle','wb') as newhandle:
                    
                    pickle.dump(result_list,newhandle)
                    
            elif (method == 'spline') and (domain == '1dTest'):
                
                with open(domain+'_All_Results_'+'Spline'+data_id+'.pickle','rb') as handle:
                    
                    result_list = pickle.load(handle)
                    
                result_list.append(results_to_append)
                
                with open(domain+'_All_Results_'+'Spline'+data_id+'.pickle','wb') as newhandle:
                    
                    pickle.dump(result_list,newhandle)
                    
            else:
                    
                with open(domain+'_All_Results_'+data_id+'.pickle','rb') as handle:
                    
                    result_list = pickle.load(handle)
                    
                result_list.append(results_to_append)
            
                with open(domain+'_All_Results_'+data_id+'.pickle','wb') as newhandle:
                    
                    pickle.dump(result_list,newhandle)
                
        print('\n\n Select ' + domain + 'RMSE Table in Latex:\n\n ',Format_RMSE_Table.to_latex(),'\n\n')
        
    #Plot network function 
    if N ==2:
        
        if 'plot' in kwargs:
            if kwargs['plot'] == True:
                h_true_graph = np.hstack((u_train[:,1:],h_true))
                h_pred_graph = np.hstack((u_train[:,1:],h_pred))
                superplot(approx_vals = h_pred_graph,targetvals = h_true_graph,\
                          view = [150,20], title = 'Network and Target', \
                          savedir = savefolder + method + 'viewnet' + saveid +'.pdf')
            else:
                pass
    if N == 1:
        
        if 'plot1d' in kwargs:
            if kwargs['plot1d'] == True:
                plot_M1_1d([h_train,u_train],[h_pred,u_pred],savedir = savefolder + saveid)
            else:
                pass
            
        if 'plot3d' in kwargs:
            if kwargs['plot3d'] == True:
                h_true_graph = np.hstack((u_train[:,:],h_true))
                h_pred_graph = np.hstack((u_train[:,:],h_pred))
                superplot(approx_vals = h_pred_graph,targetvals = h_true_graph,\
                          view = [150,20],title = 'Network and Target at Training Pts',\
                          savedir = savefolder + method + 'viewnet'+saveid+'.pdf')
            else:
                pass
    
def runerranalysis_scaled(truevals,compute_conv,predvals,savefolder,saveid,data_id,size,method,domain,L1 = False,append_results = False,N = 1):
    
    print('\n\n','MSE := (1/N)*\sum_{i} (x_i^{pred}-x_i^{true})^{2} \n MSRE := (1/N)* \sum_{i} [(x_i^{pred} - x_i^{true}) / (x_^{i} true)]^{2}')
    
    #Import true values 
    if N == 1:
        """
        testinfo = truevals
        h_true = testinfo.graph[:,2]
        u_true = testinfo.moment_domain 
        u_true2d = testinfo.moment_domain
        grad_true = testinfo.multiplier_domain
        """
        h_true,grad_true,u_true = truevals
        u_true2d = u_true[:,:]
        
    if N ==2:
        
        h_true,grad_true,u_true = truevals
        u_true2d = u_true[:,1:]
        
    #Get predicted values on the test domain
    h_pred,grad_pred,u_pred,conv_pred = predvals

    #conv_true = 0
    
    h_pred = h_pred.reshape((h_pred.shape[0],1))
    h_true = h_true.reshape((h_true.shape[0],1))
        
    ### Calculate function errors
    L2_func_title,L1_func_title = 'Squared ','Norm '
    L2_func_vals,L1_func_vals = np.square(h_pred-h_true),np.sqrt(np.square(h_pred-h_true))
    L2_func,L1_func = np.mean(L2_func_vals),np.mean(L1_func_vals)
    L2_funcrel,L1_funcrel = L2_func / np.mean(np.square(h_true)), L1_func / np.mean(np.abs(h_true),axis = 0)

    if L1 == True:
        if N == 1:
            L1_funcerr_graph = np.hstack((u_true2d,L1_func_vals))
            plot_heatmap(L1_funcerr_graph,50,title = L1_func_title + 'Function Error: Log10 Average',\
                         xlabel = 'u0',ylabel = 'u1',savedir = savefolder+method+'funcheat'+saveid+'.eps')
    elif L1 == False:
        if N == 1:
            """
            L2_funcerr_graph = np.hstack((u_true2d,L2_func_vals))
            plot_heatmap(L2_funcerr_graph,50,title = L2_func_title + 'Function Error: Log10 Average',\
                         xlabel = 'u0',ylabel = 'u1',savedir = savefolder+method+'funcheat'+saveid+'.eps')
            """
            pass
    ### Calculate moment errors
    
    L2_u0_spec = np.mean(np.square(u_true[:,0] - u_pred[:,0])) / np.mean(np.square(u_true[:,0])) 
    L2_u1_spec = np.mean(np.square(u_true[:,1] - u_pred[:,1])) / np.mean(np.square(u_true[:,1]))
    if N == 2:   
        L2_u2_spec = np.mean(np.square(u_true[:,2] - u_pred[:,2])) / np.mean(np.square(u_true[:,2]))
    
    L2_u_vals,L1_u_vals = np.sum(np.square(u_true - u_pred),axis = 1),np.sqrt(np.sum(np.square(u_true - u_pred),axis = 1))
    L2_u,L1_u = np.mean(L2_u_vals,axis = 0),np.mean(L1_u_vals,axis = 0)
    L2_u_rel, L1_u_rel  = L2_u / np.mean(np.sum(np.square(u_true),axis =1),axis =0) , L1_u / np.mean(np.sqrt(np.sum(np.square(u_true),axis =1)),axis =0)
    L2_moment_title,L1_moment_title = 'Squared ','Norm '
    
    #Calculate scaled moment (u/ u_0) errors
    L2_uoveru0_vals,L1_uoveru0_vals =  np.divide(L2_u_vals.reshape((L2_u_vals.shape[0],1)),np.square(u_true[:,0].reshape((u_true[:,0].shape[0],1)))),\
                                        np.divide(L1_u_vals.reshape((L1_u_vals.shape[0],1)),u_true[:,0].reshape((u_true[:,0].shape[0],1)))
    L2_uoveru0_vals,L1_uoveru0_vals = L2_uoveru0_vals.reshape((L2_uoveru0_vals.shape[0],1)),L1_uoveru0_vals.reshape((L1_uoveru0_vals.shape[0],1))
    L2_uoveru0_title,L1_uoveru0_title = 'Squared u_0 Scaled ','Norm u_0 Scaled '
    L2_moment_title,L1_moment_title = 'Squared ','Norm ' 
    
    if L1 == True:
        if N ==1:
            """
            L1_uerr_plot = L1_u_vals.reshape((L1_u_vals.shape[0],1))
            
            L1_scaled_u_graph =  np.hstack((u_true2d,L1_uoveru0_vals))
            
            L1_uerr_graph = np.hstack((u_true2d,L1_uerr_plot))
        
            plot_heatmap(L1_scaled_u_graph,50,title = L1_uoveru0_title + 'Moment Error: Log10 Average',\
                         xlabel = 'u0',ylabel = 'u1',savedir = savefolder+method+'momentu0heat'+saveid+'.eps')
            plot_heatmap(L1_uerr_graph,50,title = L1_moment_title + 'Moment Error: Log10 Average',\
                         xlabel = 'u0',ylabel = 'u1',savedir = savefolder+method+'momentheat'+saveid+'.eps')
            """
            pass
    elif L1 == False:
        if N ==1:
            """
            L2_uerr_plot = L2_u_vals.reshape((L2_u_vals.shape[0],1))
            L2_scaled_u_graph =  np.hstack((u_true2d,L2_uoveru0_vals))
            L2_uerr_graph = np.hstack((u_true2d,L2_uerr_plot))
            
            plot_heatmap(L2_scaled_u_graph,50,title = L2_uoveru0_title + 'Moment Error: Log10 Average',\
                         xlabel = 'u0',ylabel = 'u1',savedir = savefolder+method+'momentu0heat'+saveid+'.eps')
            plot_heatmap(L2_uerr_graph,50,title = L2_moment_title + 'Moment Error: Log10 Average',\
                         xlabel = 'u0',ylabel = 'u1',savedir = savefolder+method+'momentheat'+saveid+'.eps')
            """
            pass
            
            
    #Table for detailed moment errors
    L2_u0_vals,L2_u1_vals = np.square(u_pred[:,0]-u_true[:,0]), np.square(u_pred[:,1]-u_true[:,1])
    rel_u0_vals,rel_u1_vals = np.divide(np.square(u_pred[:,0]-u_true[:,0]),\
                                                      np.square(u_true[:,0])),np.divide(np.square(u_pred[:,1]-u_true[:,1]),np.square(u_true[:,1]))
    L2_u0,L2_u1 = np.mean(L2_u0_vals), np.mean(L2_u1_vals)
    rel_u0_err, rel_u1_err = np.mean(rel_u0_vals), np.mean(rel_u1_vals)
    L2_u_norm = np.mean(np.sum(np.square(u_true),axis  =1),axis = 0)
    if N ==2: 
        L2_u2_vals = np.square(u_pred[:,2]-u_true[:,2])
        rel_u2_vals = np.divide(L2_u2_vals,np.square(u_true[:,2]))
        L2_u2 = np.mean(L2_u2_vals)
        rel_u2_err = np.mean(rel_u2_vals)
        
    if N ==1:
        u_cols = ['MSE u_0','MSE u_1','MSE(u_0)/L2sq(u_true)','MSE(u_1)/L2sq(u_true)','MSRE u_0','MSRE u_1']
        u_vals = [['{:.2e}'.format(L2_u0),'{:.2e}'.format(L2_u1), '{:.2e}'.format(L2_u0 / L2_u_norm),'{:.2e}'.format(L2_u1 / L2_u_norm),\
                       '{:.2e}'.format(rel_u0_err),'{:.2e}'.format(rel_u1_err)]]
    if N ==2: 
        u_cols = ['MSE u_0','MSE u_1','MSE u_2','MSE(u_0)/L2sq(u_true)','MSE(u_1)/L2sq(u_true)','MSE(u_2)/L2sq(u_true)','MSRE u_0','MSRE u_1','MSRE u_2']
        u_vals = [['{:.2e}'.format(L2_u0),'{:.2e}'.format(L2_u1),'{:.2e}'.format(L2_u2),\
                   '{:.2e}'.format(L2_u0 / L2_u_norm),'{:.2e}'.format(L2_u1 / L2_u_norm),'{:.2e}'.format(L2_u2 / L2_u_norm),\
                       '{:.2e}'.format(rel_u0_err),'{:.2e}'.format(rel_u1_err),'{:.2e}'.format(rel_u2_err)]]
        
    u_Table = pd.DataFrame(u_vals,columns = u_cols)
    print('\n\n','2dTest Domain u-Error Breakdown in Latex: \n',u_Table.to_latex(),'\n')
    print('\n\n','2dTest Domain u-Error Breakdown:','\n',tabulate(u_Table,headers = 'keys',tablefmt = 'psql'))
    
    ### Gradient (Alpha) Error:
    L2_alpha_vals,L1_alpha_vals = np.sum(np.square(grad_pred - grad_true),axis = 1), np.sqrt(np.sum(np.square(grad_pred-grad_true),axis = 1))
    L2_alpha, L1_alpha = np.mean(L2_alpha_vals,axis = 0), np.mean(L1_alpha_vals,axis = 0)
    L2_alpha_norm,L1_alpha_norm = np.mean(np.sum(np.square(grad_true),axis = 1)), np.mean(np.sqrt(np.sum(np.square(grad_true),axis = 1)))
    L2_alpha_title,L1_alpha_title = 'Squared ', 'Norm ' 
    
    if L1 == True:
        if N ==1:
            L1_alpha_plot = L1_alpha_vals.reshape((L1_alpha_vals.shape[0],1))
            L1_alpha_graph = np.hstack((u_true2d,L1_alpha_plot))
            plot_heatmap(L1_alpha_graph,50,title  = L1_alpha_title + 'Alpha Error: Log10 Average',\
                         xlabel = 'u0',ylabel = 'u1',savedir = savefolder+method+'alphaheat'+saveid+'.eps')
        
    elif L1 == False:
        if N ==1:
            """
            L2_alpha_plot = L2_alpha_vals.reshape((L2_alpha_vals.shape[0],1))
            L2_alpha_graph = np.hstack((u_true2d,L2_alpha_plot))
            plot_heatmap(L2_alpha_graph,50,title = L2_alpha_title +  'Alpha Error: Log10 Average',\
                         xlabel = 'u0',ylabel = 'u1',savedir = savefolder+method+'alphaheat'+saveid+'.eps')
            """
            pass
        
    ### Alpha_0 & Alpha_1 Error : L2 and relative MSE values only 
    alpha0_vals,alpha1_vals = np.square(grad_pred[:,0]-grad_true[:,0]), np.square(grad_pred[:,1]-grad_true[:,1])
    alpha0_plot,alpha1_plot = alpha0_vals.reshape((alpha0_vals.shape[0],1)),alpha1_vals.reshape((alpha1_vals.shape[0],1))
    rel_alpha0_vals,rel_alpha1_vals = np.divide(np.square(grad_pred[:,0]-grad_true[:,0]),\
                                                      np.square(grad_true[:,0])),np.divide(np.square(grad_pred[:,1]-grad_true[:,1]),np.square(grad_true[:,1]))
    L2_alpha0, L2_alpha1 = np.mean(alpha0_vals), np.mean(alpha1_vals)
    mean_rel_alpha0, mean_rel_alpha1 = np.mean(rel_alpha0_vals), np.mean(rel_alpha1_vals)
    alpha0_title,alpha1_title = 'Squared ', 'Squared '
    alpha0err_graph,alpha1err_graph = np.hstack((u_true2d,alpha0_plot)), np.hstack((u_true2d,alpha1_plot))
    
    
    ###Table for detailed alpha errors
    if N ==2:
        alpha2_vals = np.square(grad_pred[:,2]-grad_true[:,2])
        L2_alpha2 = np.mean(alpha2_vals)
        mean_rel_alpha2 = 'N/A'
        try:
            rel_alpha2_vals = np.divide(alpha2_vals,np.square(grad_true[:,2]))
            mean_rel_alpha2 = np.mean(rel_alpha2_vals)
        except RuntimeWarning:
            print('True alpha_2 = 0 encountered; supplying N/A for pointwise relative error')
    
    if N == 1:
        """
        plot_heatmap(alpha0err_graph,50,title  = alpha0_title + 'Alpha-0 Error: Log10 Average',\
                     xlabel = 'u0',ylabel = 'u1',savedir = savefolder+method+'alpha0heat'+saveid+'.eps')
        plot_heatmap(alpha1err_graph,50,title = alpha1_title + 'Alpha-1 Error: Log10 Average',\
                     xlabel = 'u0',ylabel = 'u1',savedir = savefolder+method+'alpha1heat'+saveid+'.eps')
        """
        pass
    
    
    if N == 1:
        Alpha_cols = ['MSE alpha-0','MSE alpha-1','MSE(alpha-0)/L2sq(alpha_true)','MSE(alpha-1)/L2sq(alpha_true)','MSRE alpha-0','MSRE alpha-1']
    
        Alpha_vals = [['{:.2e}'.format(L2_alpha0),'{:.2e}'.format(L2_alpha1), '{:.2e}'.format(L2_alpha0 / L2_alpha_norm),'{:.2e}'.format(L2_alpha1 / L2_alpha_norm),\
                       '{:.2e}'.format(mean_rel_alpha0),'{:.2e}'.format(mean_rel_alpha1)]]
    if N ==2:
        
        Alpha_cols = ['MSE alpha-0','MSE alpha-1','MSE alpha-2','MSE(alpha-0)/L2sq(alpha_true)','MSE(alpha-1)/L2sq(alpha_true)',\
                      'MSE(alpha-2)/L2sq(alpha_true)','MSRE alpha-0','MSRE alpha-1','MRSE alpha-2']
        Alpha_vals = [['{:.2e}'.format(L2_alpha0),'{:.2e}'.format(L2_alpha1),'{:.2e}'.format(L2_alpha2),\
                       '{:.2e}'.format(L2_alpha0 / L2_alpha_norm),'{:.2e}'.format(L2_alpha1/L2_alpha_norm),'{:.2e}'.format(L2_alpha2/L2_alpha_norm) ,\
                       '{:.2e}'.format(mean_rel_alpha0),'{:.2e}'.format(mean_rel_alpha1),\
                       [ '{:.2e}'.format(mean_rel_alpha2) if type(mean_rel_alpha2) == float else mean_rel_alpha2][0]]]
    
    Alpha_Table = pd.DataFrame(Alpha_vals,columns = Alpha_cols)
    print('\n\n','2dTest Domain Alpha-Error Breakdown in Latex: \n',Alpha_Table.to_latex(),'\n')
    print('\n\n','2dTest Domain Alpha-Error Breakdown:','\n',tabulate(Alpha_Table,headers = 'keys',tablefmt = 'psql'))
                            #mpltable(Alpha_Table,width = 0.12,savedir = savefolder+method+'alphatable'+saveid+'.eps')
    
    
    
    if compute_conv == True:
        
        if N ==1:
            
            num_negdef_points = np.sum((conv_pred < 0))
            print('Number of negative definite test points','\n', num_negdef_points,'\n\n')
                #Pretty-print the error values
        if N ==2:
            
            num_negdef_points = np.sum(( conv_pred[:,0] < 0) + (conv_pred[:,1] <0 ))
            print('Number of negative definite test points:','\n',num_negdef_points,'\n\n')
        
    else:
        
        num_negdef_points = 'N/A'
    
    #For Paper: Select MSE & RMSE -Table
    if N == 2:
        
        Select_MSE_Cols = ['MSE r-h','MSE r-u','MSE r-u0','MSE r-u1','MSE r-u2', 'MSE r-alpha']
        
        Select_RMSE_Cols = ['RMSE r-h','RMSE r-u','RMSE r-u0','RMSE r-u1','RMSE r-u2', 'RMSE r-alpha']
        
        Select_MSE_vals = [['{:.2e}'.format(L2_funcrel),'{:.2e}'.format(L2_u_rel),'{:.2e}'.format(L2_u0_spec),\
                      '{:.2e}'.format(L2_u1_spec),'{:.2e}'.format(L2_u2_spec),'{:.2e}'.format(L2_alpha / L2_alpha_norm)]]
    
    
        Select_RMSE_vals = [['{:.2e}'.format(np.sqrt(L2_funcrel)),'{:.2e}'.format(np.sqrt(L2_u_rel)),'{:.2e}'.format(np.sqrt(L2_u0_spec)),\
                      '{:.2e}'.format(np.sqrt(L2_u1_spec)),'{:.2e}'.format(np.sqrt(L2_u2_spec)),'{:.2e}'.format(np.sqrt(L2_alpha / L2_alpha_norm))]]
    
        Select_MSE_Table = pd.DataFrame(Select_MSE_vals,columns = Select_MSE_Cols)
        
        Select_RMSE_Table = pd.DataFrame(Select_RMSE_vals,columns = Select_RMSE_Cols)
        
        print('\n\n Select 2dTest RMSE Table in Latex:\n\n ',Select_RMSE_Table.to_latex(),'\n\n')
    
    if N == 1:
        
        Select_MSE_Cols = ['MSE r-h','MSE r-u','MSE r-u0','MSE r-u1', 'MSE r-alpha']
        
        Select_RMSE_Cols = ['RMSE r-h','RMSE r-u','RMSE r-u0','RMSE r-u1', 'RMSE r-alpha']
        
        Select_MSE_vals = [['{:.2e}'.format(L2_funcrel),'{:.2e}'.format(L2_u_rel),'{:.2e}'.format(L2_u0_spec),\
                      '{:.2e}'.format(L2_u1_spec),'{:.2e}'.format(L2_alpha / L2_alpha_norm)]]
    
        Select_RMSE_vals = [['{:.2e}'.format(np.sqrt(L2_funcrel)),'{:.2e}'.format(np.sqrt(L2_u_rel)),'{:.2e}'.format(np.sqrt(L2_u0_spec)),\
                      '{:.2e}'.format(np.sqrt(L2_u1_spec)),'{:.2e}'.format(np.sqrt(L2_alpha / L2_alpha_norm))]]
    
        Select_MSE_Table = pd.DataFrame(Select_MSE_vals,columns = Select_MSE_Cols)
        
        Select_RMSE_Table = pd.DataFrame(Select_RMSE_vals,columns = Select_RMSE_Cols)
        
        print('\n\n Select 2d Test RMSE Table in Latex: \n\n',Select_RMSE_Table.to_latex())
        
        
        if append_results:
            
            results_to_append = [saveid,size,np.sqrt(L2_funcrel),np.sqrt(L2_u_rel),np.sqrt(L2_u0_spec),\
                                 np.sqrt(L2_u1_spec),np.sqrt(L2_alpha / L2_alpha_norm)]
            if method == 'net':
                
                with open(domain+'_All_Results_'+data_id+'.pickle','rb') as handle:
                    
                    result_list = pickle.load(handle)
                    
                result_list.append(results_to_append)
            
                with open(domain+'_All_Results_'+data_id+'.pickle','wb') as newhandle:
                    
                    pickle.dump(result_list,newhandle)
                    
            elif method == 'spline':
                
                with open(domain+'_All_Results_'+'Spline'+data_id+'.pickle','rb') as handle:
                    
                    result_list = pickle.load(handle)
                    
                result_list.append(results_to_append)
            
                with open(domain+'_All_Results_'+'Spline'+data_id+'.pickle','wb') as newhandle:
                    
                    pickle.dump(result_list,newhandle)
    
    ### Table for MSE: 
    MSE_cols = ['MSE h','MSE r-h','MSE u','MSE r-u','MSE r-u0','MSE alpha','MSE r-alpha','H < 0']
    MSE_vals = [['{:.2e}'.format(L2_func),'{:.2e}'.format(L2_funcrel),\
                    '{:.2e}'.format(L2_u),'{:.2e}'.format(L2_u_rel),'{:.2e}'.format(L2_u0_spec),'{:.2e}'.format(L2_alpha),\
                    '{:.2e}'.format(L2_alpha / L2_alpha_norm),num_negdef_points]]
    MSE_table = pd.DataFrame(MSE_vals,columns = MSE_cols)
    print('\n\n','2d Test Domain Error Summary in LaTeX: \n',MSE_table.to_latex(),'\n')
    print('\n\n','2d Test Domain Error Summary:','\n',tabulate(MSE_table,headers = 'keys',tablefmt = 'psql'))
                    #mpltable(MSE_table,width = 0.12,savedir = savefolder+method+'metricstable'+saveid+'.pdf')
                
    
    print('\n\n','MSE := (1/N)*\sum_{i} (x_i^{pred}-x_i^{true})^{2} \n MSRE := (1/N)* \sum_{i} [(x_i^{pred} - x_i^{true}) / (x_^{i} true)]^{2}')
    