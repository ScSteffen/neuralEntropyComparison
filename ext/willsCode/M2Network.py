# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:11:24 2020

@author: Will
"""
#Import numerical python and file type packages
import h5py
import numpy as np 
import math 
from tabulate import tabulate
import pickle

#Import required tensorflow modules; need tensorflow and not only 
#tensorflow.keras.backend since keras.backend does not have GradientTape 
import tensorflow as tf
from tensorflow.keras import initializers 
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

#Import custom modules for data-generation, visualization, and error analysis
#from entropyapprox_tools import entropyapprox_choosedata, entropyapprox_choosemake
from vizualization_tools import mpltable
from MN_Duality_Tools import moment_vector, Entropy 
from net_stat_tools import plotlosscurve, quickstats_scaled, runerranalysis_scaled, plot_heatmap
#from dualfixedpoly1Q_opts import dualfixedpoly1Q_opts 
from optstuffset import optstuffset 
from getquad import getquad 

#Wrap RuntimeWarning as an error so we can catch it when expected
import warnings 
warnings.simplefilter('error',RuntimeWarning)


class M2ConvexNet(tf.keras.Model):
    
    def __init__(self,N,nNode,nLayer,Q,**opts):
        
        super(M2ConvexNet, self).__init__()
        
        #Specify architecture and input shape
        
        self.N = N
        self.nNode = nNode
        self.nLayer = nLayer
        self.p = tf.constant(Q.p,dtype= float)
        self.w = tf.constant(Q.w,dtype = float)
        
        self.m0 = self.p[0,:]
        self.m1 = self.p[1,:]
        self.m2 = self.p[2,:]
        
        #Define variance for initialization
        
        #1. This is a modified Kaiming inititalization with a first-order taylor expansion of the 
        #softplus activation function (see S. Kumar "On Weight Initialization in
        #Deep Neural Networks").
    
        self.input_stddev = np.sqrt((1/N)*(1/((1/2)**2))*(1/(1+np.log(2)**2)) )
        self.hidden_stddev = np.sqrt((1/nNode)*(1/((1/2)**2))*(1/(1+np.log(2)**2)) )
        
        """
        
        #2. This is the He or Kaiming initialization 
        self.input_stddev = np.sqrt((2/N))
        self.hidden_stddev = np.sqrt((2/nNode))
        
        #Define the input layer, hidden layers, and output layer
        
        """
        #Standard variance for initialization: the "he" or "kaiming" init
        self.input_layer = Dense(nNode, use_bias = True,kernel_initializer =\
                  initializers.RandomNormal(mean = 0.,stddev = self.input_stddev),\
                  bias_initializer = initializers.Zeros())
        
        
        self.hidden_layers = dict() 
        
        self.bn_layers = dict()
        
        self.bn_layers['bn_input'] = BatchNormalization(axis = 0)
        
        for i in range(self.nLayer):
            
            self.hidden_layers['hidden_'+str(i)] = Dense(nNode,use_bias = True,kernel_initializer =\
                      initializers.RandomNormal(mean = 0.,stddev = self.hidden_stddev),\
                      bias_initializer = initializers.Zeros(),)
            
            self.bn_layers['bn_'+str(i)] = BatchNormalization(axis = 0)
    
        self.output_layer = Dense(1, use_bias = True,kernel_initializer = \
              initializers.RandomNormal(mean = 0.,stddev = self.hidden_stddev),\
              bias_initializer = initializers.Zeros(),name = 'function')
        
        self.activation = Activation('softplus')
        
        #batch normalization needs op conversion for function gradient to be accessible- not used here 
            
    def identity_func(self,tensor):
        
        return tensor
    
    def alpha0surface(self,alpha_N):
        
        checked_alpha_N = tf.debugging.check_numerics(alpha_N,\
                        message = 'input tensor checking error',name = 'checked')
        clipped_alpha_N = tf.clip_by_value(checked_alpha_N,\
                        clip_value_min = -50,clip_value_max = 50,name = 'checkedandclipped')
        
        Ga1a2 = tf.math.exp(tf.tensordot(clipped_alpha_N,self.p[1:,:],axes = 1))
        
        integral_Ga1a2 = tf.tensordot(Ga1a2,self.w,axes = 1)
        
        alpha0_pred = - tf.math.log(integral_Ga1a2)
    
        return alpha0_pred 
    
    #Define moment reconstruction function 

    def moment_func(self,alpha,tol = 1e-8):
        
        #Check the predicted alphas for +/- infinity or nan - raise error if found
        checked_alpha = tf.debugging.check_numerics(alpha,\
                        message = 'input tensor checking error',name = 'checked')
        
        #Clip the predicted alphas below the tf.exp overflow threshold 
        clipped_alpha = tf.clip_by_value(checked_alpha,\
                        clip_value_min = -50,clip_value_max = 50,name = 'checkedandclipped')
        
        #Calculate the closed density function at each point along velocity domain
        G_alpha = tf.math.exp(tf.tensordot(clipped_alpha[:,:],self.p[:,:],axes = 1))
        
        #Pointwise-multiply moment vector by closed denity along velocity axis
        m0G_alpha = tf.multiply(G_alpha,self.m0)
        m1G_alpha = tf.multiply(G_alpha,self.m1)
        m2G_alpha = tf.multiply(G_alpha,self.m2)
        
        #Compute integral by quadrature (dot-product with weights along velocity axis)
        u0 = tf.tensordot(m0G_alpha,self.w, axes = 1)
        u1 = tf.tensordot(m1G_alpha,self.w, axes = 1)
        u2 = tf.tensordot(m2G_alpha,self.w, axes = 1)
        
        #Stack-moments together
        moment_pred = tf.stack([u0,u1,u2],axis = 1)
        
        return moment_pred
        
    def call(self,x,training = False):
        """
        Defines network function. Can be adapted to have different paths 
        for training and non-training modes (not currently used).
        
        At each layer, applies, in order: (1) weights & biases, (2) batch normalization 
        (current: commented out), then (3) activation.
        
        Inputs: 
            
            (x,training = False,mask = False)
        
        Returns:
            
            returns [h(x),alpha(x),u(x),hess(h)(x)]
        """
            
        x = Lambda(self.identity_func,name = "input")(x)
        
        with tf.GradientTape(persistent = True) as hess_tape:
            
            hess_tape.watch(x) 
            
            with tf.GradientTape() as grad_tape: 
                
                grad_tape.watch(x) 
            
                y = self.input_layer(x)
                
                y = self.activation(y)
            
                #y = self.bn_layers['bn_input'](y)
                
                for i in range(self.nLayer):
                    
                    y = self.hidden_layers['hidden_'+str(i)](y)
                    
                    y = self.activation(y)
                    
                    #y = self.bn_layers['bn_'+str(i)](y)
                
                net = self.output_layer(y) 
            
            d_net = grad_tape.gradient(net,x)
            
            d_net = Lambda(self.identity_func,name = "d_net")(d_net)
        
        hess = hess_tape.batch_jacobian(d_net,x)
        
        dets = \
        tf.math.multiply(hess[:,0,0],hess[:,1,1]) - tf.math.multiply(hess[:,1,0],hess[:,0,1])
        
        hess_11 = hess[:,0,0]
        
        detpa  = tf.stack([dets,hess_11],axis = 1)
            
        alpha_N = d_net

        #Explicit quadrature equality for alpha_0; these are exact (up to quadrature) alpha_0 values for predicted alpha_N
        #alpha_0 = tf.expand_dims(self.alpha0surface(alpha_N),axis = 1)
        
        #Contraint equation for alpha_0
        alpha_0 = tf.constant(1,dtype = tf.float32) + net - tf.expand_dims(tf.reduce_sum(tf.multiply(x,d_net),axis = 1),axis = 1)
        
        alpha_out = tf.concat([alpha_0,alpha_N],axis = 1)
         
        return [net,alpha_out,self.moment_func(alpha_out),detpa]
    

class HaltWhen(tf.keras.callbacks.Callback):
    def __init__(self,quantity,tol):
        """
        Should be used in conjunction with 
        the saving criterion for the model; otherwise 
        training will stop without saving the model with quantity <= tol
        """
        super(HaltWhen,self).__init__()
        if type(quantity) == str:
            self.quantity = quantity
        else:
            raise TypeError('HaltWhen(quantity,tol); quantity must be a string for a monitored quantity')
        self.tol = tol
    def on_epoch_end(self,epoch,logs = None):
        if epoch > 1:
            if logs.get(self.quantity) < self.tol:
                print('\n\n',self.quantity,' has reached',logs.get(self.quantity),' < = ',self.tol,'. End Training.')
                self.model.stop_training = True
        else:
            pass
        
if __name__ == "__main__":
    
    """
    1. Which procedures to run?
    """
    
    #run_moment_training: calls model.fit for the network, with any loss function components you choose,
    #but conditions the stopping criteria on the alpha loss, or u_loss (in priciple can be any loss) - see section 
    run_moment_training = False

    #run_conv_training is deprecated. This is for 'staged' training
    run_conv_training = False
    
    #quicksummary: plots the loss curves, summarizes training-data errors in a table (
    #for which it prints the latex output), plots the network function; 
    #saves all this to the directory savefolder
    quicksummary = True
    
    #Run error analysis and save to savefolder
    error_analysis = True
    
    """
    2. Define basic training routine params. Choose quantities required for the model architecture to compute.
    """
    percent_train = 80
    
    batchsize = 20

    numepochs_moment = int( (1.5)*(10**2) )
    
    d2h_scale_init = float(1)
    
    conv_step = (1/100)
    
    #What should the loss components be? This will set initial loss weights to float(0) or float(1) acccordingly.
    enforce_func,enforce_grad,enforce_moment,enforce_conv = [True,False,True,False]
    
    #Choose weights for moment training routine 
    mt_weights = [float(enforce_func),float(enforce_grad),float(enforce_moment),float(enforce_conv)]
    
    """
    ct_weights = [float(enforce_func),float(enforce_grad),float(enforce_moment),float(True)]
    """
    
    #If L2-reg is enabled, tell plotlosscurve and quickstats_scaled to show it or not
    show_reg =  False
    
    """
    3. Where to save information? 
    """
    #Choose a folder to save images generated by error analysis and quicksummary
    savefolder = '/Users/Will/Desktop/Entropy_Approx_Notes/'
    
    #Choose a unique identifier for this network. For 1 <= num < = 25, saveid = '1dnum' have been used (on Will's device).
    saveid = 'M2_4by50_4'
    
    #Select training and test data files

    train_data_id = '7'
    test_data_id = '8'
    
    train_data_file = 'M2TrainData'+train_data_id+'.pickle'
    test_data_file = 'M2TestData'+test_data_id+'.pickle'

    train_table_file = 'M2TrainDataTable'+train_data_id+'.pickle'
    test_table_file = 'M2TestDataTable'+test_data_id+'.pickle'
        
    #Choose explicit filepath for model checkpoints (as written, this will be your current working directory). This is 
    #not necessarily the same as your 'savefolder'
    momentcheckpointpath = 'M2moment'+saveid+'.h5'
    
    """
    4. Choose network architecture and make a gaussian-initialized instance of ScaledConvexNN
    """
    
    nQuadpts = 30
    Quad = getquad('lgwt',nQuadpts,-1,1,2)
    N = 2
    nLayers = 4
    nWidth = 50
    layers = [N,*[nWidth for i in range(nLayers)],1]
    model = M2ConvexNet(N,nWidth,nLayers,Quad) 
        
    """
    5. Make (if needed) and choose the training data. Note: for compatibility with model, the data shape must satisfy:
                    (1) u_train_proj.shape[1] = N
                    (2) alpha_train.shape[1] = N+1
                    (3) len(h_train.shape) = 1
                    (4) len(conv_train.shape) = 1
    and identically for the validation data.
    """
    
    with open(train_data_file,'rb') as datahandle:
        
        alpha_data,moment_data,entropy_data = pickle.load(datahandle)
        
    with open(train_table_file,'rb') as handle:
        
        train_table = pickle.load(handle)
        
    print('\n\n','Training Domain Info: \n\n',tabulate(train_table,\
                                                       headers = 'keys',tablefmt = 'psql'))
    
    u_data_target = moment_data[:,:]
    
    h_data = entropy_data[:]
    
    grad_data = alpha_data[:,:]
    
    #Get indices for training and validation
    DataSize = u_data_target.shape[0]
    Idx = np.arange(0,DataSize)
    np.random.shuffle(Idx)
    train_size = round((percent_train/100)*DataSize)
    val_size = DataSize - train_size 
    
    #Separate data into training and validation
    h_train = h_data[Idx[:train_size]]
    h_valid = h_data[Idx[train_size:]]
    
    u_train = u_data_target[Idx[:train_size],:]
    u_valid = u_data_target[Idx[train_size:],:]
    
    alpha_train = alpha_data[Idx[:train_size],:]
    alpha_valid = alpha_data[Idx[train_size:],:]
    
    #First, simply trained for alpha_train_proj. as true and d_net as pred. Now, in accordance with Scaling Network, 
    #training with alpha_train as true and alpha_out as pred.
    
    alpha_train_proj = alpha_train[:,1:]
    alpha_valid_proj = alpha_valid[:,1:]
    
    conv_train = np.zeros(shape = (train_size,))
    conv_valid = np.zeros(shape = (val_size,))
    #Remove 0th component of the moment vector
    u_train_proj = np.reshape(u_train[:,1:],(u_train.shape[0],2))
    u_valid_proj = np.reshape(u_valid[:,1:],(u_valid.shape[0],2))
    
    #Calculate mean-L2 norms of vectors with all components
    u_train_norm = np.mean(np.sum(np.square(u_train),axis= 1),axis = 0)
    alpha_train_norm = np.mean(np.sum(np.square(alpha_train),axis = 1))
    
    """
    7. Define the loss functions, the optimizer, and compile the model.
    """
    
    def func_loss(y_true,y_pred):
    
        loss_val = tf.keras.losses.MSE(y_true,y_pred)
        
        return loss_val
    
    def alpha_loss(alpha_true,alpha_pred):
        
        loss_val = float(3)*tf.keras.losses.MeanSquaredError()(alpha_true,alpha_pred)
        
        return loss_val
        
    def moment_loss(moment_true,moment_pred):
        """
        We want to compare the mean squared difference: 
            Mean_{i}(||u_true_{i} - u_pred_{i}||^{2})
        
        The expression in true_normsq is the operation we want to apply to the difference.
            
        To do this with tf.keras.losses.MeanSquaredError(), we must multiply output by
        N+1, since it takes mean over both dimension 1 and dimension 0.
        """
        true_normsq  = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.square(moment_true),axis = 1),axis =  0)
        
        loss_val = ((N+1)/true_normsq)*tf.keras.losses.MeanSquaredError()(moment_true,moment_pred)
        
        return loss_val
            
    d2h_scale = K.variable(d2h_scale_init,dtype = tf.float32)
            
    def conv_loss(conv_true,conv_pred):
        
        d2h_min = d2h_scale*tf.math.minimum(conv_pred,conv_true)
        
        d2h_loss = tf.keras.losses.MSE(d2h_min,conv_true)
        
        return d2h_loss
    
    #Define optimizer 
    opt = Adam()
 
    #Call model.predict over a dummy input as an alternative to model.build - this "builds" the model.
    test_point = np.array([[0.5,0.7],[-0.5,-0.7],[0.01,-0.01],[0.9,-0.9]],dtype= float)
    test_point = tf.constant(test_point,dtype = tf.float32)
    test_output = model.predict(test_point)
    print(test_output[1])
    

    #Compile the model
    model.compile(optimizer=opt,loss= {'output_4':conv_loss,'output_3':moment_loss,\
                                       'output_2':alpha_loss,'output_1':func_loss},\
                                       loss_weights = mt_weights)

    """
    8. Define custom training behavior: callbacks and learning rate 
    """
    
    initial_lr = float(1e-3)
    
    mt_patience = 1000
    
    stop_tol = 1e-7
    
    drop_rate = (numepochs_moment / 3)
    
    def step_decay(epoch):
        
        step_size = initial_lr*np.power(10,(-epoch/drop_rate))
        
        return step_size 
    
    LR = LearningRateScheduler(step_decay)
    
    mt_MC =  ModelCheckpoint(momentcheckpointpath, \
                          monitor = 'val_output_3_loss',\
                          save_best_only = True,\
                          save_weights_only = False,\
                          mode = 'min',verbose=1)
    
    mt_HW = HaltWhen('val_output_3_loss',stop_tol)
    
    mt_ES = EarlyStopping(monitor = 'val_output_3_loss',mode = 'min',verbose = 1,patience = mt_patience)
    
    mt_callback_list = [mt_MC,mt_ES,mt_HW,LR]
        
    """
    9. Run moment training routine, restore 'best' model state, and apply selected analysis tools 
    """

    if run_moment_training == True:
           
        print('\n\n',' ///////// Begin Moment Training /////////// ','\n\n')

        mtrain_history = model.fit(x = u_train_proj,y = [h_train,alpha_train,u_train,conv_train],\
                            validation_data = (u_valid_proj,[h_valid,alpha_valid,u_valid,conv_valid]),\
                            epochs = numepochs_moment,batch_size = batchsize,callbacks = mt_callback_list)
        
        
        model.load_weights(momentcheckpointpath)
        
        if quicksummary == True:
            
            model.load_weights(momentcheckpointpath)
            
            h_pred_train,alpha_pred_train,u_pred_train,conv_pred_train = model.predict(u_train_proj)
            
            plotlosscurve(mtrain_history,mt_weights,show_reg,\
                          savefolder = savefolder, saveid = saveid+'m')
            
            quickstats_scaled([h_pred_train,h_train],[alpha_pred_train,alpha_train],[u_pred_train,u_train],conv_pred_train,\
                              savefolder = savefolder, saveid = saveid+'m',method = 'net', N = 2)
        
        if error_analysis == True:
            
            model.load_weights(momentcheckpointpath)
            
            with open(test_data_file,'rb') as handle:
                
                alpha_truetest,u_truetest,h_truetest = pickle.load(handle) 
        
            #Get the predictions according to our scaling technique 
            u_truetest_proj = np.divide(u_truetest[:,1:],(u_truetest[:,0])[:,None])
            
            u_predtest = np.zeros((u_truetest.shape[0],3))
            h_pred_test_proj,alpha_pred_test_proj,u_toss,conv_pred_test_proj = model.predict(u_truetest_proj)
            
            h_pred_test_proj = h_pred_test_proj.reshape((h_pred_test_proj.shape[0],))
            h_pred_test = u_truetest[:,0]*(h_pred_test_proj + np.log(u_truetest[:,0]))
            
            alpha_pred_test = alpha_pred_test_proj[:,:]
            alpha_pred_test[:,0] = alpha_pred_test[:,0] + np.log(u_truetest[:,0])
            
            u_pred_test = model.moment_func(alpha_pred_test).numpy()
            
            #Collect final predictions, run the error analysis 
            truevals = [h_truetest,alpha_truetest,u_truetest]
            
            predvals = [h_pred_test,alpha_pred_test,u_pred_test,conv_pred_test_proj]
        
            runerranalysis_scaled(truevals,True,predvals,savefolder,saveid,method = 'net',L1 = False,N = 2)
        
    