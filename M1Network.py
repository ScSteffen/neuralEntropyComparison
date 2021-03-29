# -*- coding: utf-8 -*-
"""
Created on Tue Aug 4 15:52:47 2020

Last edit: Fri Sep 4 2021

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
#from vizualization_tools import mpltable
from MN_Duality_Tools import moment_vector, Entropy 
from net_stat_tools import plotlosscurve, quickstats_scaled, runerranalysis_scaled, plot_heatmap
#from dualfixedpoly1Q_opts import dualfixedpoly1Q_opts 
#from optstuffset import optstuffset 
#from getquad import getquad 

#Wrap RuntimeWarning as an error so we can catch it when expected
import warnings 
warnings.simplefilter('error',RuntimeWarning)


class M1ConvexNN(tf.keras.Model):
    
    def __init__(self,inputShape,nNode,nLayer,**opts):
        
        super(M1ConvexNN, self).__init__()
        
        #Specify architecture and input shape
        
        self.inputShape = inputShape
        self.nNode = nNode
        self.nLayer = nLayer
        self.nNode = nNode
        
        
        #1. This is a modified Kaiming inititalization with a first-order taylor expansion of the 
        #softplus activation function (see S. Kumar "On Weight Initialization in
        #Deep Neural Networks").
        
        #Extra factor of (1/1.1) added inside sqrt to suppress inf for 1 dimensional inputs 
        self.input_stddev = np.sqrt( (1/1.1)*(1/inputShape)*(1/((1/2)**2))*(1/(1+np.log(2)**2)) )
        self.hidden_stddev = np.sqrt((1/1.1)*(1/nNode)*(1/((1/2)**2))*(1/(1+np.log(2)**2)) )
        
        
        #2. This is the He or Kaiming initialization 
        #self.input_stddev = np.sqrt((2/N))
        #self.hidden_stddev = np.sqrt((2/nNode))
        
        """
        #Define the input layer, hidden layers, and output layer
        
        """
        
        
        
        #Define the input layer, hidden layers, and output layer
        self.input_layer = Dense(nNode, use_bias = True,kernel_initializer =\
                  initializers.RandomNormal(mean = 0.,stddev = self.input_stddev),\
                  bias_initializer = initializers.Zeros())
        
                  #kernel_regularizer = regularizers.l2(1e-3))
        
        
        self.hidden_layers = dict()
        
        self.bn_layers = dict()
        
        for i in range(self.nLayer):
            
            self.hidden_layers['hidden_'+str(i)] = Dense(nNode,use_bias = True,kernel_initializer =\
                      initializers.RandomNormal(mean = 0.,stddev = self.hidden_stddev),\
                      bias_initializer = initializers.Zeros())
            
                      #kernel_regularizer = regularizers.l2(1e-3))
            
            self.bn_layers['hidden_'+str(i)] = BatchNormalization(axis = 0)
            
        self.output_layer = Dense(1, use_bias = True,kernel_initializer = \
                  initializers.RandomNormal(mean = 0.,stddev = self.hidden_stddev),\
                  bias_initializer = initializers.Zeros(),\
                  activation = "linear",name = 'function')
        
                  #kernel_regularizer = regularizers.l2(1e-3))
        
        self.activation = Activation('softplus')
        #batch normalization needs op conversion for function gradient to be accessible- not used here 
            
    def identity_func(self,tensor):
        
        return tensor
    
    #Define moment reconstruction function 
    def moment_func(self,tensor,tol = 1e-10):
        
        #Assign components of gradient tensor 
        alpha_0 = tensor[:,0]
        alpha_1 = tensor[:,1]
        
        alpha_0 = tf.debugging.check_numerics(alpha_0,\
                                              message = 'custom alpha_0_pred value error',name = 'alpha_0_checked')
        alpha_1 = tf.debugging.check_numerics(alpha_1,\
                                              message = 'custom alpha_1_pred value error',name = 'alpha_1_checked')
        
        #Compute scale in reconstruction map; applied at the end 
        scale = 2*tf.math.exp(alpha_0)
        
        #Compute list of bools with bools[i] = False if abs(alpha_1[i]) < tol
        bools = tf.math.greater(tf.math.abs(alpha_1),\
                                   tf.constant([tol],dtype = tf.float32))
        
        #Convert that list to floats; required for tf multiplication
        bools = tf.cast(bools,dtype = tf.float32)
        
        #Make bool_op[i] = 1 for abs(alpha_1[i]) < tol 
        bools_op = tf.constant([1],dtype = tf.float32)-bools
    
        #Make new alpha_1s so that alpha_1s[i] = 2*tol + alpha_1[i]
        #for i such that bool[i] = False
        shift = tf.multiply(tf.constant(float(3)*tol,dtype = tf.float32),bools_op) 
        alpha_1s = tf.add(alpha_1,shift)
        
        alpha_1s = tf.clip_by_value(alpha_1s,clip_value_min = -70,clip_value_max = 70, name = 'alpha_1s_clipped')
        
        #Compute moment_0 from shifted values; then, correct the shifted moments 
        moment_0 = tf.math.divide(tf.math.sinh(alpha_1s),alpha_1s,name = 'moment_0_divide')
        
        #Correct moment_0: if alpha_1s[i] != alpha_1[i] then moment_0 = 1
        moment_0 = tf.add(tf.multiply(moment_0,bools),bools_op,\
                          name = 'modified_moment_0')
        
        moment_0 = tf.debugging.check_numerics(moment_0,message = 'moment_0 value error',\
                                               name = 'moment_0_checked')
        
        #Compute moment_1 from shifted values; then, correct the shifted moments
        sinh_checked = tf.debugging.check_numerics(tf.math.sinh(alpha_1s),name = 'sinh_vals',\
                                             message = 'sinh value error')
        cosh_checked = tf.debugging.check_numerics(tf.math.cosh(alpha_1s),name = 'cosh_vals',message = 'cosh value error')
        mult_checked =  tf.debugging.check_numerics(tf.multiply(alpha_1s,cosh_checked),name = 'mult_vals',\
                                              message = 'mult value error')
        
        top = tf.subtract(mult_checked,sinh_checked)
        top = tf.debugging.check_numerics(top,message = 'moment_1 numerator value err',\
                                          name = 'modified_moment_1_numerator')
        bottom = tf.math.square(alpha_1s)
        
        moment_1 = tf.math.divide(top,bottom,name = 'moment_1_divide')
        #Correct moment_1: if alpha_1s[i] != alpha_1[i] then moment_0[i] = 0
        
        moment_1 = tf.multiply(moment_1,bools,name = 'modified_moment_1')
        
        tf.debugging.check_numerics(moment_1,message = 'moment_1 value error',\
                                    name = 'moment_1_checked')
        
        
        #Scale moments according to the mapping:
        
        moment_0 = tf.multiply(scale,moment_0)
        moment_1 = tf.multiply(scale,moment_1)
        
        #Join the moments together 
        moment_out = tf.stack([moment_0,moment_1],axis = 1)
        
        return moment_out
        
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
                
                #y = self.bn(y,training = True)
                
                y = self.activation(y)
                
                for i in range(self.nLayer):

                    y = self.hidden_layers['hidden_'+str(i)](y)
                    
                    #y = self.bn(y, training = True)
                    
                    y = self.activation(y)
                
                net = self.output_layer(y) 

            d_net = grad_tape.gradient(net,x)
            
            d_net = Lambda(self.identity_func,name = "d_net")(d_net)
     
        d2_net = hess_tape.gradient(d_net,x)
         
        alpha_0 = tf.constant(1,dtype = tf.float32) + net - tf.multiply(x,d_net)
         
        alpha_1 = d_net 

        alpha = tf.stack([alpha_0,alpha_1],axis = 1)[:,:,0]
         
        return [net,alpha,self.moment_func(alpha),d2_net]
            
class AdaptivePenalty(tf.keras.callbacks.Callback):
    def __init__(self,w_det,w_a,c_0,m_0,s):
        """
        A callback which is adaptively increases
        the convex penatly value for a 2-d network
        """
        super(AdaptivePenalty,self).__init__()
        self.c_0 = c_0
        self.m_0 = m_0
        self.s = s
        self.w_det = w_det
        self.w_a = w_a
    def on_train_begin(self,epoch,logs = None):
        if self.c_0 > 0:
            self.model.loss_weights = [1.,1.,1.,(self.m_0/self.c_0)]
        elif self.c_0 == 0:
            self.model.loss_weights = [1.,1.,1.,1.]
    def on_epoch_end(self,epoch,logs = None):
        """
        if np.less(moment_loss,1e-1):
            pass
            #self.model.loss_weights =\
            #[float(1),float(0),float(1),float(1)]
        else:
            pass 
            #self.model.loss_weights = \
            #[float(1),float(0),float(1),float(0)]
        """
        
        if epoch >= 2:
            val_moment_loss = logs.get('val_output_3_loss')
            if val_moment_loss <= 1e-2:
                K.set_value(self.w_det,K.get_value(self.w_det + self.s))
                K.set_value(self.w_a,K.get_value(self.w_a + self.s))
                print('\n','ConvWeights:',self.w_a.numpy(),self.w_det.numpy(),'\n')
            else:
                K.set_value(self.w_det,K.get_value(self.w_det))
                K.set_value(self.w_a,K.get_value(self.w_a))
                print('\n','ConvWeights:',self.w_a.numpy(),self.w_det.numpy(),'\n')
            
class ConvexSave(tf.keras.callbacks.Callback):
    """
    The purpose of this class is to save, 
    from the subset of all convex networks which 
    occur during training, the one with lowest 
    validation moment error.
    
    Note that this only has to save weights as 
    often as the standard callback.
    """
    def __init__(self,filepath):
        super(ConvexSave,self).__init__()
        self.filepath = filepath
    def on_train_begin(self,logs = None):
        self.best_val_moment = np.Inf
    def on_epoch_end(self,epoch,logs = None):
        #Only save the improvement if it is convex
        if epoch >= 1:
            current_val_moment = logs.get('val_output_3_loss')
            val_convloss = logs.get('val_output_4_loss')
            train_convloss = logs.get('output_4_loss')
            if (val_convloss <= 1e-12) and (train_convloss <= 1e-12):
                if current_val_moment <= self.best_val_moment:
                    self.best_val_moment = current_val_moment
                    self.model.save_weights(self.filepath)
                    print('ConvexSave: val_output_3_loss improved to ', self.best_val_moment,'\n')
                else:
                    print('ConvexSave: val_output_3_loss did not improve from ',self.best_val_moment)
            else:
                pass
#                print(' Network is nonconvex. ')
                
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
        if logs.get(self.quantity) < self.tol:
            print('\n\n',self.quantity,' has reached',logs.get(self.quantity),' < = ',self.tol,'. End Training.')
            self.model.stop_training = True
        
if __name__ == "__main__":
    
    """
    1. Which procedures to run?
    """
    
    #run_moment_training: calls model.fit for the network, with any loss function components you choose,
    #but conditions the stopping criteria on the alpha loss, or u_loss (in priciple can be any loss) - see section 

    run_moment_training =  True

    #run_conv_training is deprecated. This is for 'staged' training, which we have not used in 1D
    run_conv_training = False
    
    #quicksummary: plots the loss curves, summarizes training-data errors in a table (
    #for which it prints the latex output), plots the network function; 
    #saves all this to the directory savefolder
    
    quicksummary = True
    
    #Run error analysis and save to savefolder
    
    error_analysis_1d = True
    
    error_analysis_2d = True
    

    """
    2. Define basic training routine params. Choose quantities required for the model architecture to compute.
    """
    percent_train = 90
    
    batchsize = 50
    
    numepochs_moment = int((1.5)*(10**4))

    #Basic params for conv training - not available for 1dScaled
    numepochs_conv = 0
    
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
    #savefolder = '../trainedNN/'
    
    #Choose a unique identifier for this network. For 1 <= num < = 25, saveid = '1dnum' have been used (on Will's device).
    
    saveid = 'M1_4by45_new_3'

    #Choose explicit filepath for model checkpoints (as written, this will be your current working directory. Note this is 
    #not necessarily the same as your savefolder
    
    momentcheckpointpath = 'M1moment'+saveid+'.h5'
    
    
    best_convex_checkpoint = 'M1_convex_checkpoint'+saveid+'.h5'
    
    """
    convcheckpointpath = 'scaledconvmodel'+saveid+'.h5'
    """
    
    """
    4. Choose network architecture and make a gaussian-initialized instance of M1ConvexNN
    """
    
    N = 1
    nLayers = 4
    nWidth = 45
    total_size = (nLayers + 3)*(nWidth) +  nLayers*(nWidth**2) + 1
    layers = [N,*[nWidth for i in range(nLayers)],1]
    model = M1ConvexNN(N,nWidth,nLayers)
     
    
    """
    5. Make (if needed) and choose the training data, as identified by the 'saveappend' keyword.
    
    Default: saveappend = 1dtrain2
    Note: for compatibility with model, the data shape must satisfy 
    
                    (1) u_train_proj.shape[1] = N
                    (2) alpha_train.shape[1] = N+1
                    (3) len(h_train.shape) = 1
                    (4) len(conv_train.shape) = 1
                    
    and identically for the validation data.
    """
    
    #Standard train-test pair is currently ('2','2')
    
    train_data_id  = 'M1_set_A'
    test_1d_id  = 'M1_set_A'
    test_2d_id = 'M1_set_A'
    
    train_data_file = 'M1TrainData'+train_data_id+'.pickle'
    test_1d_data_file = 'M11dTestData'+test_1d_id+'.pickle'
    test_2d_data_file = 'M12dTestData'+test_2d_id + '.pickle'

    train_table_file =   'M1TrainDataTable'   +   train_data_id  + '.pickle'
    test_1d_table_file = 'M11dTestDataTable'  +  test_1d_id     + '.pickle'
    test_2d_table_file = 'M12dTestDataTable'  +   test_2d_id     + '.pickle'
    
    with open(train_data_file,'rb') as handle:
        alpha_data,u_data,h_data = pickle.load(handle)
        
    with open(train_table_file,'rb') as handle:
        train_table = pickle.load(handle)
        
    print('\n\n','Training Domain Info: \n\n',tabulate(train_table,\
                                                       headers = 'keys',tablefmt = 'psql'))
    
    DataSize = u_data.shape[0]
    
    Idx = np.arange(0,DataSize)
    np.random.shuffle(Idx)
    
    train_size = round((percent_train/100)*DataSize)
    val_size = DataSize - train_size 
    
    conv_train = np.zeros(shape = (train_size,))
    conv_valid = np.zeros(shape = (val_size,))
    
    u_train = u_data[Idx[:train_size],:]
    u_valid = u_data[Idx[train_size:],:]
    
    u1_train = np.reshape(u_train[:,1],(u_train.shape[0],1)) #Formerly u_train_proj
    u1_valid = np.reshape(u_valid[:,1],(u_valid.shape[0],1)) #Formerly u_valid_proj
    #u_train_norm = np.mean(np.sum(np.square(u_train),axis= 1),axis = 0)
    
    alpha_train = alpha_data[Idx[:train_size],:]
    alpha_valid = alpha_data[Idx[train_size:],:]
    #alpha_train_norm = np.mean(np.sum(np.square(alpha_train),axis = 1))
    
    h_train = h_data[Idx[:train_size]]
    h_valid = h_data[Idx[train_size:]]
    
    """
    6. Define the loss functions, the optimizer, and compile the model.
    """
    
    #Define loss functions for loss dictionary. 
    def func_loss(y_true,y_pred):
    
        loss_val = tf.keras.losses.MSE(y_true,y_pred)
        
        return loss_val
    
    def alpha_loss(alpha_true,alpha_pred):
        
        loss_val = (N+1)*tf.keras.losses.MSE(alpha_true,alpha_pred)
        
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
        
        loss_val = ((N+1)/true_normsq)*(tf.keras.losses.MeanSquaredError()(moment_true,moment_pred))
        
        return loss_val
            
    d2h_scale = K.variable(d2h_scale_init,dtype = tf.float32)
            
    def conv_loss(conv_true,conv_pred):
        
        d2h_min = d2h_scale*tf.math.minimum(conv_pred,conv_true)
        
        d2h_loss = tf.keras.losses.MSE(d2h_min,conv_true)
        
        return d2h_loss
    
    #Define optimizer 
    opt = Adam()

    #Call model.predict over a dummy input as an alternative to model.build - this "builds" the model.
    test_point = np.array([0.5,0.7,-0.5,-0.7,0.01,-0.01,0.9,-0.9],dtype= float)
    test_point = test_point.reshape((test_point.shape[0],1))
    test_point = tf.constant(test_point,dtype = tf.float32)
    test_output = model.predict(test_point)

    model.compile(optimizer=opt,loss= {'output_4':conv_loss,'output_3':moment_loss,\
                                       'output_2':alpha_loss,'output_1':func_loss},\
                                       loss_weights = mt_weights)
    
    """
    8. Define custom training behavior: callbacks and learning rate 
    """
    
    initial_lr = float(1e-3)
    
    mt_patience = int(numepochs_moment / 10)
    
    stop_tol = 1e-8
    
    min_delta = stop_tol / 10
    
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
    
    mt_CS = ConvexSave(best_convex_checkpoint)
    
    mt_ES = EarlyStopping(monitor = 'val_output_3_loss',mode = 'min',\
                          verbose = 1,patience = mt_patience,\
                          min_delta = 1e-8)
    
    mt_callback_list = [mt_MC,mt_ES,mt_HW,mt_CS,LR]
        
    """
    9. Run moment training routine, restore 'best' model state, and apply selected analysis tools 
    """

    if run_moment_training == True:
           
        print('\n\n',' ///////// Begin Moment Training /////////// ','\n\n')

        moment_history = model.fit(x = u1_train,y = [h_train,alpha_train,u_train,conv_train],\
                            validation_data = (u1_valid,[h_valid,alpha_valid,u_valid,conv_valid]),\
                            epochs = numepochs_moment,batch_size = batchsize,callbacks = mt_callback_list)
        
        
        model.load_weights(momentcheckpointpath)
        
        h_pred_train,alpha_pred_train,u_pred_train,conv_pred_train = model.predict(u1_train)
            
        if quicksummary == True:
            
            model.load_weights(momentcheckpointpath)
        
        
            h_pred_train,alpha_pred_train,u_pred_train,conv_pred_train = model.predict(u1_train)
            
            plotlosscurve(moment_history,mt_weights,show_reg,\
                          savefolder = savefolder, saveid = saveid+'m')
            
            quickstats_scaled([h_pred_train,h_train],[alpha_pred_train,alpha_train],[u_pred_train,u_train],conv_pred_train,\
                              savefolder = savefolder, saveid = saveid+'m',data_id = train_data_id,size = total_size, method = 'net',\
                              domain = 'Train',N = 1,append_results = True,plot3d = True)
            
        if error_analysis_1d == True:
        
            with open(test_1d_table_file,'rb') as handle:
                test_table_1d = pickle.load(handle)
                
            print('\n\n','1d Test Domain Info: \n\n',tabulate(test_table_1d,\
                                                               headers = 'keys',tablefmt = 'psql'))
            
            with open(test_1d_data_file,'rb') as handle:
                alpha_true_test_1d,u_true_test_1d,h_true_test_1d = pickle.load(handle)
                
            h_pred_test_1d,alpha_pred_test_1d,u_pred_test_1d,conv_pred_test_1d = model.predict(u_true_test_1d[:,1][:,None])
            
            quickstats_scaled([h_pred_test_1d,h_true_test_1d],[alpha_pred_test_1d,alpha_true_test_1d],\
                              [u_pred_test_1d,u_true_test_1d],conv_pred_test_1d,\
                              savefolder = savefolder,saveid = saveid+'m',data_id = test_1d_id,size= total_size,method = 'net',\
                              domain = '1dTest',N = 1,append_results = True,plot1d = True)
        
        if error_analysis_2d == True:
            
            with open(test_2d_table_file,'rb') as handle:
                test_table = pickle.load(handle)
        
            print('\n\n','Test Domain Info: \n\n',tabulate(test_table,\
                                                       headers = 'keys',tablefmt = 'psql'))
            
            with open(test_2d_data_file,'rb') as handle:
                alpha_true_test_2d,u_true_test_2d,h_true_test_2d = pickle.load(handle)
                
            u_true_test_proj  = np.divide(u_true_test_2d[:,1],u_true_test_2d[:,0])[:,None]
            h_pred_test_proj,alpha_pred_test_proj,u_toss,conv_pred_test_proj = model.predict(u_true_test_proj) #Formerly input test_ratios
            
            #Extend predictions to the 2d domain with relationships which are convexity preserving:
            h_pred_test_proj = h_pred_test_proj.reshape((h_pred_test_proj.shape[0],))
            h_pred_test_2d = u_true_test_2d[:,0]*(h_pred_test_proj + np.log(u_true_test_2d[:,0]))
            
            alpha_pred_test_2d = alpha_pred_test_proj[:,:]
            alpha_pred_test_2d[:,0] = alpha_pred_test_2d[:,0] + np.log(u_true_test_2d[:,0])
            
            u_pred_test_2d = model.moment_func(alpha_pred_test_2d).numpy()
            
            true_vals = [h_true_test_2d,alpha_true_test_2d,u_true_test_2d]
            pred_vals = [h_pred_test_2d,alpha_pred_test_2d,u_pred_test_2d,conv_pred_test_proj]

            runerranalysis_scaled(true_vals,True,pred_vals,savefolder,saveid = saveid + 'm',data_id = test_2d_id,size = total_size,\
                                  method = 'net',domain = '2dTest',L1 = False,append_results = True,N = 1)
            
            """
            if alpha_method == int(0):
                
                grad_pred_test[:,0] = np.log(testinfo.moment_domain[:,0]) + alpha_scaled[:,0]
                grad_pred_test[:,1] = alpha_scaled[:,1]
                
                u_predict = model.moment_func(tf.constant(grad_pred_test,dtype = tf.float32)).numpy()
                
                h_pred_test = Entropy(grad_pred_test)
    
                conv_scaled = np.zeros(u_scaled.shape[0],)
    
                #saveid+m
                runerranalysis_scaled(testinfo,True,h_pred_test,grad_pred_test,u_predict,conv_scaled,savefolder,'1m',method = 'net',L1 = False)
        
            if alpha_method == int(1):
                
                grad_scaled = alpha_scaled[:,1]
                
                #Popualte predicted gradient:
                grad_pred_test[:,0] = 1 + np.log(testinfo.moment_domain[:,0]) + h_scaled - np.multiply(test_ratios[:,0],grad_scaled)
                scaler = np.divide(1,testinfo.moment_domain[:,0])
                grad_pred_test[:,1] = np.multiply(scaler,grad_scaled)
                
                #Populate predicted moment 
                u_predict = model.moment_func(tf.constant(grad_pred_test,dtype = tf.float32)).numpy()
                
                #Populate predicted func: 
                h_pred_test = np.multiply(testinfo.moment_domain[:,0],(np.log(testinfo.moment_domain[:,0])+h_scaled))
            """
    """
    10. Make callbacks for conv training routine:
    """
    
    """
    model.load_weights(momentcheckpointpath)
        
    #Compute initial conv loss value, and pass it to the adaptive penalty
    
    conv_out = tf.constant(model.predict(u_train)[3])
    conv_true = tf.constant(0.,shape = conv_out.shape)
    convloss_initial = conv_loss(conv_true,conv_out)
    
    u_out = tf.constant(model.predict(u_train)[2],dtype = tf.float32)
    u_true = tf.constant(u_train,dtype = tf.float32)
    MSE = tf.keras.losses.MeanSquaredError()
    utruenorm = tf.math.reduce_mean(tf.math.reduce_sum(tf.math.square(u_true),axis = 1),axis =  0)
    uloss_initial = (2/utruenorm)*MSE(u_out,u_true)
    
    ct_AS = ConvexSave(convcheckpointpath)
    
    ct_AP = AdaptivePenalty(W_det,W_a,convloss_initial,uloss_initial,conv_step)
    
    ct_callback_list = [ct_AS,ct_AP,mt_ES]

    
    
    11. Run Conv Training Routine
    
    
    if run_conv_training == True:
        
        print('\n\n',' ///////// Begin Conv Training /////////// ','\n\n')
        
        conv_history = model.fit(x = u_train,y = [h_train,alpha_train,u_train,conv_train],\
                                validation_data = (u_valid,[h_valid,alpha_valid,u_valid,conv_valid]),\
                                epochs = numepochs_conv,batch_size = batchsize,callbacks = ct_callback_list)
        
            
        h_pred,alpha_pred,u_pred,conv_pred = model.predict(u_train)
            
        
        if quicksummary == True:
        
            plotlosscurve(conv_history,ct_weights,\
                          savefolder = savefolder, saveid = saveid+'cs')
        
            quickstats_scaled([h_pred,h_train],[alpha_pred,alpha_train],[u_pred,u_train],conv_pred,\
                       savefolder = savefolder, saveid = saveid+'cs')
            
        #Perform inference and analyze over a test domain
        if error_analysis == True:
            
            
            h_pred_test,grad_pred_test,u_predict,conv_test_pred = model.predict(testinfo.moment_domain)
            det_pred = conv_test_pred[:,0]
            a_pred = conv_test_pred[:,1]
            
            runerranalysis_scaled(testinfo,True,h_pred_test,grad_pred_test,u_predict,conv_test_pred,savefolder,saveid+'c')     
    """