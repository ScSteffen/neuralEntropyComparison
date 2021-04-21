"""
WILLS MODEL GOES HERE
"""
import h5py
import numpy as np 
import math 
from tabulate import tabulate

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

def createEcnnClosure(inputDim,trainableParamBracket,model_losses,**kwargs):
    
    """
    What needs to be passed via trainableParamBracket is the following:
        N = inputDim
        nNode
        nLayers
        Q = Quadrature-Object
        loss choices: a set of bools telling us which losses to enforce 
        
    I've implemented this via kwargs for now, but would you be able to make this 
    switch over to trainableParamBracket? I have no clue how this argument works. 
    
    Also, is it correct that N  = inputDim?
    
    """
    N = inputDim
    nNode = kwargs['nNode']
    nLayer = kwargs['nLayer']
    
    if N > 1:
        #When N = 1 we don't need quadrature. Otherwise yes. 
        Quad = kwargs['Q']
    
    enforce_func,enforce_grad,enforce_moment,enforce_conv = kwargs['loss_choices']
    
    loss_weights = [float(enforce_func),float(enforce_grad),float(enforce_moment),float(enforce_conv)]
    
    """
    #1. Define a keras model via subclassing
    """
    if N == 2:
        #For now, conditional behavior built outside the model 
        #Once we know what we're doing, I can probably make a model class for variable N
        
        
        #1. Define a keras model via subclassing
        model =  M2ConvexNet(N,nNode,nLayer,Quad)
        """
        #2. "Build" the model. Here, calling model.predict() is used as an alternative to build. 
        #this is valid in the subclassing approach and, das I'm aware, elsewhere too. 
        """
        test_point = np.array([[0.5,0.7],[-0.5,-0.7],[0.01,-0.01],[0.9,-0.9]],dtype= float)
        test_point = tf.constant(test_point,dtype = tf.float32)
        test_output = model.predict(test_point)
        
        print('\n\n CHECK - Here is output of model.predict() for M2ConvexNet: ',\
              test_output, '\n\n')
        
    if N == 1:
        
        model = M1ConvexNet(N,nNode,nLayer)
        """
        #2. "Build" the model. Here, calling model.predict() is used as an alternative to build. 
        #this is valid in the subclassing approach and, das I'm aware, elsewhere too. 
        """
        test_point = np.array([0.5,0.7,-0.5,-0.7,0.01,-0.01,0.9,-0.9],dtype= float)
        test_point = test_point.reshape((test_point.shape[0],1))
        test_point = tf.constant(test_point,dtype = tf.float32)
        test_output = model.predict(test_point)
        print('\n\n CHECK - Here is output of model.predict() for M1ConvexNet:\n ',\
              test_output, '\n\n')
    
    """
    #3. Define the loss functions and their weights:
    """
    
    func_loss,alpha_loss,moment_loss,conv_loss = get_losses(N)
    
    enforce_func,enforce_grad,enforce_moment,enforce_conv = kwargs['loss_choices']
    
    loss_weights = [float(enforce_func),float(enforce_grad),float(enforce_moment),float(enforce_conv)]
    
    """
    #4. Define the optimizer 
    """
    
    opt = Adam()
    
    """
    #5. Compile the model
    """

    model.compile(optimizer=opt,loss= {'output_4':conv_loss,'output_3':moment_loss,\
                                   'output_2':alpha_loss,'output_1':func_loss},\
                                   loss_weights = loss_weights)
        
    return model


def get_losses(N,hess_scale_init = float(1)):
    """
    Defines loss functions needed for convex-training 
    """
    
    def func_loss(y_true,y_pred):

        loss_val = tf.keras.losses.MSE(y_true,y_pred)
        
        return loss_val

    def alpha_loss(alpha_true,alpha_pred):
    
        loss_val = (N+1)*tf.keras.losses.MeanSquaredError()(alpha_true,alpha_pred)
        
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
    
    hess_scale = K.variable(hess_scale_init,dtype = tf.float32)
            
    def conv_loss(conv_true,conv_pred):
        
        d2h_min = hess_scale*tf.math.minimum(conv_pred,conv_true)
        
        d2h_loss = tf.keras.losses.MSE(d2h_min,conv_true)
        
        return d2h_loss
    
    return func_loss,alpha_loss,moment_loss,conv_loss 


class M1ConvexNet(tf.keras.Model):
    
    def __init__(self,inputShape,nNode,nLayer,**opts):
        
        super(M1ConvexNet, self).__init__()
        
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



class M2ConvexNet(tf.keras.Model):
    #Model built via subclassing 
    
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


if __name__ == "__main__":
    
    pass 