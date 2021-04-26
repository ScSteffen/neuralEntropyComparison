'''
File: Model framework
Author: William Porteous and Steffen Schotthöfer
Date: 9.04.2021
'''

### imports
import tensorflow  as tf
from os import path, makedirs, walk
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
### imports of own scripts
from src.IcnnClosure import createIcnnClosure
from src.EcnnClosure import createEcnnClosure, HaltWhen

"""
This callback needs to be moved but is useful for hitting a certain threshold 
"""


class ModelFrame:

    def __init__(self, architecture=0, shapeTuple = (20,5), lossChoices=0, inputDim=1,quad = None):
        """constructor"""
        
        """
        Create quadrature object and pass to createxxxClosure
        """
        
        self.nWidth = shapeTuple[0]
        self.nLength = shapeTuple[1]
        
        self.architecture = architecture
        
        self.lossChoices = lossChoices
        
        self.N = inputDim

        self.saveFolder = "models/losscombi_" + str(lossChoices) + "/" + str(trainableParamBracket)
        
        if (architecture == 0):  # Steffen (synonymous with modelchoice) ==> ICNN
            
            self.saveFolder = self.saveFolder + "/icnn"
            self.model = createIcnnClosure(inputDim, shapeTuple,
                                           lossChoices = lossChoices)  # @Steffen: Model creation Function here

        elif (architecture ==1):  # Will: (model choice is ECNN)
        
            self.saveFolder = self.saveFolder + "/ecnn"
            self.model = createEcnnClosure(inputDim = inputDim, shapeTuple = (self.nWidth,self.nLength),\
                                           lossChoices = lossChoices,Quad = quad)  # @Will: Model creation Function here
            
        #Alternate behavior: 
            #return uncompiled model and compile here (i.e. assign the losses), possibly changing
            #ICNN to accomodate hessian loss (but with weight zero)
            
        else:
            
            raise ValueError('architecture must be zero or 1')

    def showModel(self):
        self.model.summary()
        return 0

    def loadWeights(self):
        usedFileName = self.saveFolder + '/model.h5'
        self.model.load_weights(usedFileName)
        print("Model loaded from file ")
        return 0

    def trainingProcedure(self, trainData, curr):
        
        """
        Will's Note:
            
        (1) We need to pass a list of zeros for hess_train 
            
        
        (2) Some argument should control the 
        choice of tensorflow callbacks 
        that we deploy during training.
        
        (3) Another argument should control the number of epochs and the batch size 
        
        
        Can 2 and 3 be done with the parser?
        """
        
        u_train,alpha_train,h_train,hess_train = trainData
        
        ### TODO
        #   @WILL
        if curr == 0:
            
            #We only use this at the moment
            num_epochs = int(1.5*(1e+04))
            initial_lr = float(1e-3)
            drop_rate = (num_epochs / 3)
            
            
            mt_patience = int(num_epochs/ 10)
            min_delta = stop_tol / 10
            
            stop_tol = 1e-7
            batch_size = int(100)
            
        elif curr == 1:
            
            num_epochs = 1000
            batch_size = 128

            initial_lr = float(1e-3)
            mt_patience = int(num_epochs/ 10)
            stop_tol = 1e-8
            min_delta = stop_tol / 10
            drop_rate = (num_epochs / 3)
            
        elif curr == 2:
            
            num_epochs = 1000
            batch_size = 128

            initial_lr = float(1e-3)
            mt_patience = int(num_epochs/ 10)
            stop_tol = 1e-8
            min_delta = stop_tol / 10
            drop_rate = (num_epochs / 3)
            
        
        def step_decay(epoch):
            
            step_size = initial_lr*np.power(10,(-epoch/drop_rate))
            
            return step_size 
        
        #this callback good to go 
        LR = LearningRateScheduler(step_decay)
        
        MC =  ModelCheckpoint(self.saveFolder+'/best_model.h5', \
                              monitor = 'val_output_'+str(self.lossChoices)+'_loss',\
                              save_best_only = True,\+
                              save_weights_only = False,\
                              mode = 'min',verbose=1)
        
        HW = HaltWhen('val_output_'+str(self.lossChoices)+'_loss',stop_tol)
        
        ES = EarlyStopping(monitor = 'val_output_'+str(self.lossChoices)+'_loss',mode = 'min',\
                              verbose = 1,patience = mt_patience,\
                              min_delta = 1e-8)
        
        CSV_Logger = self.createCSVLoggerCallback()
        
        callback_list = [MC,ES,HW,CS,LR,CSV_Logger]
      
        """
        Model bool / string to switch training for Steffen or Will
        because Will needs 1 more argument - the hessian target values 
        """"
        if model.arch == 'icnn':
        
            moment_history = self.model.fit(x=u_train, y=[h_train, alpha_train, u_train],
                                            validation_split=0.20,
                                            epochs=num_epochs, batch_size=batch_size, callbacks=callbackList)
        elif model.arch = 'ecnn':
            
            moment_history = self.model.fit(x=u_train, y=[h_train, alpha_train, u_train,hess_train],
                                            validation_split=0.20,
                                            epochs=num_epochs, batch_size=batch_size, callbacks=callbackList)
            
        return moment_history

    def createCSVLoggerCallback(self):
        '''
        dynamically creates a csvlogger
        '''
        # check if dir exists
        if not path.exists(self.saveFolder + '/historyLogs/'):
            makedirs(self.saveFolder + '/historyLogs/')

        # checkfirst, if history file exists.
        logFile = self.saveFolder + '/historyLogs/history_001_'
        count = 1
        while path.isfile(logFile + '.csv'):
            count += 1
            logFile = self.saveFolder + '/historyLogs/history_' + str(count).zfill(3) + '_'

        logFile = logFile + '.csv'
        # create logger callback
        csv_logger = tf.keras.callbacks.CSVLogger(logFile)

        return csv_logger

    def errorAnalysis(self):
        # TODO
        # @WIll
        return 0
    
if __name__ == "__main__":
    pass 
