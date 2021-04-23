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

    def __init__(self, architecture=0, trainableParamBracket=1, model_losses=0, inputDim=1):
        """constructor"""
        
        """
        Create quadrature object and pass to createxxxClosure
        """

        self.saveFolder = "models/bracket_" + str(trainableParamBracket) + "/losscombi_" + str(model_losses)
        if (architecture == 0):  # Steffen (synonymous with modelchoice) ==> ICNN
            self.saveFolder = self.saveFolder + "/icnn"
            self.model = createIcnnClosure(inputDim, trainableParamBracket,
                                           model_losses)  # @Steffen: Model creation Function here

        elif (architecture ==1):  # Will: (model choice is ECNN)
        
            self.saveFolder = self.saveFolder + "/ecnn"
            self.model = createEcnnClosure(inputDim, trainableParamBracket,
                                           model_losses)  # @Will: Model creation Function here
            
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

    def trainingProcedure(self, u_train, alpha_train, h_train,hess_train,**opts):
        
        """
        Will's Note:
            
        (1) We need to pass a list of zeros for hess_train 
            
        
        (2) Some argument should control the 
        choice of tensorflow callbacks 
        that we deploy during training.
        
        (3) Another argument should control the number of epochs and the batch size 
        
        
        Can 2 and 3 be done with the parser?
        """
        
        ### TODO
        #   @WILL
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
        
        will_LR = LearningRateScheduler(step_decay)
        
        will_MC =  ModelCheckpoint(self.saveFolder, \
                              monitor = 'val_output_3_loss',\
                              save_best_only = True,\
                              save_weights_only = False,\
                              mode = 'min',verbose=1)
        
        will_HW = HaltWhen('val_output_3_loss',stop_tol)
        
        will_ES = EarlyStopping(monitor = 'val_output_3_loss',mode = 'min',\
                              verbose = 1,patience = mt_patience,\
                              min_delta = 1e-8)
        
        will_callback_list = [mt_MC,mt_ES,mt_HW,mt_CS,LR]

        # Create Callbacks
        mc_best = tf.keras.callbacks.ModelCheckpoint(self.saveFolder + '/best_model.h5', monitor='loss', mode='min',
                                                     save_best_only=True, verbose=1)
        csv_logger = self.createCSVLoggerCallback()

        callbackList = [mc_best, csv_logger]
    
        """
        Model bool / string to switch training for Steffen or Will
        because Will needs 1 more argument - the hessian target values 
        """"
        
        moment_history = self.model.fit(x=u_train, y=[h_train, alpha_train, u_train],
                                        validation_split=0.01,
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
