'''
File: Model framework
Author: William Porteous and Steffen Schotthöfer
Date: 9.04.2021
'''

### imports
import tensorflow  as tf
from os import path, makedirs, walk

### imports of own scripts
from src.IcnnClosure import createIcnnClosure
from src.EcnnClosure import createEcnnClosure


class ModelFrame:

    def __init__(self, architecture=0, trainableParamBracket=1, model_losses=0, inputDim=1):
        """constructor"""

        self.saveFolder = "models/bracket_" + str(trainableParamBracket) + "/losscombi_" + str(model_losses)
        if (architecture == 0):  # Steffen (synonymous with modelchoice) ==> ICNN
            self.saveFolder = self.saveFolder + "/icnn"
            self.model = createIcnnClosure(inputDim, trainableParamBracket,
                                           model_losses)  # @Steffen: Model creation Function here

        else:  # Will: (model choice is ECNN)
            self.saveFolder = self.saveFolder + "/ecnn"
            self.model = createEcnnClosure(inputDim, trainableParamBracket,
                                           model_losses)  # @Will: Model creation Function here

    def showModel(self):
        self.model.summary()
        return 0

    def loadWeights(self):
        usedFileName = self.saveFolder + '/model.h5'
        self.model.load_weights(usedFileName)
        print("Model loaded from file ")
        return 0

    def trainingProcedure(self, u_train, alpha_train, h_train):
        ### TODO
        #   @WILL
        num_epochs = 1000
        batch_size = 128

        # Create Callbacks
        mc_best = tf.keras.callbacks.ModelCheckpoint(self.saveFolder + '/best_model.h5', monitor='loss', mode='min',
                                                     save_best_only=True, verbose=1)
        csv_logger = self.createCSVLoggerCallback()

        callbackList = [mc_best, csv_logger]
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
