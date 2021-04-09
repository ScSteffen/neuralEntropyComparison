'''
File: Model framework
Author: William Porteous and Steffen Schotthöfer
Date: 9.04.2021
'''

### imports
import tensorflow  as tf

### imports of own scripts
from IcnnClosure import createIcnnClosure
from EcnnClosure import createEcnnClosure

class ModelFrame:

    def __init__(self, author = 0, trainableParamBracket = 1, model_losses = 0, inputDim = 1):
        """constructor"""

        self.saveFolder = "models/bracket_"+str(trainableParamBracket)+"/losscombi_"+str(model_losses)
        if (author == 0): # Steffen (synonymous with modelchoice) ==> ICNN
            self.saveFolder  = self.saveFolder + "/icnn"
            self.model = createIcnnClosure(inputDim,trainableParamBracket,model_losses) # @Steffen: Model creation Function here

        else:   # ==> WILL
            self.saveFolder = self.saveFolder + "/ecnn"
            self.model = createEcnnClosure(inputDim,trainableParamBracket,model_losses) # @Will: Model creation Function here

    def showModel(self):
        self.model.summary()
        return 0

    def loadWeights(self):
        usedFileName =  self.saveFolder  + '/model.h5'
        self.model.load_weights(usedFileName)
        print("Model loaded from file ")
        return 0

    def trainingProcedure(self):
        ### TODO
            #       @WILL
        return 0

    def errorAnalysis(self ):
        # TODO
        #@WIll
        return 0
