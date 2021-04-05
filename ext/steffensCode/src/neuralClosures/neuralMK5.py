'''
Derived network class "MK4" for the neural entropy closure.
It features the ICNN approach by Amos et al.
Author: Steffen Schotthöfer
Version: 0.0
Date 17.12.2020
'''
from .neuralBase import neuralBase
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import Tensor
from tensorflow.keras.constraints import NonNeg


class neuralMK5(neuralBase):
    '''
    MK4 Model: Train u to alpha
    Training data generation: b) read solver data from file: Uses C++ Data generator
    Loss function:  MSE between h_pred and real_h
    '''

    def __init__(self, polyDegree=0, folderName="testFolder", optimizer='adam'):
        if (folderName == "testFolder"):
            tempString = "MK5_N" + str(polyDegree)
        else:
            tempString = folderName
        self.polyDegree = polyDegree
        # --- Determine inputDim by MaxDegree ---
        if (self.polyDegree == 0):
            self.inputDim = 1
        elif (self.polyDegree == 1):
            self.inputDim = 4
        else:
            raise ValueError("Polynomial degeree higher than 1 not supported atm")

        self.opt = optimizer
        self.model = self.createModel()
        self.filename = "models/" + tempString

    
    def createModel(self):
        return 0

    def selectTrainingData(self):
        return [True, False, True]
