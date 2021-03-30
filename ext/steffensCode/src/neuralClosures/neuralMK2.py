'''
Derived network class "MK2" for the neural entropy closure.
Author: Steffen Schotthöfer
Version: 0.0
Date 29.10.2020
'''
from .neuralBase import neuralBase
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras


class neuralMK2(neuralBase):
    '''
    MK2 Model: Train u to alpha
    Training data generation: c) sample u, train on entropy functional
    Loss function: Entropy functional derivative is loss
    '''

    def __init__(self, polyDegree=0, folderName="testFolder", optimizer='adam'):
        if (folderName == "testFolder"):
            tempString = "MK1_N" + str(polyDegree)
        else:
            tempString = folderName

        self.opt = optimizer
        self.polyDegree = polyDegree
        self.model = self.createModel()
        self.filename = "models/" + tempString
        self.trainingData = ()

    def createModel(self):
      
        return 0

    def selectTrainingData(self):
        return [True, False, False]

    def selectTrainingData(self):
        return [True, False, True]

    def trainingDataPostprocessing(self):
        # dublicate u
        self.trainingData.append(self.trainingData[0])
        print("Moments U dublicated")
        return 0

    ### helper functions

    # Custom Loss
    def custom_loss1dMB(self, u_input, alpha_pred):  # (label,prediciton)
        return 4 * math.pi * tf.math.exp(alpha_pred * np.sqrt(1 / (4 * np.pi))) - alpha_pred * u_input

    # Custom Loss
    def custom_loss1dMBPrime(self):  # (label,prediciton)
        def loss(u_input, alpha_pred):
            return 0.5 * tf.square(
                4 * math.pi * np.sqrt(1 / (4 * np.pi)) * tf.math.exp(alpha_pred * np.sqrt(1 / (4 * np.pi))) - u_input)

        return loss
