'''
Derived network class "MK3" for the neural entropy closure.
Author: Steffen Schotthöfer
Version: 0.0
Date 29.10.2020
'''
from .neuralBase import neuralBase
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow import Tensor
import csv

'''
import pandas as pd
import sphericalquadpy as sqp
from joblib import Parallel, delayed
import multiprocessing
'''


class neuralMK3(neuralBase):
    '''
    MK1 Model: Train u to alpha
    Training data generation: b) read solver data from file
    Loss function:  MSE between alpha and real_alpha
    '''

    def __init__(self, polyDegree=0, folderName="testFolder", optimizer='adam'):
        if (folderName == "testFolder"):
            tempString = "MK1_N" + str(polyDegree)
        else:
            tempString = folderName
        self.polyDegree = polyDegree
        self.opt = optimizer
        self.model = self.createModel()
        self.filename = "models/" + tempString

    def createModel(self):
        return 0

    def selectTrainingData(self):
        return [True, True, False]

    def getIdxSphericalHarmonics(self, k, l):
        # Returns the global idx from spherical harmonics indices
        return l * l + k + l
