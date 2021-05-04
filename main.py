'''
File: Main Script for comparison study
Author: William Porteous and Steffen Schotthöfer
Date: 16.03.2021
'''

### imports
from optparse import OptionParser

### imports of own scripts
from src.modelFrame import ModelFrame
from src import utils
from src.utils import AnalysisTools
import numpy as np
import os
import tensorflow as tf
from os import path
import pandas as pd
import numpy as np

pd.set_option('display.float_format', '{:.2e}'.format)
import pickle


def main():
    '''
    Main execution Skript to conduct the comparison study between
    
    Input Convex neural closure by Steffen Schotthöfer (iccnClosure)
    and
    
    Empirical Convex neural closure by Will Porteous  (eCnnClosure)
    '''
    print("---------- Start Network Training Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-a", "--architecture", dest="architecture", default=2,
                      help="0 : icnn\n 1 : ecnn\n 2 : both", metavar="AUTHOR")
    parser.add_option("-c", "--curriculum", dest="curr", default=0, \
                      help="integer between 0 and 2 for learning curriculum", \
                      metavar="CURRICULUM")
    parser.add_option("-d", "--degreeBasis", dest="degreeBasis", default=1,
                      help="degree of the basis",
                      metavar="DEGREE")
    parser.add_option("-e", "--evaluation", dest="evaluation", default=1,
                      help="evaluation and error analysis", metavar="EVALUATE")
    parser.add_option("-l", "--load", dest="load", default=1,
                      help="0: Dont load weights\n1:Load weights", metavar="LOAD")
    parser.add_option("-o", "--optimizationObjective", dest="losses", default=1,
                      help="combination of model losses (objective functions): \n"
                           "1 : [h] \n"
                           "2:  [h,alpha] \n"
                           "3 : [h,u] \n"
                           "4:  [h,u,flux]", metavar="LOSSES")
    parser.add_option("-p", "--processingmode", dest="processingmode", default=1,
                      help="0: CPU\n1:GPU", metavar="LOAD")
    parser.add_option("-t", "--train", dest="train", default=0,
                      help="train the models", metavar="TRAIN")
    parser.add_option("-w", "--width", dest='nWidth', default=10, \
                      help="integer for input shape of dense hidden layers; default 10", \
                      metavar="WIDTH")
    parser.add_option("-x", "--length", dest='nLength', default=0, \
                      help="integer for number of dense hidden layers; default 0", \
                      metavar="LENGTH")

    """
    Add options to parser for 
    other loss combinations: this will correspond to bools 
    """

    """
    Parser add options, let's call it '-c', for 
    choice of epochs, callbacks, and learning-drop rate.
    """

    (options, args) = parser.parse_args()
    options.architecture = int(options.architecture)
    options.losses = int(options.losses)
    options.train = bool(int(options.train))
    options.degreeBasis = int(options.degreeBasis)
    options.evaluation = bool(int(options.evaluation))
    options.load = bool(int(options.load))
    options.processingmode = int(options.processingmode)

    # Will added these options; if they don't work, he will fix
    options.nWidth = int(options.nWidth)
    options.nLength = int(options.nLength)
    options.curr = int(options.curr)

    # witch to CPU mode, if wished
    if options.processingmode == 0:
        # Set CPU as available physical device
        # Set CPU as available physical device
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if tf.test.gpu_device_name():
            print('GPU found. Using GPU')
        else:
            print("Disabled GPU. Using CPU")

    print("Getting train and test data")
    # Creating settings to run
    nq = 40

    """
    Create method to save data, or, simply 
    training data parameters, according to 
    the dataset, inside MN_Data
    or in the main file here
    """
    epsilon = 0.001
    sampleSize = int(1e+04)
    alphaMax = 70

    Q = utils.getquad('lgwt', nq, -1, 1, options.degreeBasis)  # Create Quadrature Object
    DataClass = utils.MN_Data(options.degreeBasis, Q, 'M_N')  # Create Datacreator

    """
    Will: Let's change this to a save-load data structure
    """
    # Compute distance to boundary in N =  1 case from analytic map

    [u_train, alpha_train, h_train] = DataClass.make_train_data_wrapper(epsilon, alphaMax, sampleSize)

    hess_train = np.zeros((u_train.shape[0],), dtype=float)

    print("---- Set the networks - depending on the input flags ----")

    ### Choose Network size (as discussed, 1000, 2000, 5000 params) ==> Translate to size bracket (1 = 1000,2 = 2000,3 = 5000)
    # Depending on the size bracket, each network needs to adapt its width and depth (to get the corr. number of trainable parameters)
    # trainableParamBracket = int(options.bracket)
    losses = int(options.losses)  # [mse(h), mse(alpha), mse(u), mse(flux)]
    # inputDim = int(options.degreeBasis) + 1  # CAREFULL HERE; we adjusted for new net input sizes
    inputDim = int(options.degreeBasis)

    modelList = []  # list of models

    for width_idx in [15]:
        for depth_idx in [2, 5, 10]:
            if options.architecture == 0:
                modelList.append(
                    ModelFrame(architecture=options.architecture, shapeTuple=(width_idx, depth_idx), lossChoices=losses,
                               inputDim=inputDim))
            elif options.architecture == 1:
                modelList.append(
                    ModelFrame(architecture=options.architecture, shapeTuple=(width_idx, depth_idx), lossChoices=losses,
                               inputDim=inputDim, quad=Q))

            else:
                modelList.append(
                    ModelFrame(architecture=0, shapeTuple=(width_idx, depth_idx), lossChoices=losses,
                               inputDim=inputDim))
                modelList.append(
                    ModelFrame(architecture=1, shapeTuple=(width_idx, depth_idx), lossChoices=losses,
                               inputDim=inputDim, quad=Q))

    print("---- Load the model weights, if flag is set ----")
    if (options.load):
        for model in modelList:
            model.loadWeights()
        print("Loaded weights")
    else:
        print("Skipped weight loading")

    print("---- Model training ----")
    if (options.train):
        # Train all models in the list
        for model in modelList:
            model.trainingProcedure([u_train[:, 1], alpha_train[:, 1], h_train, hess_train], options.curr)
        print("Training successfull")
    else:
        print("Training skipped")

    print("---- Evaluation and error Analysis ----")
    if (options.evaluation == True):
        # testData = utils.getTestData()  # @Will: Your Script goes here
        # This should be changed to 'loading' the training data from file

        trainData_pass = DataClass.make_train_data_wrapper(epsilon, alphaMax, sampleSize)
        testData_pass = DataClass.make_test_data_wrapper('uniform', [1e-8, 8, int(100)], [-70, 70, int(1e+03)])
        print('Done passing Train and Test Data')
        for model in modelList:
            model.errorAnalysis(trainData_pass, testData_pass, '10',
                                '10')  # @Will: Your model error analysis Function goes here
        print("Evalution successfull")
    else:
        print("Evaluation skipped")

    return 0


if __name__ == '__main__':

    tex_dataframe = True

    new_dataframe = False

    run_main = True

    if new_dataframe == True:
        datID = '10'

        netNames = ['L1_S15x2', 'L1_S15x5', 'L1_S15x10']
        # 'L1_S20x2','L1_S20x5','L1_S20x10',\
        # 'L1_S30x2','L1_S30x5','L1_S30x10']

        deg = 1
        AT = AnalysisTools('M_N')
        for domain in ['test', 'train']:
            for method in ['ecnn', 'icnn']:
                AT.newDF(N=deg, domain=domain, datID=datID, method=method, saveNames=netNames)

    if tex_dataframe:
        netNames = ['L1_S15x2', 'L1_S15x5', 'L1_S15x10']
        # 'L1_S20x2','L1_S20x5','L1_S20x10',\
        # 'L1_S30x2','L1_S30x5','L1_S30x10']

        for method in ['ecnn', 'icnn']:

            train_path = 'analysis/raw/results_' + method + '_train_' + '10' + '.pickle'
            test_path = 'analysis/raw/results_' + method + '_test_' + '10' + '.pickle'

            with open(train_path, 'rb') as handle:
                df_train = pickle.load(handle)
            with open(test_path, 'rb') as handle:
                df_test = pickle.load(handle)

            for name in netNames:
                df_train[name] = pd.to_numeric(df_train[name], downcast='float')
                df_test[name] = pd.to_numeric(df_test[name], downcast='float')

            df_test = df_test.transpose()
            df_train = df_train.transpose()
            print('\n Test Frame for ' + method + ':', df_test.to_latex(), '\n')
            print('\n Train Frame for ' + method + ':', df_train.to_latex(), '\n')

    if run_main:
        main()
