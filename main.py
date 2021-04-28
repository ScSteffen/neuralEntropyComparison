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
import numpy as np
import os
import tensorflow as tf


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
    parser.add_option("-b", "--bracket", dest="bracket", default=1,
                      help="size bracket of network parameters", metavar="BRACKET")
    parser.add_option("-c", "--curriculum", dest="curr", default=0, \
                      help="integer between 0 and 2 for learning curriculum", \
                      metavar="CURRICULUM")
    parser.add_option("-d", "--degreeBasis", dest="degreeBasis", default=1,
                      help="degree of the basis",
                      metavar="DEGREE")
    parser.add_option("-e", "--evalutation", dest="evaluation", default=1,
                      help="evaluation and error analysis", metavar="EVALUATE")
    parser.add_option("-l", "--load", dest="load", default=0,
                      help="0: Dont load weights\n1:Load weights", metavar="LOAD")
    parser.add_option("-o", "--lossCombi", dest="losses", default=0,
                      help="combination of model losses (objective functions): \n"
                           "1 : [h] \n"
                           "2:  [h,alpha] \n"
                           "3 : [h,u] \n"
                           "4:  [h,u,flux]", metavar="LOSSES")
    parser.add_option("-p", "--processingmode", dest="processingmode", default=0,
                      help="0: CPU\n1:GPU", metavar="LOAD")
    parser.add_option("-t", "--train", dest="train", default=0,
                      help="train the models", metavar="TRAIN")
    parser.add_option("-w", "--width", dest='nWidth', default=10, \
                      help="integer for input shape of dense hidden layers; default 10", \
                      metavar="WIDTH")
    parser.add_option("-s", "--length", dest='nLength', default=0, \
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
    options.bracket = int(options.bracket)
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
    trainableParamBracket = int(options.bracket)
    losses = int(options.losses)  # [mse(h), mse(alpha), mse(u), mse(flux)]
    # inputDim = int(options.degreeBasis) + 1  # CAREFULL HERE; we adjusted for new net input sizes
    inputDim = int(options.degreeBasis)

    modelList = []  # list of models

    for width_idx in [15, 20, 30]:
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
        testData = utils.getTestData()  # @Will: Your Script goes here
        for model in modelList:
            model.errorAnalysis(testData)  # @Will: Your model error analysis Function goes here
        print("Evalution successfull")
    else:
        print("Evaluation skipped")

    return 0


if __name__ == '__main__':
    main()
