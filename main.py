'''
File: Main Script for comparison study
Author: William Porteous and Steffen Schotthöfer
Date: 16.03.2021
'''

### imports
from optparse import OptionParser

### imports of own scripts
from src.modelFrame import modelFrame
from src import utils


def main():
    '''
    Main execution Skript to conduct the comparison study between
    Input Convex neural closure by Steffen Schotthöfer (iccnClosure)
    and
    Empirical convex neural closure by Will Porteous  (eCnnClosure)
    '''
    print("---------- Start Network Training Suite ------------")
    print("Parsing options")
    # --- parse options ---
    parser = OptionParser()
    parser.add_option("-a", "--author", dest="author", default="both",
                      help="author of the network", metavar="AUTHOR")
    parser.add_option("-b", "--bracket", dest="bracket", default=1,
                      help="size bracket of network parameters", metavar="BRACKET")
    parser.add_option("-t", "--train", dest="train", default=False,
                      help="train the models", metavar="TRAIN")
    parser.add_option("-e", "--evalutation", dest="evaluation", default=True,
                      help="evaluation and error analysis", metavar="EVALUATE")
    parser.add_option("-l", "--lossCombi", dest="losses", default=0,
                      help="combination of model losses: \n"
                           "0 : [h] \n"
                           "1 :[h,u] \n"
                           "2 :[h,u,flux]", metavar="LOSSES")

    (options, args) = parser.parse_args()

    print("Getting train and test data")

    [u_train,alpha_train,h_train] = utils.getTrainingData() # @Will, insert your function here

    print("---- Set the networks - depending on the input flags ----")

    ### Choose Network size (as discussed, 1000, 2000, 5000 params) ==> Translate to size bracket (1 = 1000,2 = 2000,3 = 5000)
    # Depending on the size bracket, each network needs to adapt its width and depth (to get the corr. number of trainable parameters)
    trainableParamBracket = int(options.bracket)
    losses = int(options.losses) # [mse(h), mse(alpha), mse(u), mse(flux)]


    modelList = [] # list of models
    if(options.author == "steffen" or options.author == "s" or options.author == "Steffen"):
        authorNum = 0
        modelList.append(modelFrame(architecure = 0, trainableParamBracket = trainableParamBracket, model_losses = losses))
    elif( options.author == "will" or options.author == "w" or options.author == "Will"):
        authorNum = 1
        modelList.append(modelFrame(architecure = 1,trainableParamBracket = trainableParamBracket, model_losses = losses))
    else: # default: Choose both
        modelList.append(modelFrame(architecure = 0,trainableParamBracket = trainableParamBracket, model_losses = losses))
        modelList.append(modelFrame(architecure = 1,trainableParamBracket = trainableParamBracket, model_losses = losses))

    print("---- Load the model weights, if flag is set ----")
    if(options.load):
        for model in modelList:
            model.loadWeights()
        print("Loaded weights")
    else:
        print("Skipped weight loading")

    print("---- Model training ----")
    if(options.train):
        # Train all models in the list
        for model in modelList:
            model.trainingProcedure()# @Will: Model Training Function goes here
        print("Training successfull")
    else:
        print("Training skipped")

    print("---- Evaluation and error Analysis ----")
    if(options.evaluation == True):
        testData = utils.getTestData() # @Will: Your Script goes here
        for model in modelList:
            model.errorAnalysis(testData) # @Will: Your model error analysis Function goes here
        print("Evalution successfull")
    else:
        print("Evaluation skipped")

    return 0

if __name__ == '__main__':
    main()
