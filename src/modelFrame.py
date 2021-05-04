'''
File: Model framework
Author: William Porteous and Steffen Schotthöfer
Date: 9.04.2021
'''

### imports
import tensorflow  as tf
from os import path, makedirs
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
### imports of own scripts
from src.IcnnClosure import createIcnnClosure
from src.EcnnClosure import createEcnnClosure, HaltWhen

import numpy as np
from src.utils import dualityTools

# DataFrame manipulattion stuff
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from tabulate import tabulate

"""
This callback needs to be moved but is useful for hitting a certain threshold 
"""


class ModelFrame:

    def __init__(self, architecture=0, shapeTuple=(20, 5), lossChoices=1, inputDim=1, quad=None):
        """constructor"""

        """
        Create quadrature object and pass to createxxxClosure
        """

        self.nWidth = shapeTuple[0]
        self.nLength = shapeTuple[1]

        self.architecture = architecture

        self.lossChoices = lossChoices

        self.N = inputDim

        self.DT = dualityTools('M_N', 1, None)

        self.saveFolder = "models/losscombi_" + str(lossChoices) + "/width" + str(shapeTuple[0]) + "_depth" + str(
            shapeTuple[1])

        if (architecture == 0):  # Steffen (synonymous with modelchoice) ==> ICNN

            self.saveFolder = self.saveFolder + "/icnn"
            self.model = createIcnnClosure(inputDim=inputDim, shapeTuple=(self.nWidth, self.nLength),
                                           lossChoices=lossChoices, Quad=quad)

            # top line weights, lower line biases
            self.nParams = self.nWidth + (self.nWidth ** 2) * (self.nLength) + (self.nWidth * self.nLength) + 2 + \
                           self.nWidth + (self.nWidth + self.nLength) + 1

        elif (architecture == 1):  # Will: (model choice is ECNN)

            self.saveFolder = self.saveFolder + "/ecnn"
            self.model = createEcnnClosure(inputDim=inputDim, shapeTuple=(self.nWidth, self.nLength),
                                           lossChoices=lossChoices, Quad=quad)

            self.nParams = (self.nLength + 3) * (self.nWidth) + self.nLength * (self.nWidth ** 2) + 1

            # Alternate behavior:
        # return uncompiled model and compile here (i.e. assign the losses), possibly changing
        # ICNN to accomodate hessian loss (but with weight zero)

        else:

            raise ValueError('architecture must be zero or 1')

    def showModel(self):
        self.model.summary()
        return 0

    def loadWeights(self):
        usedFileName = self.saveFolder + '/best_model.h5'
        self.model.load_weights(usedFileName)
        print("Model loaded from file ")
        return 0

    def trainingProcedure(self, trainData, curr):

        u_train, alpha_train, h_train, hess_train = trainData

        ### TODO
        #   @WILL
        if curr == 0:

            # We only use this at the moment
            num_epochs = int(1.5 * (1e+04))
            initial_lr = float(1e-3)
            drop_rate = (num_epochs / 3)
            stop_tol = 1e-7
            mt_patience = int(num_epochs / 10)
            min_delta = stop_tol / 10
            batch_size = int(100)

        elif curr == 1:

            num_epochs = int(1.5 * (1e+04))
            initial_lr = float(1e-3)
            drop_rate = (num_epochs / 3)
            stop_tol = 1e-7
            mt_patience = int(num_epochs / 10)
            min_delta = stop_tol / 10
            batch_size = int(100)

        elif curr == 2:

            num_epochs = int(1.5 * (1e+04))
            initial_lr = float(1e-3)
            drop_rate = (num_epochs / 3)
            stop_tol = 1e-7
            mt_patience = int(num_epochs / 10)
            min_delta = stop_tol / 10
            batch_size = int(100)

        def step_decay(epoch):

            step_size = initial_lr * np.power(10, (-epoch / drop_rate))

            return step_size

            # this callback good to go

        LR = LearningRateScheduler(step_decay)

        MC = ModelCheckpoint(self.saveFolder + '/best_model.h5', \
                             monitor='val_output_' + str(self.lossChoices) + '_loss', \
                             save_best_only=True, \
                             save_weights_only=False, \
                             mode='min', verbose=1)

        HW = HaltWhen('val_output_' + str(self.lossChoices) + '_loss', stop_tol)

        ES = EarlyStopping(monitor='val_output_' + str(self.lossChoices) + '_loss', mode='min', \
                           verbose=1, patience=mt_patience, \
                           min_delta=1e-8)

        CSV_Logger = self.createCSVLoggerCallback()

        callback_list = [MC, ES, HW, LR, CSV_Logger]

        """
        Model bool / string to switch training for Steffen or Will
        because Will needs 1 more argument - the hessian target values 
         """
        if self.model.arch == 'icnn':

            moment_history = self.model.fit(x=u_train, y=[h_train, alpha_train, u_train],
                                            validation_split=0.20,
                                            epochs=num_epochs, batch_size=batch_size, callbacks=callback_list)
        elif self.model.arch == 'ecnn':

            moment_history = self.model.fit(x=u_train, y=[h_train, alpha_train, u_train, hess_train],
                                            validation_split=0.20,
                                            epochs=num_epochs, batch_size=batch_size, callbacks=callback_list)

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

    def errorAnalysis(self, trainData, testData=None, datID_train=None, datID_test=None):
        # this is written for N = 1 only right now
        if self.N == 1:

            history_path = 'models/losscombi_1/' + 'width' + str(self.nWidth) + '_depth' + str(self.nLength) + '/' + \
                           str(self.model.arch) + '/' + 'historyLogs/' + 'history_001_.csv'
            history_frame = pd.read_csv(history_path)
            fig, ax = plt.subplots(1)
            # ax.plot(history_frame['epoch'],history_frame['output_loss_1'],label = 'Entropy')
            for i in range(3):
                ax.plot(history_frame['epoch'], history_frame['output_' + str(i + 1) + '_loss'],
                        label='Loss ' + str(i + 1))
            ax.set_yscale('log')
            ax.set_xlabel('Num Epochs')
            ax.set_ylabel('Errors')
            ax.set_title('Loss Curves for Model ' + self.model.arch + '-' + str(self.nWidth) + 'by' + str(self.nLength))
            ax.legend(loc='upper left', bbox_to_anchor=(1.00, 0.75))
            fig.savefig('analysis/figures/' + self.model.arch + str(self.nWidth) + 'by' + str(self.nLength) + '.eps', \
                        bbox_inches='tight')
            """
            ax.title('History')
            #ax.set_xscale('log')
            ax.set_yscale('log')
            ax.plot(history_frame[]
            print(history_frame)
            """

            train_df_path = 'analysis/raw/results_' + self.model.arch + '_train_' + datID_train + '.pickle'
            test_df_path = 'analysis/raw/results_' + self.model.arch + '_test_' + datID_test + '.pickle'

            # this should be adapted soon to include the value of N; for now this works

            SaveID = 'L' + str(self.lossChoices) + '_S' + str(self.nWidth) + 'x' + str(self.nLength)
            print('Here is network ID: ', SaveID)
            # this is a variable to be cosnstructed from the loss combination and shape.
            # Should be a unique network identifier which we have already loaded into the dataframe

            """
            Get the train data
            """
            u_train_true, alpha_train_true, h_train_true = trainData
            # hess_train_true = np.zeros((u_train_true.shape[0],))
            u1_train_true = u_train_true[:, 1][:, np.newaxis]

            """
            Get the test data
            """
            # Get the testing data
            u_test_true, alpha_test_true, h_test_true = testData
            u_test_true_proj = np.divide(u_test_true[:, 1], u_test_true[:, 0])[:, None]

            # The signatures of our functions are just slightly different so,
            # we use switching here

            if self.model.arch == 'ecnn':

                """
                Calculate Training Predictions
                """
                h_train_pred, alpha_train_pred, u_train_pred, hess_train_pred = self.model.predict(u1_train_true)

                hess_train_pred = hess_train_pred[:, 0]
                num_nonconv_train = np.sum((hess_train_pred < 0))

                """
                Calculate test predictions
                """
                h_test_pred_proj, alpha_test_pred_proj, u_toss, conv_pred_test_proj = self.model.predict(
                    u_test_true_proj)
                h_test_pred_proj = h_test_pred_proj.reshape((h_test_pred_proj.shape[0],))

                h_test_pred = u_test_true[:, 0] * (h_test_pred_proj + np.log(u_test_true[:, 0]))
                alpha_test_pred = alpha_test_pred_proj[:, :]
                alpha_test_pred[:, 0] = alpha_test_pred[:, 0] + np.log(u_test_true[:, 0])

                u_test_pred = self.model.moment_func(alpha_test_pred).numpy()

                num_nonconv_test = np.sum((conv_pred_test_proj[:, 0] < 0))

                """
                u_true_test_proj  = np.divide(u_true_test_2d[:,1],u_true_test_2d[:,0])[:,None]
                h_pred_test_proj,alpha_pred_test_proj,u_toss,conv_pred_test_proj = model.predict(u_true_test_proj)
                
                h_pred_test_proj = h_pred_test_proj.reshape((h_pred_test_proj.shape[0],))
                h_pred_test_2d = u_true_test_2d[:,0]*(h_pred_test_proj + np.log(u_true_test_2d[:,0]))
                
                alpha_pred_test_2d = alpha_pred_test_proj[:,:]
                alpha_pred_test_2d[:,0] = alpha_pred_test_2d[:,0] + np.log(u_true_test_2d[:,0])
                
                u_pred_test_2d = model.moment_func(alpha_pred_test_2d).numpy()
                
                true_vals = [h_true_test_2d,alpha_true_test_2d,u_true_test_2d]
                pred_vals = [h_pred_test_2d,alpha_pred_test_2d,u_pred_test_2d,conv_pred_test_proj]
                """

            elif self.model.arch == 'icnn':

                """
                Calculate Training Predictions
                """
                h_train_pred, alpha_train_pred, toss_ = self.model.predict(u1_train_true)

                # May need to reshape and stack first argument
                alpha0_train_pred = self.DT.alpha0surface(alpha_train_pred[:, 0])
                alpha_train_pred = np.hstack([alpha0_train_pred[:, np.newaxis], alpha_train_pred])
                u_train_pred = self.DT.moment_vector(alpha_train_pred)
                num_nonconv_train = 0

                """
                Calculate test predictions
                """

                h_test_pred_proj, alpha_test_pred_proj, toss_ = self.model.predict(u_test_true_proj)
                h_test_pred_proj = h_test_pred_proj.reshape((h_test_pred_proj.shape[0],))

                h_test_pred = u_test_true[:, 0] * (h_test_pred_proj + np.log(u_test_true[:, 0]))
                alpha0_test_pred_proj = self.DT.alpha0surface(alpha_test_pred_proj[:, 0])
                alpha_test_pred_proj = np.hstack([alpha0_test_pred_proj[:, np.newaxis], alpha_test_pred_proj])
                alpha_test_pred = alpha_test_pred_proj[:, :]
                alpha_test_pred[:, 0] = alpha_test_pred[:, 0] + np.log(u_test_true[:, 0])
                u_test_pred = self.DT.moment_vector(alpha_test_pred)
                num_nonconv_test = 0

            """
            Calculate training errors (This can be wrapped but we are doing manually here)
            """

            h_train_pred = h_train_pred.reshape((h_train_pred.shape[0], 1))
            h_train_true = h_train_true.reshape((h_train_true.shape[0], 1))

            train_L2_h = np.mean(np.square(h_train_pred - h_train_true), axis=0)[0]
            train_L2_hrel = np.sum(np.square(h_train_pred - h_train_true), axis=0)[0] / \
                            np.sum(np.square(h_train_true), axis=0)[0]

            train_L2_u = np.mean(np.sum(np.square(u_train_pred - u_train_true), axis=1))
            train_L2norm_u = np.mean(np.sum(np.square(u_train_true), axis=1))
            train_L2_urel = train_L2_u / train_L2norm_u

            train_L2_u0 = np.mean(np.square(u_train_pred[:, 0] - u_train_true[:, 0]))
            train_u0_norm = np.mean(np.square(u_train_true[:, 0]))
            train_L2_u0rel = train_L2_u0 / train_u0_norm

            train_L2norm_alpha = np.mean(np.sum(np.square(alpha_train_true), axis=1))
            train_L2_alpha = np.mean(np.sum(np.square(alpha_train_pred - alpha_train_true), axis=1))
            train_L2_alpharel = train_L2_alpha / train_L2norm_alpha

            train_L2_u0_spec = np.mean(np.square(u_train_true[:, 0] - u_train_pred[:, 0])) / np.mean(
                np.square(u_train_true[:, 0]))
            train_L2_u1_spec = np.mean(np.square(u_train_true[:, 1] - u_train_pred[:, 1])) / np.mean(
                np.square(u_train_true[:, 1]))

            train_RMSE_vals = [self.nParams, np.sqrt(train_L2_hrel), np.sqrt(train_L2_urel), np.sqrt(train_L2_u0_spec), \
                               np.sqrt(train_L2_u1_spec), np.sqrt(train_L2_alpharel), num_nonconv_train]

            with open(train_df_path, 'rb') as handle:
                df_train = pickle.load(handle)

            df_train[SaveID] = train_RMSE_vals

            print('\n Here is ' + self.model.arch + ' df_train: \n',
                  tabulate(df_train, tablefmt='psql', headers='keys'))

            # Save these changes to the dataframe
            with open(train_df_path, 'wb') as handle:
                pickle.dump(df_train, handle)

            """
            Calculate Test Errors
            """

            h_test_pred = h_train_pred.reshape((h_train_pred.shape[0], 1))
            h_test_true = h_train_true.reshape((h_train_true.shape[0], 1))

            test_L2_h = np.mean(np.square(h_test_pred - h_test_true), axis=0)[0]
            test_L2_hrel = np.sum(np.square(h_test_pred - h_test_true), axis=0)[0] / \
                           np.sum(np.square(h_test_true), axis=0)[0]

            test_L2_u = np.mean(np.sum(np.square(u_test_pred - u_test_true), axis=1))
            test_L2norm_u = np.mean(np.sum(np.square(u_test_true), axis=1))
            test_L2_urel = test_L2_u / test_L2norm_u

            test_L2_u0 = np.mean(np.square(u_test_pred[:, 0] - u_test_true[:, 0]))
            test_u0_norm = np.mean(np.square(u_test_true[:, 0]))
            test_L2_u0rel = test_L2_u0 / test_u0_norm

            test_L2norm_alpha = np.mean(np.sum(np.square(alpha_test_true), axis=1))
            test_L2_alpha = np.mean(np.sum(np.square(alpha_test_pred - alpha_test_true), axis=1))
            test_L2_alpharel = test_L2_alpha / test_L2norm_alpha

            test_L2_u0_spec = np.mean(np.square(u_test_true[:, 0] - u_test_pred[:, 0])) / np.mean(
                np.square(u_test_true[:, 0]))
            test_L2_u1_spec = np.mean(np.square(u_test_true[:, 1] - u_test_pred[:, 1])) / np.mean(
                np.square(u_test_true[:, 1]))

            test_RMSE_vals = [self.nParams, np.sqrt(test_L2_hrel), np.sqrt(test_L2_urel), np.sqrt(test_L2_u0_spec), \
                              np.sqrt(test_L2_u1_spec), np.sqrt(test_L2_alpharel), num_nonconv_test]

            with open(test_df_path, 'rb') as handle:
                df_test = pickle.load(handle)

            df_test[SaveID] = test_RMSE_vals

            print('\n Here is ' + self.model.arch + ' df_test: \n', tabulate(df_test, tablefmt='psql', headers='keys'))

            with open(test_df_path, 'wb') as handle:

                pickle.dump(df_test, handle)



        else:
            print('errorAnalysis only for N == 1 right now')
            pass
        # TODO
        # @WIll
        return 0


if __name__ == "__main__":
    pass
