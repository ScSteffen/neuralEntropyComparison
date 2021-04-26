"""
File: Icnn network
Author: William Porteous and Steffen Schotthöfer
Date: 9.04.2021
"""

import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras
from src import math


def createIcnnClosure(inputDim, shapeTuple, lossChoices, Quad=None):
    """
    :param shapeTuple: tuple which determines network architecture. 0-th element 
    is number of nodes per dense hidden layer (width); 1st element is number of
    dense hidden layers (depth). 
    
    :param model_losses: a set of losses for the model
    :return:  the compiled model
    """

    modelWidth, modelDepth = shapeTuple
    # translate parameter brackets into model width and depth
    """
    #Commented this out; this parameter was replaced with 'shapeTuple'
    # which receives its default values at the level of modelFrame script instead of within 
    # the createIcnnClosure function. THIS CAN BE DELETED 
    
    if (trainableParamBracket == 0):
        modelWidth = 10
        modelDepth = 6
    else:
        # TODO
        modelWidth = 10
        modelDepth = 6
    """
    # translate set of model losses into a model readble format

    """
        This bool set is determined by model_losses; model_losses is passed as 
        an integer from the 'options' parser in main.py 
        """
    if lossChoices == 0:
        loss_choices = [True, False, False]

    elif lossChoices == 1:

        loss_choices = [True, False, True]

    elif lossChoices == 2:
        # This is identical, as of now, to lossChoices == 1
        loss_choices = [True, False, True]

    loss_weights = [float(x) for x in loss_choices]

    model = createModel(inputDim, modelWidth, modelDepth, loss_weights)
    return model


def createModel(inputDim, modelWidth, modelDepth, loss_weights):
    layerDim = modelWidth

    # Weight initializer #TODO: unify initalizer with Will!
    initializerNonNeg = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)
    initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)

    def convexLayer(layerInput_z: Tensor, netInput_x: Tensor, layerIdx=0) -> Tensor:
        """
        WAP 14/4/21: Is this python? I've never used  : for binding variable type to argument
        or -> assignment opearator. These Cpp shortcuts work?
        """
        # Weighted sum of previous layers output plus bias
        weightedNonNegSum_z = keras.layers.Dense(layerDim, kernel_constraint=keras.constraints.NonNeg(),
                                                 activation=None,
                                                 kernel_initializer=initializerNonNeg,
                                                 use_bias=True, bias_initializer='zeros',
                                                 name='non_neg_component_' + str(layerIdx)
                                                 )(layerInput_z)
        # Weighted sum of network input
        weightedSum_x = keras.layers.Dense(layerDim, activation=None,
                                           kernel_initializer=initializer,
                                           use_bias=False, name='dense_component_' + str(layerIdx)
                                           )(netInput_x)
        # Wz+Wx+b
        intermediateSum = keras.layers.Add(name='add_component_' + str(layerIdx))([weightedSum_x, weightedNonNegSum_z])

        # activation
        out = tf.keras.activations.softplus(intermediateSum)
        # batch normalization
        # out = layers.BatchNormalization(name='bn_' + str(layerIdx))(out)
        return out

    # WAP 14/4/21: Is this python? I've never used  : for binding variable type to argument
    # or -> assignment opearator 
    def convexLayerOutput(layerInput_z: Tensor, netInput_x: Tensor) -> Tensor:
        # Weighted sum of previous layers output plus bias
        weightedNonNegSum_z = keras.layers.Dense(1, kernel_constraint=keras.constraints.NonNeg(), activation=None,
                                                 kernel_initializer=initializerNonNeg,
                                                 use_bias=True,
                                                 bias_initializer='zeros'
                                                 # name='in_z_NN_Dense'
                                                 )(layerInput_z)
        # Weighted sum of network input
        weightedSum_x = keras.layers.Dense(1, activation=None,
                                           kernel_initializer=initializer,
                                           use_bias=False
                                           # name='in_x_Dense'
                                           )(netInput_x)
        # Wz+Wx+b
        intermediateSum = keras.layers.Add()([weightedSum_x, weightedNonNegSum_z])

        # activation
        # out = tf.keras.activations.softplus(intermediateSum)
        # batch normalization
        # out = layers.BatchNormalization()(out)
        return intermediateSum

    ### build the core network with icnn closure architecture ###
    input_ = keras.Input(shape=(inputDim,))
    hidden = keras.layers.Dense(layerDim, activation="softplus", kernel_initializer=initializer,
                                bias_initializer='zeros', name="first_dense"
                                )(input_)
    for idx in range(0, modelDepth):
        hidden = convexLayer(hidden, input_, layerIdx=idx)
    output_ = convexLayerOutput(hidden, input_)  # outputlayer

    ### Create the core model
    coreModel = keras.Model(inputs=[input_], outputs=[output_], name="Icnn_closure")

    ### Build the model wrapper
    model = sobolevModel(coreModel, name="sobolev_icnn_wrapper")
    batchSize = 2  # dummy entry
    model.build(input_shape=(batchSize, inputDim))
    model.compile(loss={'output_1': tf.keras.losses.MeanSquaredError(), 'output_2': tf.keras.losses.MeanSquaredError(),
                        'output_3': tf.keras.losses.MeanSquaredError()},
                  loss_weights=loss_weights, optimizer='adam', metrics=['mean_absolute_error'])

    return model


class sobolevModel(tf.keras.Model):
    # Sobolev implies, that the model outputs also its derivative
    def __init__(self, coreModel, polyDegree=1, **opts):
        super(sobolevModel, self).__init__()

        self.arch = 'icnn'

        # Member is only the model we want to wrap with sobolev execution
        self.coreModel = coreModel  # must be a compiled tensorflow model

        # Create quadrature and momentBasis. Currently only for 1D problems
        self.polyDegree = polyDegree
        self.nq = 100
        [quadPts, quadWeights] = math.qGaussLegendre1D(self.nq)  # dims = nq
        self.quadPts = tf.constant(quadPts, shape=(1, self.nq), dtype=tf.float32)  # dims = (batchSIze x N x nq)
        self.quadWeights = tf.constant(quadWeights, shape=(1, self.nq),
                                       dtype=tf.float32)  # dims = (batchSIze x N x nq)
        mBasis = math.computeMonomialBasis1D(quadPts, self.polyDegree)  # dims = (N x nq)
        self.inputDim = mBasis.shape[0]
        self.momentBasis = tf.constant(mBasis, shape=(self.inputDim, self.nq),
                                       dtype=tf.float32)  # dims = (batchSIze x N x nq)

    def call(self, x, training=False):
        """
        Defines the sobolev execution
        """

        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x)
            h = self.coreModel(x)
        alpha = grad_tape.gradient(h, x)

        # u = self.reconstruct_u(self.reconstruct_alpha(alpha))
        return [h, alpha, alpha]

    def callDerivative(self, x, training=False):
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x)
            y = self.coreModel(x)
        derivativeNet = grad_tape.gradient(y, x)

        return derivativeNet

    def reconstruct_alpha(self, alpha):
        """
        brief:  Reconstructs alpha_0 and then concats alpha_0 to alpha_1,... , from alpha1,...
                Only works for maxwell Boltzmann entropy so far.
        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: alpha, dims = (nS x N-1)
               m    , dims = (N x nq)
               w    , dims = nq
        returns alpha_complete = [alpha_0,alpha], dim = (nS x N), where alpha_0 = - ln(<exp(alpha*m)>)
        """
        tmp = tf.math.exp(tf.tensordot(alpha, self.momentBasis[1:, :], axes=([1], [0])))  # tmp = alpha * m
        alpha_0 = -tf.math.log(tf.tensordot(tmp, self.quadWeights, axes=([1], [1])))  # ln(<tmp>)
        return tf.concat([alpha_0, alpha], axis=1)  # concat [alpha_0,alpha]

    def reconstruct_u(self, alpha):
        """
        brief: reconstructs u from alpha
        nS = batchSize
        N = basisSize
        nq = number of quadPts

        input: alpha, dims = (nS x N)
               m    , dims = (N x nq)
               w    , dims = nq
        returns u = <m*eta_*'(alpha*m)>, dim = (nS x N)
        """
        # Currently only for maxwell Boltzmann entropy
        f_quad = tf.math.exp(tf.tensordot(alpha, self.momentBasis, axes=([1], [0])))  # alpha*m
        tmp = tf.math.multiply(f_quad, self.quadWeights)  # f*w
        return tf.tensordot(tmp, self.momentBasis[:, :], axes=([1], [1]))  # f * w * momentBasis
