"""
File: Icnn network
Author: William Porteous and Steffen Schotthöfer
Date: 9.04.2021
"""

import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras

def createIcnnClosure(inputDim,trainableParamBracket,model_losses):

    """
    :param trainableParamBracket: bracket to determine the number of trainable parameters
    :param model_losses: a set of losses for the model
    :return:  the compiled model
    """

    #translate parameter brackets into model width and depth
    if(trainableParamBracket == 0):
        modelWidth = 10
        modelDepth = 6
    else:
        #TODO
        modelWidth = 10
        modelDepth = 6

    #translate set of model losses into a model readble format
    # TODO
    losses = []
    model = createModel(inputDim,modelWidth, modelDepth, losses)
    return model


def createModel(inputDim,modelWidth, modelDepth, losses):

    layerDim = modelWidth

    # Weight initializer #TODO: unify initalizer with Will!
    initializerNonNeg = tf.keras.initializers.RandomUniform(minval=0, maxval=0.5, seed=None)
    initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)

    def convexLayer(layerInput_z: Tensor, netInput_x: Tensor, layerIdx=0) -> Tensor:
        # Weighted sum of previous layers output plus bias
        weightedNonNegSum_z = keras.layers.Dense(layerDim, kernel_constraint=keras.constraints.NonNeg(), activation=None,
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

    def convexLayerOutput(layerInput_z: Tensor, netInput_x: Tensor) -> Tensor:
        # Weighted sum of previous layers output plus bias
        weightedNonNegSum_z = keras.layers.layers.Dense(1, kernel_constraint=keras.constraints.NonNeg(), activation=None,
                                           kernel_initializer=initializerNonNeg,
                                           use_bias=True,
                                           bias_initializer='zeros'
                                           # name='in_z_NN_Dense'
                                           )(layerInput_z)
        # Weighted sum of network input
        weightedSum_x = keras.layers.layers.Dense(1, activation=None,
                                     kernel_initializer=initializer,
                                     use_bias=False
                                     # name='in_x_Dense'
                                     )(netInput_x)
        # Wz+Wx+b
        intermediateSum = keras.layers.layers.Add()([weightedSum_x, weightedNonNegSum_z])

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
    model.build(input_shape=(self.inputDim,))
    model.compile(
        loss={'output_1': tf.keras.losses.MeanSquaredError(), 'output_2': tf.keras.losses.MeanSquaredError()},
        loss_weights={'output_1': 1, 'output_2': 1},
        optimizer='adam',
        metrics=['mean_absolute_error'])

    # model.summary()
    # tf.keras.utils.plot_model(model, to_file=self.filename + '/modelOverview', show_shapes=True,
    # show_layer_names = True, rankdir = 'TB', expand_nested = True)

    return model

class sobolevModel(tf.keras.Model):
    # Sobolev implies, that the model outputs also its derivative
    def __init__(self, coreModel, **opts):
        # tf.keras.backend.set_floatx('float64')  # Full precision training
        super(sobolevModel, self).__init__()

        # Member is only the model we want to wrap with sobolev execution
        self.coreModel = coreModel  # must be a compiled tensorflow model

    def call(self, x, training=False):
        """
        Defines the sobolev execution
        h  is y
        alpha is derivative
        """

        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x)
            y = self.coreModel(x)
        derivative = grad_tape.gradient(y, x)

        return [y, derivative]
