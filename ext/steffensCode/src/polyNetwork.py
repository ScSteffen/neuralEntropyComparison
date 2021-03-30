def create_model_MK9_poly():  # Build the network:

    # Define the input
    weightIniMean = 0.0
    weightIniStdDev = 0.05
    # Number of basis functions used:

    # Weight initializer
    initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5, seed=None)
    #### input layer ####
    input_ = keras.Input(shape=(1,))
    # Hidden layers
    '''
    hidden = layers.Dense(3, activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='ones')(input_)
    hidden = layers.Dense(3, activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='ones'
                          )(hidden)
    hidden = layers.Dense(3, activation="softplus",
                          kernel_initializer=initializer,
                          bias_initializer='ones'
                          )(hidden)

    # Define the output
    output_ = layers.Dense(1,
                           kernel_initializer=initializer,
                           bias_initializer='ones'
                           )(hidden)
    '''
    hidden = layers.Dense(10, activation="softplus")(input_)
    hidden = layers.Dense(10, activation="softplus")(hidden)
    hidden = layers.Dense(10, activation="softplus")(hidden)
    hidden = layers.Dense(10, activation="softplus")(hidden)
    output_ = layers.Dense(1, activation=None)(hidden)

    # Create the model
    model = keras.Model(inputs=[input_], outputs=[output_], name="FCNN")
    model.summary()

    # model.compile(loss=cLoss_FONC_varD(quadOrder,BasisDegree), optimizer='adam')#, metrics=[custom_loss1dMB, custom_loss1dMBPrime])
    model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mean_absolute_error'])
    return model
