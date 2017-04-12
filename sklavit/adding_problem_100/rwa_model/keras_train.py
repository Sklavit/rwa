#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2017-01-01 (This is my new year's resolution)
# Purpose: Train recurrent neural network
# License: For legal information see LICENSE in the home directory.
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

from pprint import pprint

import numpy

import dataplumbing as dp
from keras import backend as K
from keras import optimizers
from keras.layers import Dense, LSTM
from keras.models import Sequential


##########################################################################################
# Settings
##########################################################################################

# Model settings
#
from simple_keras_rwa import SimpleRWA

num_features = dp.train.num_features
max_steps = dp.train.max_length
hidden_units = 250
num_classes = dp.train.num_classes
activation = K.tanh
initialization_factor = 1.0

# Training parameters
#
num_iterations = 50000 / 10.0
batch_size = 100
epochs = int(num_iterations / (dp.train.num_samples / batch_size))
learning_rate = 0.001

##########################################################################################
# Model
##########################################################################################

# Inputs
#

# x_input = Input(shape=(batch_size, max_steps, num_features))
# l_input = Input(shape=(batch_size, ))

model = Sequential()
# model.add(SimpleRWA(hidden_units,
#                     batch_size=batch_size,
#                     input_shape=[max_steps, num_features], unroll=True
#                     ))
model.add(LSTM(hidden_units, activation='linear',
                    batch_size=batch_size,
                    input_shape=[max_steps, num_features], unroll=True))
model.add(Dense(num_classes))  # ly = K.dot(h, W_o) + b_o

model.summary()

# cost = K.var(K.square(ly_flat - y))
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
sgd = optimizers.Adam(lr=learning_rate)  #, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

##########################################################################################
# Train
##########################################################################################

# weights = [('simple_rwa_1/kernel',           numpy.array([[1.0        ,  0.00000000,  0.00000000],
#                                                           [0.000000000,100.00000000, 15.00000000]], dtype=numpy.float32)),
#            ('simple_rwa_1/recurrent_kernel', numpy.array([[ 0.00000000,  0.00000000]], dtype=numpy.float32)),
#            ('simple_rwa_1/bias',             numpy.array( [ 0.00000000,  0.00000000], dtype=numpy.float32)),
#            ('dense_1/kernel',                numpy.array([[ 2.00000000]], dtype=numpy.float32)),
#            ('dense_1/bias',                   numpy.array([ 0.00000000], dtype=numpy.float32))]

# model.set_weights([w[1] for w in weights])

history = model.fit(dp.train.xs, dp.train.ys,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(dp.test.xs, dp.test.ys))

predict = model.predict(dp.test.xs[0, :, :].reshape(1, -1, 2))

print("X")
print(dp.test.xs[0, :, :].reshape(1, -1, 2))
print("Predict")
pprint(predict)

print("Weights")
pprint(list(zip(model.weights, model.get_weights())))

score = model.evaluate(dp.test.xs, dp.test.ys, verbose=1)
print('Test score:')
pprint(score)
