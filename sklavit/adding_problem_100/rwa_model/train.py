#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2017-01-01 (This is my new year's resolution)
# Purpose: Train recurrent neural network
# License: For legal information see LICENSE in the home directory.
##########################################################################################
# CODE HEAVILY MODIFIED BY Sergii Nechuiviter (@sklavit) snechuiviter@gmail.com
# @date 2017-04-13


##########################################################################################
# Libraries
##########################################################################################

from pprint import pprint

from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential

from adding_problem_100 import dataset
from dataplumbing import Dataset
from simple_keras_rwa import SimpleRWA
##########################################################################################
# Settings
##########################################################################################
#

# Create split of data
#
train = Dataset(dataset.xs_train, dataset.ls_train, dataset.ys_train)
test = Dataset(dataset.xs_test, dataset.ls_test, dataset.ys_test)

# Model settings
num_features = train.num_features
max_steps = train.max_length
hidden_units = 1
num_classes = train.num_classes

# Training parameters
#
num_iterations = 50000 / 1.0
batch_size = 100
epochs = int(num_iterations / (train.num_samples / batch_size))
learning_rate = 0.001

##########################################################################################
# Model
##########################################################################################

model = Sequential()
model.add(SimpleRWA(hidden_units,
                    activation='linear',
                    batch_size=batch_size,
                    input_shape=[max_steps, num_features],
                    unroll=True))
model.add(Dense(num_classes))

model.summary()

model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=learning_rate))

##########################################################################################
# Train
##########################################################################################

history = model.fit(train.xs, train.ys,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    validation_data=(test.xs, test.ys))

score = model.evaluate(test.xs, test.ys, batch_size=batch_size)
print('Test score:')
pprint(score)
