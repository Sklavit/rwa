# encoding = utf-8
# :Date: 2017-04-19
# :Author: Sergii Nechuiviter (@sklavit) (snechuiviter@gmail.com)
# :License: MIT (TODO: add link here...)
from pprint import pprint

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from keras import optimizers
from keras.layers import LSTM, Dense, Reshape, TimeDistributed
from keras.models import Sequential

from dataplumbing import Dataset
from new_seq2seq import dataset

train = Dataset(dataset.xs_train, None, dataset.ys_train, dataset.num_train, dataset.n_features, dataset.max_length, 1)
test = Dataset(dataset.xs_test, None, dataset.ys_test, dataset.num_test, dataset.n_features, dataset.max_length, 1)

# Model settings
num_features = train.num_features
max_steps = train.max_length
hidden_units = 250
num_classes = train.num_classes

# Training parameters
#
num_iterations = 50000 / 10.0
batch_size = 100
epochs = int(num_iterations / (train.num_samples / batch_size))
learning_rate = 0.001

##########################################################################################
# Model
##########################################################################################

model = Sequential()
model.add(LSTM(hidden_units,
               activation='linear',
               # batch_size=batch_size,
               input_shape=[None, num_features],
               unroll=False,
               return_sequences=True))
model.add(Dense(1))
# model.add(Reshape(target_shape=(None, None)))

model.summary()

model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=learning_rate))

##########################################################################################
# Train
##########################################################################################

train_xs = pad_sequences(train.xs, maxlen=None, dtype='float32', padding='pre', truncating='pre', value=0.)
train_ys = pad_sequences(train.ys, maxlen=None, dtype='float32', padding='pre', truncating='pre', value=0.)
test_xs = pad_sequences(test.xs, maxlen=None, dtype='float32', padding='pre', truncating='pre', value=0.)
test_ys = pad_sequences(test.ys, maxlen=None, dtype='float32', padding='pre', truncating='pre', value=0.)

score = model.evaluate(test_xs, test_ys[:, :, np.newaxis])
print('Test score: {}'.format(score))

history = model.fit(train_xs, train_ys[:, :, np.newaxis],
                    # batch_size=batch_size,
                    # epochs=epochs,
                    verbose=1,
                    validation_data=(test_xs, test_ys[:, :, np.newaxis]))

score = model.evaluate(test_xs, test_ys[:, :, np.newaxis])
print('Test score: {}'.format(score))

