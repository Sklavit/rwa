# encoding = utf-8
# :Date: 2017-04-19
# :Author: Sergii Nechuiviter (@sklavit) (snechuiviter@gmail.com)
# :License: MIT (TODO: add link here...)
from pprint import pprint

import numpy as np
from tqdm import tqdm
from keras import optimizers
from keras.layers import LSTM, Dense
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
               return_sequences=False))
model.add(Dense(num_classes))

model.summary()

model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(lr=learning_rate))

##########################################################################################
# Train
##########################################################################################

# history = model.fit(train.xs, train.ys,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=2,
#                     validation_data=(test.xs, test.ys))

score = model.evaluate(test.xs, np.array([label[-1] for label in test.ys]))
print('Test score: {}'.format(score))

for i in range(10):
    for seq, label in tqdm(zip(train.xs, train.ys), total=dataset.num_train):
       model.train_on_batch(np.array([seq]), np.array([label[-1]]))

    score = model.evaluate(test.xs, np.array([label[-1] for label in test.ys]))
    print('Test score: {}'.format(score))

