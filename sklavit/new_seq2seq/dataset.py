# encoding = utf-8
# :Date: 2017-04-19
# :Author: Sergii Nechuiviter (@sklavit) (snechuiviter@gmail.com)
# :License: MIT (TODO: add link here...)
import os

import numpy as np

num_train = 100000 // 10000
num_test = 10000   // 10000
max_length = 10  # TODO originally: 100
n_features = 2

np.random.seed(42)

##########################################################################################
# Generate/Load dataset
##########################################################################################

# Make directory
#
parent_directory = os.path.dirname(__file__)
os.makedirs(os.path.join(parent_directory, 'bin'), exist_ok=True)


def generate_xs_ys(size, max_length, n_features):
    xs = []
    ys = []
    for i in range(size):
        seq_length = np.random.randint(1, max_length)
        h = np.zeros(n_features, dtype=np.float32)
        x_seq = np.zeros((seq_length, n_features), dtype=np.float32)
        y_seq = np.zeros(seq_length, dtype=np.float32)
        for j in range(seq_length):
            value = np.random.rand(n_features)
            x_seq[j] = value
            h = (h + value) * value
            y_seq[j] = np.sum(h)
        xs.append(x_seq)
        ys.append(y_seq)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

# Generate or load Training data
train_file_path = os.path.join(parent_directory, 'bin', 'train')
if not os.path.isfile(train_file_path + '.npz'):
    xs_train, ys_train = generate_xs_ys(num_train, max_length, n_features)
    np.savez_compressed(train_file_path, xs=xs_train, ys=ys_train)
else:
    npzfile = np.load(train_file_path + '.npz')
    xs_train = npzfile['xs']
    ys_train = npzfile['ys']

# Generate or load Test data
test_file_path = os.path.join(parent_directory, 'bin', 'test')
if not os.path.isfile(test_file_path + '.npz'):
    xs_test, ys_test = generate_xs_ys(num_test, max_length, n_features)
    np.savez_compressed(test_file_path, xs=xs_test, ys=ys_test)
else:
    npzfile = np.load(test_file_path + '.npz')
    xs_test = npzfile['xs']
    ys_test = npzfile['ys']


