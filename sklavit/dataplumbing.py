#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2017-01-01
# Purpose: Load dataset and create interfaces for piping the data to the model
# License: For legal information see LICENSE in the home directory.
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import numpy as np


##########################################################################################
# Class definitions
##########################################################################################


# Defines interface between the data and model
#
class Dataset:
    def __init__(self, xs, ls, ys, num_samples, num_features, max_length, num_classes):
        self.xs = xs  # Store the features
        self.ls = ls  # Store the length of each sequence
        self.ys = ys  # Store the labels
        self.num_samples = num_samples
        self.num_features = num_features
        self.max_length = max_length
        self.num_classes = num_classes

    def from_xs_ls_ys(self, xs, ls, ys):
        return Dataset(xs, ls, ys, len(ys), len(xs[0, 0, :]), len(xs[0, :, 0]), 1)

    def batch(self, batch_size):
        js = np.random.randint(0, self.num_samples, batch_size)
        return self.xs[js, :, :], self.ls[js], self.ys[js]
