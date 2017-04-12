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

import os

import numpy as np
import tensorflow as tf

import dataplumbing as dp

##########################################################################################
# Settings
##########################################################################################

# Model settings
#
num_features = dp.train.num_features
max_steps = dp.train.max_length
num_cells = 250
num_classes = dp.train.num_classes

# Training parameters
#
num_iterations = 50000
batch_size = 100
learning_rate = 0.001

##########################################################################################
# Model
##########################################################################################

# Inputs
#
x = tf.placeholder(tf.float32, [batch_size, max_steps, num_features])	# Features
l = tf.placeholder(tf.float32, [batch_size])	# Sequence length
y = tf.placeholder(tf.float32, [batch_size, max_steps, num_classes])	# Labels

# Trainable parameters
#
W_ig = tf.Variable(
	tf.random_uniform(
		[num_features+num_cells, num_cells],
		minval=-np.sqrt(6.0/(num_features+2.0*num_cells)),
		maxval=np.sqrt(6.0/(num_features+2.0*num_cells))
	)
)
b_ig = tf.Variable(tf.zeros([num_cells]))

W_fg = tf.Variable(
	tf.random_uniform(
		[num_features+num_cells, num_cells],
		minval=-np.sqrt(6.0/(num_features+2.0*num_cells)),
		maxval=np.sqrt(6.0/(num_features+2.0*num_cells))
	)
)
b_fg = tf.Variable(tf.ones([num_cells]))	# Initial bias of 1

W_og = tf.Variable(
	tf.random_uniform(
		[num_features+num_cells, num_cells],
		minval=-np.sqrt(6.0/(num_features+2.0*num_cells)),
		maxval=np.sqrt(6.0/(num_features+2.0*num_cells))
	)
)
b_og = tf.Variable(tf.zeros([num_cells]))

W_c = tf.Variable(
	tf.random_uniform(
		[num_features+num_cells, num_cells],
		minval=-np.sqrt(6.0/(num_features+2.0*num_cells)),
		maxval=np.sqrt(6.0/(num_features+2.0*num_cells))
	)
)
b_c = tf.Variable(tf.zeros([num_cells]))

W_o = tf.Variable(
	tf.random_uniform(
		[num_cells, num_classes],
		minval=-np.sqrt(6.0/(num_cells+num_classes)),
		maxval=np.sqrt(6.0/(num_cells+num_classes))

	)
)
b_o = tf.Variable(tf.zeros([1]))

# Internal states
#
h = tf.zeros([batch_size, num_cells])
c = tf.zeros([batch_size, num_cells])

# Define model
#
error = tf.zeros([batch_size])

for i in range(max_steps):

	x_step = x[:,i,:]
	xh_join = tf.concat(1, [x_step, h])	# Combine the features and hidden state into one tensor

	ig = tf.sigmoid(tf.matmul(xh_join, W_ig)+b_ig)
	fg = tf.sigmoid(tf.matmul(xh_join, W_fg)+b_fg)
	og = tf.sigmoid(tf.matmul(xh_join, W_og)+b_og)
	c_ = tf.tanh(tf.matmul(xh_join, W_c)+b_c)
	c = fg*c+ig*c_
	h = og*tf.tanh(c)

	ly = tf.matmul(h, W_o)+b_o

	error_step = tf.nn.softmax_cross_entropy_with_logits(ly, y[:,i,:])	# Cross-entropy cost function
	error += tf.select(tf.greater(l, i), error_step, tf.zeros([batch_size]))	# Include cost from this step only if the sequence length has not been exceeded

##########################################################################################
# Optimizer
##########################################################################################

# Optimizer
#
cost = tf.reduce_mean(tf.div(error, l))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

##########################################################################################
# Train
##########################################################################################

# Operation to initialize session
#
initializer = tf.global_variables_initializer()

# Open session
#
with tf.Session() as session:

	# Initialize variables
	#
	session.run(initializer)

	# Each training session represents one batch
	#
	for iteration in range(num_iterations):

		# Grab a batch of training data
		#
		xs, ls, ys = dp.train.batch(batch_size)
		feed = {x: xs, l: ls, y: ys}

		# Update parameters
		#
		out = session.run((cost,  optimizer), feed_dict=feed)
		print('Iteration:', iteration, 'Dataset:', 'train', 'Cost:', out[0]/np.log(2.0))

		# Periodically run model on test data
		#
		if iteration%100 == 0:

			# Grab a batch of test data
			#
			xs, ls, ys = dp.test.batch(batch_size)
			feed = {x: xs, l: ls, y: ys}

			# Run model
			#
			out = session.run(cost, feed_dict=feed)
			print('Iteration:', iteration, 'Dataset:', 'test', 'Cost:', out/np.log(2.0))

	# Save the trained model
	#
	os.makedirs('bin', exist_ok=True)
	saver = tf.train.Saver()
	saver.save(session, 'bin/train.ckpt')

