# encoding: utf-8
#
# @author Sergii Nechuiviter (@sklavit) snechuiviter@gmail.com
# @date 2017-04-12
#
# Baseon original code at https://github.com/jostmey/rwa
# And original article  https://arxiv.org/pdf/1703.01253.pdf
#

import numpy as np

from keras.engine import InputSpec

from keras.layers.recurrent import _time_distributed_dense

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.legacy import interfaces

from keras.layers import Recurrent


class SimpleRWA(Recurrent):
    """RWA

    For a step-by-step description of the algorithm, see
    [this tutorial](http://dlink here).

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs.
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state.
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean.
            If True, add 1 to the bias of the forget gate at initialization.
            Setting it to true will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the inputs.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.

    # References
        - [name here](http://link here) (remark here...)

    """
    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='glorot_uniform',  #'orthogonal',
                 bias_initializer='zeros',
                 # unit_forget_bias=True,  # TODO not relevant here ?!
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(SimpleRWA, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))  # type: float
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))

    def build(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        self.input_dim = input_shape[2]
        self.input_spec = InputSpec(shape=(batch_size, None, self.input_dim))
        self.state_spec = [InputSpec(shape=(batch_size, self.units)),
                           InputSpec(shape=(batch_size, self.units)),
                           InputSpec(shape=(batch_size, self.units)),
                           InputSpec(shape=(batch_size, self.units))]

        self.states = [None, None, None, None]  # We have 4 state variables for RWA

        if self.stateful:
            self.reset_states()

        self.kernel = self.add_weight((self.input_dim, self.units * 3),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            (self.units, self.units * 2),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            self.bias = self.add_weight((self.units * 2,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            # if self.unit_forget_bias:
            #     bias_value = np.zeros((self.units * 4,))
            #     bias_value[self.units: self.units * 2] = 1.
            #     K.set_value(self.bias, bias_value)
        else:
            self.bias = None

        self.kernel_u = self.kernel[:, :self.units]
        self.kernel_g = self.kernel[:, self.units: self.units * 2]
        self.kernel_a = self.kernel[:, self.units * 2:]

        self.recurrent_kernel_g = self.recurrent_kernel[:, : self.units]
        self.recurrent_kernel_a = self.recurrent_kernel[:, self.units:]

        if self.use_bias:
            self.bias_u = self.bias[:self.units]
            self.bias_g = self.bias[self.units: ]
        else:
            self.bias_u = None
            self.bias_g = None
        self.built = True

    def preprocess_input(self, inputs, training=None):
        if self.implementation == 0:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            x_u = _time_distributed_dense(inputs, self.kernel_u, self.bias_u,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_g = _time_distributed_dense(inputs, self.kernel_g, self.bias_g,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_a = _time_distributed_dense(inputs, self.kernel_a, None,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            return K.concatenate([x_u, x_g, x_a], axis=2)
        else:
            return inputs

    def get_constants(self, inputs, training=None):
        constants = []
        if self.implementation == 0 and 0. < self.dropout < 1.:
            input_shape = K.int_shape(inputs)
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, int(input_dim)))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(3)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(3)])

        if 0. < self.recurrent_dropout < 1.:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)
            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(2)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(2)])
        return constants

    def get_initial_states(self, inputs):
        # # TODO check does this broke state initialization ?
        # batch_size = input_shape[0]
        # self.states = [K.zeros((batch_size, self.units)) for _ in self.states]
        #
        # init_max_a = np.full((batch_size, self.units), -1E38)
        # K.set_value(self.states[0], init_max_a)

        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.units])  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]

        initial_states[0] = K.ones_like(initial_state) * -1E38

        return initial_states

    def step(self, inputs, states):
        max_a_tm1 = states[0]
        n_tm1 = states[1]
        d_tm1 = states[2]
        h_tm1 = states[3]

        dp_mask = states[4]
        rec_dp_mask = states[5]

        if self.implementation == 2:
            raise NotImplementedError()
            # z = K.dot(inputs * dp_mask[0], self.kernel)
            # z += K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel)
            # if self.use_bias:
            #     z = K.bias_add(z, self.bias)
            #
            # z0 = z[:, :self.units]
            # z1 = z[:, self.units: 2 * self.units]
            # z2 = z[:, 2 * self.units: 3 * self.units]
            # z3 = z[:, 3 * self.units:]
            #
            # i = self.recurrent_activation(z0)
            # f = self.recurrent_activation(z1)
            # c = f * c_tm1 + i * self.activation(z2)
            # o = self.recurrent_activation(z3)
        else:
            if self.implementation == 0:
                x_u = inputs[:, :self.units]
                x_g = inputs[:, self.units: 2 * self.units]
                x_a = inputs[:, 2 * self.units:]
            elif self.implementation == 1:
                x_u = K.dot(inputs * dp_mask[0], self.kernel_u) + self.bias_u
                x_g = K.dot(inputs * dp_mask[1], self.kernel_g) + self.bias_g
                x_a = K.dot(inputs * dp_mask[2], self.kernel_a)
            else:
                raise ValueError('Unknown `implementation` mode.')

            g = self.recurrent_activation(x_g + K.dot(h_tm1 * rec_dp_mask[0],
                                                      self.recurrent_kernel_g))

            a = x_a + K.dot(h_tm1 * rec_dp_mask[1], self.recurrent_kernel_a)
            z = x_u * g

            new_max_a = K.maximum(max_a_tm1, a)
            exp_diff = K.exp(max_a_tm1 - new_max_a)
            exp_scaled = K.exp(a - new_max_a)

            n = n_tm1 * exp_diff + z * exp_scaled  # Numerically stable update of numerator
            d = d_tm1 * exp_diff + exp_scaled  # Numerically stable update of denominator
            new_h = self.activation(n / d)
            max_a = new_max_a

            # Use new hidden state only if the sequence length has not been exceeded
            # h = K.switch(K.greater(l, i), h_new, h)
            # TODO how to implement this? Is it needed?
            h = new_h

        if 0. < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True

        return h, [max_a, n, d, h]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  # 'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(SimpleRWA, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
