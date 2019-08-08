# -*- coding: utf-8 -*-
"""Recurrent layers and their base classes.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import warnings

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints

from keras.engine.base_layer import Layer
from keras.engine.base_layer import InputSpec
from keras.utils.generic_utils import has_arg
from keras.utils.generic_utils import to_list
from keras.backend.recurrent import RNN

# Legacy support.
from keras.legacy.layers import Recurrent
from keras.legacy import interfaces


def compute_attention(h_tm1, pctx_, context, att_dp_mask, attention_recurrent_kernel,
                      attention_context_wa, bias_ca, mask_context, attention_mode='add'):
    """Computes an attended vector over an input sequence of vectors (context).

    The resulting attention vector 'phi' at time 't' is formed by applying a weighted sum over the sequence of inputs 'x_1^I':
            phi(x_1^I, t) = ∑_i alpha_i(t) * x_i,
        where each 'alpha_i' at time 't' is a weighting vector over all the input dimension that accomplishes the following condition:
            ∑_i alpha_i = 1
        and is dynamically adapted at each timestep w.r.t. the following formula:
            alpha_i(t) = exp{e_i(t)} /  ∑_j exp{e_j(t)}
        where each 'e_i' at time 't' is calculated as:
            e_i(t) = score(h_tm1, x_i)

        score is a function that assigns a weight depending on how well h_tm1 and x_i match.
        The following scoring functions are implemented:
            - 'add'/'bahdanau':
               e_i(t) = wa' * tanh( Wa * x_i  +  Ua * h_tm1 +  ba ),
            - 'dot'/'luong':
               e_i(t) = h_tm1' · x_i # Requires the dimensions to be the same
            - 'scale-dot':
               e_i(t) = (h_tm1' · x_i) / \sqrt(|x_i|) # Requires the dimensions to be the same

    # Arguments
        h_tm1: Last decoder state.
        pctx_: Projected context (i.e. context * Ua + ba).
        context: Original context.
        att_dp_mask: Dropout for the attention MLP.
        attention_recurrent_kernel:  attention MLP weights.
        attention_context_wa:  attention MLP weights.
        bias_ca:  attention MLP bias.
        mask_context: mask of the context.
        attention_mode: 'add', 'dot' or function that accepts as arguments: `h_tm1, pctx_, context, att_dp_mask, attention_recurrent_kernel, attention_context_wa, bias_ca, mask_context`
        and should return the scores `e` for the input annotations.

    # Returns
        ctx_: attended representation of the input.
        alphas: weights computed by the attention mechanism.

    # Raises
        NotImplementedError: If the attention_mode specified is not implemented.

    # References
        - [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
        - [Effective Approaches to Attention-based Neural Machine Translation](http://www.aclweb.org/anthology/D15-1166)
    """
    p_state_ = K.dot_product(h_tm1 * att_dp_mask[0], attention_recurrent_kernel)

    pctx_ = K.tanh(pctx_ + p_state_[:, None, :])
    e = K.dot_product(pctx_, attention_context_wa) + bias_ca

    if mask_context is not None and K.ndim(mask_context) > 1:  # Mask the context (only if necessary)
        e = K.cast(mask_context, K.dtype(e)) * e
    alphas = K.softmax(K.reshape(e, [K.shape(e)[0], K.shape(e)[1]]))

    # sum over the in_timesteps dimension resulting in [batch_size, input_dim]
    ctx_ = K.sum(context * alphas[:, :, None], axis=1)

    return ctx_, alphas




class AttLSTMCond(Recurrent):
    """Long-Short Term Memory unit with Attention + the previously generated word fed to the current timestep.

    You should give two inputs to this layer:
        1. The shifted sequence of words (shape: (batch_size, output_timesteps, embedding_size))
        2. The complete input sequence (shape: (batch_size, input_timesteps, input_dim))
    Optionally, you can set the initial hidden state, with a tensor of shape: (batch_size, units)

    # Arguments
        units: Positive integer, dimensionality of the output space.
        att_units:  Positive integer, dimensionality of the attention space.
        return_extra_variables: Return the attended context vectors and the attention weights (alphas)
        return_states: Whether it should return the internal RNN states.
        attention_mode: 'add', 'dot' or custom function.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step
            (see [activations](../activations.md)).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        conditional_initializer: Initializer for the `conditional_kernel`
            weights matrix,
            used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        attention_recurrent_initializer:  Initializer for the `attention_recurrent_kernel`
            weights matrix, used for the linear transformation of the conditional inputs
            (see [initializers](../initializers.md)).
        attention_context_initializer:  Initializer for the `attention_context_kernel`
            weights matrix,
            used for the linear transformation of the attention context inputs
            (see [initializers](../initializers.md)).
        attention_context_wa_initializer:  Initializer for the `attention_wa_kernel`
            weights matrix,
            used for the linear transformation of the attention context
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        bias_ba_initializer: Initializer for the bias_ba vector from the attention mechanism
            (see [initializers](../initializers.md)).
        bias_ca_initializer: Initializer for the bias_ca vector from the attention mechanism
            (see [initializers](../initializers.md)).
        unit_forget_bias: Boolean, whether the forget gate uses a bias vector.
        mask_value: Value of the mask of the context (0. by default)
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        conditional_regularizer: Regularizer function applied to
            the `conditional_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_recurrent_regularizer:  Regularizer function applied to
            the `attention_recurrent__kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_regularizer:  Regularizer function applied to
            the `attention_context_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        attention_context_wa_regularizer:  Regularizer function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        bias_ba_regularizer:  Regularizer function applied to the bias_ba vector
            (see [regularizer](../regularizers.md)).
        bias_ca_regularizer:  Regularizer function applied to the bias_ca vector
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
        conditional_constraint: Constraint function applied to
            the `conditional_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_recurrent_constraint: Constraint function applied to
            the `attention_recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_constraint: Constraint function applied to
            the `attention_context_kernel` weights matrix
            (see [constraints](../constraints.md)).
        attention_context_wa_constraint: Constraint function applied to
            the `attention_context_wa_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        bias_ba_constraint: Constraint function applied to
            the `bias_ba` weights matrix
            (see [constraints](../constraints.md)).
        bias_ca_constraint: Constraint function applied to
            the `bias_ca` weights matrix
            (see [constraints](../constraints.md)).
        dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the context.
        recurrent_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the recurrent state.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        conditional_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the input.
        attention_dropout: Float between 0 and 1.
            Fraction of the units to drop for
            the linear transformation of the attention mechanism.
        num_inputs: Number of inputs of the layer.

    # References
        - [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259)
        - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](http://arxiv.org/abs/1412.3555v1)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 att_units=0,
                 return_extra_variables=False,
                 return_states=False,
                 attention_mode='add',
                 activation='tanh',
                 recurrent_activation='sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 conditional_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 attention_recurrent_initializer='glorot_uniform',
                 attention_context_initializer='glorot_uniform',
                 attention_context_wa_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 bias_ba_initializer='zeros',
                 bias_ca_initializer='zero',
                 unit_forget_bias=True,
                 mask_value=0.,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 conditional_regularizer=None,
                 attention_recurrent_regularizer=None,
                 attention_context_regularizer=None,
                 attention_context_wa_regularizer=None,
                 bias_regularizer=None,
                 bias_ba_regularizer=None,
                 bias_ca_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 conditional_constraint=None,
                 attention_recurrent_constraint=None,
                 attention_context_constraint=None,
                 attention_context_wa_constraint=None,
                 bias_constraint=None,
                 bias_ba_constraint=None,
                 bias_ca_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 conditional_dropout=0.,
                 attention_dropout=0.,
                 num_inputs=4,
                 **kwargs):
        super(AttLSTMCond, self).__init__(**kwargs)
        self.return_extra_variables = return_extra_variables
        self.return_states = return_states

        # Main parameters
        self.units = units
        self.att_units = units if att_units == 0 else att_units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias
        self.mask_value = mask_value
        self.attention_mode = attention_mode.lower()
        # Initializers
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.conditional_initializer = initializers.get(conditional_initializer)
        self.attention_recurrent_initializer = initializers.get(attention_recurrent_initializer)
        self.attention_context_initializer = initializers.get(attention_context_initializer)
        self.attention_context_wa_initializer = initializers.get(attention_context_wa_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_ba_initializer = initializers.get(bias_ba_initializer)
        self.bias_ca_initializer = initializers.get(bias_ca_initializer)
        self.unit_forget_bias = unit_forget_bias

        # Regularizers
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.conditional_regularizer = regularizers.get(conditional_regularizer)
        self.attention_recurrent_regularizer = regularizers.get(attention_recurrent_regularizer)
        self.attention_context_regularizer = regularizers.get(attention_context_regularizer)
        self.attention_context_wa_regularizer = regularizers.get(attention_context_wa_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_ba_regularizer = regularizers.get(bias_ba_regularizer)
        self.bias_ca_regularizer = regularizers.get(bias_ca_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        # Constraints
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.conditional_constraint = constraints.get(conditional_constraint)
        self.attention_recurrent_constraint = constraints.get(attention_recurrent_constraint)
        self.attention_context_constraint = constraints.get(attention_context_constraint)
        self.attention_context_wa_constraint = constraints.get(attention_context_wa_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.bias_ba_constraint = constraints.get(bias_ba_constraint)
        self.bias_ca_constraint = constraints.get(bias_ca_constraint)

        # Dropouts
        self.dropout = min(1., max(0., dropout)) if dropout is not None else 0.
        self.recurrent_dropout = min(1., max(0., recurrent_dropout)) if recurrent_dropout is not None else 0.
        self.conditional_dropout = min(1., max(0., conditional_dropout)) if conditional_dropout is not None else 0.
        self.attention_dropout = min(1., max(0., attention_dropout)) if attention_dropout is not None else 0.
        self.num_inputs = num_inputs
        self.input_spec = [InputSpec(ndim=3), InputSpec(ndim=3)]
        for _ in range(len(self.input_spec), self.num_inputs):
            self.input_spec.append(InputSpec(ndim=2))

    def build(self, input_shape):

        assert len(input_shape) >= 2, 'You should pass two inputs to AttLSTMCond ' \
                                      '(previous_embedded_words and context) ' \
                                      'and two optional inputs (init_state and init_memory)'
        self.input_dim = input_shape[0][2]
        self.context_steps = input_shape[1][1]
        self.context_dim = input_shape[1][2]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensors of shape (units)
            self.states = [None, None, None]  # [h, c, x_att]

        self.kernel = self.add_weight(shape=(self.context_dim, self.units * 4),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.conditional_kernel = self.add_weight(shape=(self.input_dim, self.units * 4),
                                                  name='conditional_kernel',
                                                  initializer=self.conditional_initializer,
                                                  regularizer=self.conditional_regularizer,
                                                  constraint=self.conditional_constraint)

        self.attention_recurrent_kernel = self.add_weight(
            shape=(self.units, self.att_units),
            name='attention_recurrent_kernel',
            initializer=self.attention_recurrent_initializer,
            regularizer=self.attention_recurrent_regularizer,
            constraint=self.attention_recurrent_constraint)

        self.attention_context_kernel = self.add_weight(
            shape=(self.context_dim, self.att_units),
            name='attention_context_kernel',
            initializer=self.attention_context_initializer,
            regularizer=self.attention_context_regularizer,
            constraint=self.attention_context_constraint)

        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            self.attention_context_wa = self.add_weight(
                shape=(self.att_units,),
                name='attention_context_wa',
                initializer=self.attention_context_wa_initializer,
                regularizer=self.attention_context_wa_regularizer,
                constraint=self.attention_context_wa_constraint)

        else:
            self.attention_context_wa = None

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.bias_ba = self.add_weight(shape=(self.att_units,),
                                       name='bias_ba',
                                       initializer=self.bias_ba_initializer,
                                       regularizer=self.bias_ba_regularizer,
                                       constraint=self.bias_ba_constraint)
        if self.attention_mode == 'add' or self.attention_mode == 'bahdanau':
            bias_ca_shape = self.context_steps if self.context_steps is None else (self.context_steps,)
            self.bias_ca = self.add_weight(shape=bias_ca_shape,
                                           name='bias_ca',
                                           initializer=self.bias_ca_initializer,
                                           regularizer=self.bias_ca_regularizer,
                                           constraint=self.bias_ca_constraint)
        else:
            self.bias_ca = None
        self.built = True

    def reset_states(self, states=None):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.units)))
            K.set_value(self.states[2],
                        np.zeros((input_shape[0], input_shape[3])))
        else:
            self.states = [K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], self.units)),
                           K.zeros((input_shape[0], input_shape[3]))]

    def preprocess_input(self, inputs, training=None):

        if 0 < self.conditional_dropout < 1:
            ones = K.ones_like(K.squeeze(inputs[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.conditional_dropout)

            cond_dp_mask = [K.in_train_phase(dropped_inputs,
                                             ones,
                                             training=training) for _ in range(4)]
            return K.dot(inputs * cond_dp_mask[0][:, None, :], self.conditional_kernel)
        else:
            return K.dot(inputs, self.conditional_kernel)

    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            main_out = (input_shape[0][0], input_shape[0][1], self.units)
        else:
            main_out = (input_shape[0][0], self.units)

        if self.return_extra_variables:
            dim_x_att = (input_shape[0][0], input_shape[0][1], self.context_dim)
            dim_alpha_att = (input_shape[0][0], input_shape[0][1], input_shape[1][1])
            main_out = [main_out, dim_x_att, dim_alpha_att]

        if self.return_states:
            if not isinstance(main_out, list):
                main_out = [main_out]
            states_dim = (input_shape[0][0], input_shape[0][1], self.units)
            main_out += [states_dim, states_dim]

        return main_out

    def _ln(self, x, slc):
        # sample-wise normalization
        m = K.mean(x, axis=-1, keepdims=True)
        std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon_layer_normalization)
        x_normed = (x - m) / (std + self.epsilon_layer_normalization)
        x_normed = eval('self.gamma_' + slc) * x_normed + eval('self.beta_' + slc)
        return x_normed

    def call(self, inputs, mask=None, training=None, initial_state=None):
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = K.int_shape(inputs[0])
        state_below = inputs[0]
        self.context = inputs[1]
        if self.num_inputs == 2:  # input: [state_below, context]
            self.init_state = None
            self.init_memory = None
        elif self.num_inputs == 3:  # input: [state_below, context, init_generic]
            self.init_state = inputs[2]
            self.init_memory = inputs[2]
        elif self.num_inputs == 4:  # input: [state_below, context, init_state, init_memory]
            self.init_state = inputs[2]
            self.init_memory = inputs[3]

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(state_below)
        constants = self.get_constants(state_below, mask[1], training=training)
        preprocessed_input = self.preprocess_input(state_below, training=training)
        last_output, outputs, states = K.rnn(self.step,
                                             preprocessed_input,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask[0],
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=K.shape(state_below)[1],
                                             pos_extra_outputs_states=[2, 3])
        if self.stateful:
            self.updates = []
            for i in range(len(states)):
                self.updates.append((self.states[i], states[i]))

        # Properly set learning phase
        if 0 < self.dropout + self.recurrent_dropout:
            last_output._uses_learning_phase = True
            outputs._uses_learning_phase = True

        if self.return_sequences:
            ret = outputs
        else:
            ret = last_output

        if self.return_extra_variables:
            ret = [ret, states[2], states[3]]

        # intermediate states as additional outputs
        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [states[0], states[1]]
        return ret

    def compute_mask(self, input, mask):
        if self.return_extra_variables:
            ret = [mask[0], mask[0], mask[0]]
        else:
            ret = mask[0]

        if self.return_states:
            if not isinstance(ret, list):
                ret = [ret]
            ret += [mask[0], mask[0]]

        return ret

    def step(self, x, states):
        h_tm1 = states[0]  # State
        c_tm1 = states[1]  # Memory
        non_used_x_att = states[2]  # Placeholder for returning extra variables
        non_used_alphas_att = states[3]  # Placeholder for returning extra variables
        dp_mask = states[4]  # Dropout W (input)
        rec_dp_mask = states[5]  # Dropout U (recurrent)
        # Att model dropouts
        att_dp_mask = states[6]  # Dropout Wa
        pctx_ = states[7]  # Projected context (i.e. context * Ua + ba)
        context = states[8]  # Original context
        mask_context = states[9]  # Context mask
        if K.ndim(mask_context) > 1:  # Mask the context (only if necessary)
            pctx_ = K.cast(mask_context[:, :, None], K.dtype(pctx_)) * pctx_
            context = K.cast(mask_context[:, :, None], K.dtype(context)) * context

        ctx_, alphas = compute_attention(h_tm1, pctx_, context, att_dp_mask, self.attention_recurrent_kernel,
                                         self.attention_context_wa, self.bias_ca, mask_context,
                                         attention_mode=self.attention_mode)
        # LSTM
        z = x + \
            K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel) + \
            K.dot(ctx_ * dp_mask[0], self.kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)
        z0 = z[:, :self.units]
        z1 = z[:, self.units: 2 * self.units]
        z2 = z[:, 2 * self.units: 3 * self.units]
        z3 = z[:, 3 * self.units:]
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        o = self.recurrent_activation(z3)
        c = f * c_tm1 + i * self.activation(z2)
        h = o * self.activation(c)
        if 0 < self.dropout + self.recurrent_dropout:
            h._uses_learning_phase = True
        return h, [h, c, ctx_, alphas]

    def get_constants(self, inputs, mask_context, training=None):
        constants = []
        # States[4] - Dropout_W
        if 0 < self.dropout < 1:
            ones = K.ones_like(K.squeeze(self.context[:, 0:1, :], axis=1))

            def dropped_inputs():
                return K.dropout(ones, self.dropout)

            dp_mask = [K.in_train_phase(dropped_inputs,
                                        ones,
                                        training=training) for _ in range(4)]
            constants.append(dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[5] - Dropout_U
        if 0 < self.recurrent_dropout < 1:
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.units))

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            rec_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training) for _ in range(4)]
            constants.append(rec_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        # States[6]  - Dropout_Wa
        if 0 < self.attention_dropout < 1:
            input_dim = self.units
            ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)

            def dropped_inputs():
                return K.dropout(ones, self.recurrent_dropout)

            att_dp_mask = [K.in_train_phase(dropped_inputs,
                                            ones,
                                            training=training)]
            constants.append(att_dp_mask)
        else:
            constants.append([K.cast_to_floatx(1.)])

        if 0 < self.attention_dropout < 1:
            input_dim = self.context_dim
            ones = K.ones_like(K.reshape(self.context[:, :, 0], (-1, K.shape(self.context)[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.attention_dropout), ones)]
            pctx = K.dot(self.context * B_Ua[0], self.attention_context_kernel)
        else:
            pctx = K.dot(self.context, self.attention_context_kernel)
        if self.use_bias:
            pctx = K.bias_add(pctx, self.bias_ba)
        # States[7] - pctx_
        constants.append(pctx)

        # States[8] - context
        constants.append(self.context)

        # States[9] - mask_context
        if mask_context is None:
            mask_context = K.not_equal(K.sum(self.context, axis=2), self.mask_value)
            mask_context = K.cast(mask_context, K.floatx())
        constants.append(mask_context)

        return constants

    def get_initial_states(self, inputs):
        # build an all-zero tensor of shape (samples, units)
        if self.init_state is None:
            initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
            initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
            initial_state = K.expand_dims(initial_state)  # (samples, 1)
            initial_state = K.tile(initial_state, [1, self.units])  # (samples, units)
            if self.init_memory is None:
                initial_states = [initial_state for _ in range(2)]
            else:
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
        else:
            initial_state = self.init_state
            if self.init_memory is not None:  # We have state and memory
                initial_memory = self.init_memory
                initial_states = [initial_state, initial_memory]
            else:
                initial_states = [initial_state for _ in range(2)]

        initial_state = K.zeros_like(self.context)  # (samples, input_timesteps, ctx_dim)
        initial_state_alphas = K.sum(initial_state, axis=2)  # (samples, input_timesteps)
        initial_state = K.sum(initial_state, axis=1)  # (samples, ctx_dim)
        extra_states = [initial_state, initial_state_alphas]  # (samples, ctx_dim)

        return initial_states + extra_states


