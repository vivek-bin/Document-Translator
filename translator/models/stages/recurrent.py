from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
import tensorflow as tf


class EncoderLSTMCell(DropoutRNNCellMixin, Layer):
	"""
	Specific adaptations:
	- depth : stack similar layers for deep encoder
	- always use fused weights, ie the four gate inputs are calcualted together with a single combined weight matrix
	- no recurrent dropout available
	- optional peephole connections to cell state
	- optional layer normalization

	"""

	def __init__(self,
					units,
					depth=1,
					use_residual_connection=False,
					use_layer_norm=False,
					use_peephole=False,
					activation='tanh',
					recurrent_activation='hard_sigmoid',
					use_bias=True,
					kernel_initializer='glorot_uniform',
					recurrent_initializer='orthogonal',
					bias_initializer='zeros',
					unit_forget_bias=True,
					kernel_regularizer=None,
					recurrent_regularizer=None,
					bias_regularizer=None,
					kernel_constraint=None,
					recurrent_constraint=None,
					bias_constraint=None,
					beta_initializer='zeros',
					gamma_initializer='ones',
					beta_regularizer=None,
					gamma_regularizer=None,
					beta_constraint=None,
					gamma_constraint=None,
					dropout=0.,
					recurrent_dropout=0.,
					**kwargs):
		assert depth > 0
		assert recurrent_dropout == 0

		self._enable_caching_device = False
		super(EncoderLSTMCell, self).__init__(**kwargs)
		self.units = units
		self.depth = depth
		self.use_residual_connection = use_residual_connection
		self.use_layer_norm = use_layer_norm
		self.use_peephole = use_peephole

		self.activation = activations.get(activation)
		self.recurrent_activation = activations.get(recurrent_activation)
		self.use_bias = use_bias

		self.kernel_initializer = initializers.get(kernel_initializer)
		self.recurrent_initializer = initializers.get(recurrent_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.unit_forget_bias = unit_forget_bias

		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)

		self.kernel_constraint = constraints.get(kernel_constraint)
		self.recurrent_constraint = constraints.get(recurrent_constraint)
		self.bias_constraint = constraints.get(bias_constraint)

		self.beta_initializer = initializers.get(beta_initializer)
		self.gamma_initializer = initializers.get(gamma_initializer)
		self.beta_regularizer = regularizers.get(beta_regularizer)
		self.gamma_regularizer = regularizers.get(gamma_regularizer)
		self.beta_constraint = constraints.get(beta_constraint)
		self.gamma_constraint = constraints.get(gamma_constraint)

		self.dropout = min(1., max(0., dropout))
		self.recurrent_dropout = 0
		self.state_size = data_structures.NoDependency([self.units for _ in range(1 + 2*self.depth)])
		self.output_size = self.units

	@tf_utils.shape_type_conversion
	def build(self, input_shape):
		input_dim = input_shape[-1]
		self.kernel = []
		self.recurrent_kernel = []
		self.bias = []
		self.input_gate_peephole_weights = []
		self.forget_gate_peephole_weights = []
		self.output_gate_peephole_weights = []
		self.gamma = []
		self.beta = []

		for d in range(self.depth):
			self.kernel.append(self.add_weight(
					shape=(input_dim, self.units * 4),
					name="kernel_"+str(d),
					initializer=self.kernel_initializer,
					regularizer=self.kernel_regularizer,
					constraint=self.kernel_constraint))
			self.recurrent_kernel.append(self.add_weight(
					shape=(self.units, self.units * 4),
					name="recurrent_kernel_"+str(d),
					initializer=self.recurrent_initializer,
					regularizer=self.recurrent_regularizer,
					constraint=self.recurrent_constraint))

			if self.use_bias:
				if self.unit_forget_bias:
					def bias_initializer(_, *args, **kwargs):
						return K.concatenate([
								self.bias_initializer((self.units,), *args, **kwargs),
								initializers.Ones()((self.units,), *args, **kwargs),
								self.bias_initializer((self.units * 2,), *args, **kwargs),
						])
				else:
					bias_initializer = self.bias_initializer
				self.bias.append(self.add_weight(
						shape=(self.units * 4,),
						name="bias_"+str(d),
						initializer=bias_initializer,
						regularizer=self.bias_regularizer,
						constraint=self.bias_constraint))
			else:
				self.bias.append(None)

			if self.use_peephole:
				self.input_gate_peephole_weights.append(self.add_weight(
						shape=(self.units,),
						name="input_gate_peephole_weights_"+str(d),
						initializer=self.kernel_initializer))
				self.forget_gate_peephole_weights.append(self.add_weight(
						shape=(self.units,),
						name="forget_gate_peephole_weights_"+str(d),
						initializer=self.kernel_initializer))
				self.output_gate_peephole_weights.append(self.add_weight(
						shape=(self.units,),
						name="output_gate_peephole_weights_"+str(d),
						initializer=self.kernel_initializer))
			else:
				self.input_gate_peephole_weights.append(None)
				self.forget_gate_peephole_weights.append(None)
				self.output_gate_peephole_weights.append(None)


			if self.use_layer_norm:
				self.gamma.append(self.add_weight(shape=(self.units,),
											name="gamma_"+str(d),
											initializer=self.gamma_initializer,
											regularizer=self.gamma_regularizer,
											constraint=self.gamma_constraint))
				
				self.beta.append(self.add_weight(shape=(self.units,),
											name="beta_"+str(d),
											initializer=self.beta_initializer,
											regularizer=self.beta_regularizer,
											constraint=self.beta_constraint))
			else:
				self.gamma.append(None)
				self.beta.append(None)

		super(EncoderLSTMCell, self).build(input_shape)  # Be sure to call this at the end
		self.built = True

	def _compute_carry_and_output_fused(self, z, c_tm1, d):
		"""Computes carry and output using fused kernels."""
		z0, z1, z2, z3 = array_ops.split(z, num_or_size_splits=4, axis=1)
		
		if self.use_peephole:
			i = self.recurrent_activation(z0 + self.input_gate_peephole_weights[d] * c_tm1)
			f = self.recurrent_activation(z1 + self.forget_gate_peephole_weights[d] * c_tm1)
			c = f * c_tm1 + i * self.activation(z2)
			o = self.recurrent_activation(z3 + self.output_gate_peephole_weights[d] * c)
		else:
			i = self.recurrent_activation(z0)
			f = self.recurrent_activation(z1)
			c = f * c_tm1 + i * self.activation(z2)
			o = self.recurrent_activation(z3)
		h = o * self.activation(c)
		return c, h

	def call(self, inputs, states, training=None):
		outputStates = []
		dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=self.depth)

		for d in range(self.depth):
			h_tm1 = states[1 + 0 + d*2]	# previous memory state
			c_tm1 = states[1 + 1 + d*2]	# previous carry state

			if 0. < self.dropout < 1.:
				inputs = inputs * dp_mask[d]
			z = K.dot(inputs, self.kernel[d])
			z += K.dot(h_tm1, self.recurrent_kernel[d])
			if self.use_bias:
				z = K.bias_add(z, self.bias[d])

			c, h = self._compute_carry_and_output_fused(z, c_tm1, d)

			outputStates.extend([h, c])
			output = h

			if self.use_residual_connection:
				output = output + inputs

			if self.use_layer_norm:
				std = K.std(output, axis=-1, keepdims=True)
				mean = K.mean(output, axis=-1, keepdims=True)
				output = (output - mean) / (std + 1e-3)
				
				output = (output - self.beta[d]) * self.gamma[d]
			inputs = output

		outputStates.insert(0, output)
		return output, outputStates

	def get_config(self):
		config = {
				'units':				self.units,
				'depth':				self.depth,
				'use_residual_connection':self.use_residual_connection,
				'use_layer_norm':			self.use_layer_norm,
				'use_peephole':			self.use_peephole,

				'activation':			activations.serialize(self.activation),
				'recurrent_activation':	activations.serialize(self.recurrent_activation),
				'use_bias':				self.use_bias,
				'kernel_initializer':	initializers.serialize(self.kernel_initializer),
				'recurrent_initializer':initializers.serialize(self.recurrent_initializer),
				'bias_initializer':		initializers.serialize(self.bias_initializer),
				'unit_forget_bias':		self.unit_forget_bias,
				'kernel_regularizer':	regularizers.serialize(self.kernel_regularizer),
				'recurrent_regularizer':regularizers.serialize(self.recurrent_regularizer),
				'bias_regularizer':		regularizers.serialize(self.bias_regularizer),
				'kernel_constraint':	constraints.serialize(self.kernel_constraint),
				'recurrent_constraint':	constraints.serialize(self.recurrent_constraint),
				'bias_constraint':		constraints.serialize(self.bias_constraint),

				'beta_initializer': 	initializers.serialize(self.beta_initializer),
				'gamma_initializer': 	initializers.serialize(self.gamma_initializer),
				'beta_regularizer': 	regularizers.serialize(self.beta_regularizer),
				'gamma_regularizer': 	regularizers.serialize(self.gamma_regularizer),
				'beta_constraint': 		constraints.serialize(self.beta_constraint),
				'gamma_constraint': 	constraints.serialize(self.gamma_constraint),
				'dropout':				self.dropout,
				'recurrent_dropout':	self.recurrent_dropout
		}
		base_config = super(EncoderLSTMCell, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))



class DecoderLSTMCell(DropoutRNNCellMixin, Layer):
	"""
	Specific adaptations:
	- depth : stack similar layers for deep encoder
	- always use fused weights, ie the four gate inputs are calcualted together with a single combined weight matrix
	- no recurrent dropout available
	- optional peephole connections to cell state
	- optional layer normalization

	"""

	def __init__(self,
					units,
					depth=1,
					use_residual_connection=False,
					use_layer_norm=False,
					use_peephole=False,
					activation='tanh',
					recurrent_activation='hard_sigmoid',
					use_bias=True,
					kernel_initializer='glorot_uniform',
					recurrent_initializer='orthogonal',
					bias_initializer='zeros',
					unit_forget_bias=True,
					kernel_regularizer=None,
					recurrent_regularizer=None,
					bias_regularizer=None,
					kernel_constraint=None,
					recurrent_constraint=None,
					bias_constraint=None,
					beta_initializer='zeros',
					gamma_initializer='ones',
					beta_regularizer=None,
					gamma_regularizer=None,
					beta_constraint=None,
					gamma_constraint=None,
					dropout=0.,
					recurrent_dropout=0.,
					**kwargs):
		assert depth > 0
		assert recurrent_dropout == 0

		self._enable_caching_device = False
		super(DecoderLSTMCell, self).__init__(**kwargs)
		self.units = units
		self.depth = depth
		self.use_residual_connection = use_residual_connection
		self.use_layer_norm = use_layer_norm
		self.use_peephole = use_peephole

		self.activation = activations.get(activation)
		self.recurrent_activation = activations.get(recurrent_activation)
		self.use_bias = use_bias

		self.kernel_initializer = initializers.get(kernel_initializer)
		self.recurrent_initializer = initializers.get(recurrent_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		self.unit_forget_bias = unit_forget_bias

		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)

		self.kernel_constraint = constraints.get(kernel_constraint)
		self.recurrent_constraint = constraints.get(recurrent_constraint)
		self.bias_constraint = constraints.get(bias_constraint)

		self.beta_initializer = initializers.get(beta_initializer)
		self.gamma_initializer = initializers.get(gamma_initializer)
		self.beta_regularizer = regularizers.get(beta_regularizer)
		self.gamma_regularizer = regularizers.get(gamma_regularizer)
		self.beta_constraint = constraints.get(beta_constraint)
		self.gamma_constraint = constraints.get(gamma_constraint)

		self.dropout = min(1., max(0., dropout))
		self.recurrent_dropout = 0
		self.state_size = tuple([self.units for _ in range(1 + 2*self.depth)] + [(None, self.units), (1, None)])
		#self.state_size = data_structures.NoDependency(self.state_size)
		self.state_size = data_structures.NoDependency(tuple(tf.TensorShape(x) for x in self.state_size))
		self.output_size = self.units

	@tf_utils.shape_type_conversion
	def build(self, input_shape):
		input_dim = input_shape[-1]
		self.kernel = []
		self.recurrent_kernel = []
		self.recurrent_context_kernel = []
		self.bias = []
		self.input_gate_peephole_weights = []
		self.forget_gate_peephole_weights = []
		self.output_gate_peephole_weights = []
		self.gamma = []
		self.beta = []

		self.key_kernel = self.add_weight(
				shape=(self.units, self.units),
				name="att_key_kernel",
				initializer=self.kernel_initializer,
				regularizer=self.kernel_regularizer,
				constraint=self.kernel_constraint)
		self.key_bias = self.add_weight(
				shape=(self.units,),
				name="att_key_bias",
				initializer=self.bias_initializer,
				regularizer=self.bias_regularizer,
				constraint=self.bias_constraint)
		
		for d in range(self.depth):
			self.kernel.append(self.add_weight(
					shape=(input_dim, self.units * 4),
					name="kernel_"+str(d),
					initializer=self.kernel_initializer,
					regularizer=self.kernel_regularizer,
					constraint=self.kernel_constraint))
			self.recurrent_kernel.append(self.add_weight(
					shape=(self.units, self.units * 4),
					name="recurrent_kernel_"+str(d),
					initializer=self.recurrent_initializer,
					regularizer=self.recurrent_regularizer,
					constraint=self.recurrent_constraint))
			self.recurrent_context_kernel.append(self.add_weight(
					shape=(self.units, self.units * 4),
					name="recurrent_context_kernel_"+str(d),
					initializer=self.recurrent_initializer,
					regularizer=self.recurrent_regularizer,
					constraint=self.recurrent_constraint))

			if self.use_bias:
				if self.unit_forget_bias:
					def bias_initializer(_, *args, **kwargs):
						return K.concatenate([
								self.bias_initializer((self.units,), *args, **kwargs),
								initializers.Ones()((self.units,), *args, **kwargs),
								self.bias_initializer((self.units * 2,), *args, **kwargs),
						])
				else:
					bias_initializer = self.bias_initializer
				self.bias.append(self.add_weight(
						shape=(self.units * 4,),
						name="bias_"+str(d),
						initializer=bias_initializer,
						regularizer=self.bias_regularizer,
						constraint=self.bias_constraint))
			else:
				self.bias.append(None)

			if self.use_peephole:
				self.input_gate_peephole_weights.append(self.add_weight(
						shape=(self.units,),
						name="input_gate_peephole_weights_"+str(d),
						initializer=self.kernel_initializer))
				self.forget_gate_peephole_weights.append(self.add_weight(
						shape=(self.units,),
						name="forget_gate_peephole_weights_"+str(d),
						initializer=self.kernel_initializer))
				self.output_gate_peephole_weights.append(self.add_weight(
						shape=(self.units,),
						name="output_gate_peephole_weights_"+str(d),
						initializer=self.kernel_initializer))
			else:
				self.input_gate_peephole_weights.append(None)
				self.forget_gate_peephole_weights.append(None)
				self.output_gate_peephole_weights.append(None)


			if self.use_layer_norm:
				self.gamma.append(self.add_weight(shape=(self.units,),
											name="gamma_"+str(d),
											initializer=self.gamma_initializer,
											regularizer=self.gamma_regularizer,
											constraint=self.gamma_constraint))
				
				self.beta.append(self.add_weight(shape=(self.units,),
											name="beta_"+str(d),
											initializer=self.beta_initializer,
											regularizer=self.beta_regularizer,
											constraint=self.beta_constraint))
			else:
				self.gamma.append(None)
				self.beta.append(None)

		super(DecoderLSTMCell, self).build(input_shape)  # Be sure to call this at the end
		self.built = True

	def _compute_carry_and_output_fused(self, z, c_tm1, d):
		"""Computes carry and output using fused kernels."""
		z0, z1, z2, z3 = array_ops.split(z, num_or_size_splits=4, axis=1)
		
		if self.use_peephole:
			i = self.recurrent_activation(z0 + self.input_gate_peephole_weights[d] * c_tm1)
			f = self.recurrent_activation(z1 + self.forget_gate_peephole_weights[d] * c_tm1)
			c = f * c_tm1 + i * self.activation(z2)
			o = self.recurrent_activation(z3 + self.output_gate_peephole_weights[d] * c)
		else:
			i = self.recurrent_activation(z0)
			f = self.recurrent_activation(z1)
			c = f * c_tm1 + i * self.activation(z2)
			o = self.recurrent_activation(z3)
		
		h = o * self.activation(c)
		return c, h

	def _dotAttention(self, key, query, value):
		#generate alphas
		query = K.expand_dims(query, axis=1)
		alphaScale = K.sqrt(K.cast(self.units, K.dtype(self.key_kernel)))
		alphas = K.batch_dot(query, key, axes=-1)
		alphas = alphas / alphaScale
		alphas = K.softmax(alphas)
		#create weighted encoder context
		context = K.batch_dot(alphas, value, axes=[-1, 1])
		context = K.squeeze(context, axis=1)

		return context, alphas

	def _attention(self, key, query):
		#key->enc				query->dec			val->[->enc]
		#key query pair
		units = K.shape(key)[2]
		originalShape = K.shape(key)

		keyAttIn = K.reshape(key, (-1, units))
		keyAttIn = K.bias_add(K.dot(keyAttIn, self.key_kernel), self.key_bias)
		keyAttIn = K.relu(keyAttIn)
		keyAttIn = K.reshape(keyAttIn, originalShape)

		#### attention!
		context, alphas = self._dotAttention(keyAttIn, query, key)

		return context, alphas

	def call(self, inputs, states, training=None):
		outputStates = []
		dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=self.depth)

		for d in range(self.depth):
			h_tm1 = states[1 + 0 + d*2]	# previous memory state
			c_tm1 = states[1 + 1 + d*2]	# previous carry state

			if 0. < self.dropout < 1.:
				inputs = inputs * dp_mask[d]
			z = K.dot(inputs, self.kernel[d])
			z += K.dot(h_tm1, self.recurrent_kernel[d])
			z += K.dot(states[0], self.recurrent_context_kernel[d])
			if self.use_bias:
				z = K.bias_add(z, self.bias[d])

			c, h = self._compute_carry_and_output_fused(z, c_tm1, d)

			outputStates.extend([h, c])
			output = h

			if self.use_residual_connection:
				output = output + inputs

			if self.use_layer_norm:
				std = K.std(output, axis=-1, keepdims=True)
				mean = K.mean(output, axis=-1, keepdims=True)
				output = (output - mean) / (std + 1e-3)
				
				output = (output - self.beta[d]) * self.gamma[d]
			inputs = output
		
		context, alphas = self._attention(states[-2], output)
		output = context

		outputStates.insert(0, output)
		outputStates.append(states[-2])
		outputStates.append(alphas)
		return output, outputStates

	def get_config(self):
		config = {
				'units':				self.units,
				'depth':				self.depth,
				'use_residual_connection':self.use_residual_connection,
				'use_layer_norm':			self.use_layer_norm,
				'use_peephole':			self.use_peephole,

				'activation':			activations.serialize(self.activation),
				'recurrent_activation':	activations.serialize(self.recurrent_activation),
				'use_bias':				self.use_bias,
				'kernel_initializer':	initializers.serialize(self.kernel_initializer),
				'recurrent_initializer':initializers.serialize(self.recurrent_initializer),
				'bias_initializer':		initializers.serialize(self.bias_initializer),
				'unit_forget_bias':		self.unit_forget_bias,
				'kernel_regularizer':	regularizers.serialize(self.kernel_regularizer),
				'recurrent_regularizer':regularizers.serialize(self.recurrent_regularizer),
				'bias_regularizer':		regularizers.serialize(self.bias_regularizer),
				'kernel_constraint':	constraints.serialize(self.kernel_constraint),
				'recurrent_constraint':	constraints.serialize(self.recurrent_constraint),
				'bias_constraint':		constraints.serialize(self.bias_constraint),

				'beta_initializer': 	initializers.serialize(self.beta_initializer),
				'gamma_initializer': 	initializers.serialize(self.gamma_initializer),
				'beta_regularizer': 	regularizers.serialize(self.beta_regularizer),
				'gamma_regularizer': 	regularizers.serialize(self.gamma_regularizer),
				'beta_constraint': 		constraints.serialize(self.beta_constraint),
				'gamma_constraint': 	constraints.serialize(self.gamma_constraint),
				'dropout':				self.dropout,
				'recurrent_dropout':	self.recurrent_dropout
		}
		base_config = super(DecoderLSTMCell, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))



