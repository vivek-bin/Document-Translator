from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.engine.base_layer import DropoutRNNCellMixin
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest


class EncoderLSTMCell(DropoutRNNCellMixin, Layer):
	"""Cell class for the EncoderLSTM layer.

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
					useResidualConnection=False,
					useLayerNorm=False,
					usePeephole=False,
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
					**kwargs):
		self._enable_caching_device = kwargs.pop('enable_caching_device', False)
		super(EncoderLSTMCell, self).__init__(**kwargs)
		self.units = units
		self.depth = depth
		self.useResidualConnection = useResidualConnection
		self.useLayerNorm = useLayerNorm
		self.usePeephole = usePeephole

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
		self.state_size = data_structures.NoDependency([self.units, self.units])
		self.output_size = self.units

	@tf_utils.shape_type_conversion
	def build(self, input_shape):
		input_dim = input_shape[-1]
		self.kernel = self.add_weight(
				shape=(input_dim, self.units * 4),
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
			self.bias = self.add_weight(
					shape=(self.units * 4,),
					name='bias',
					initializer=bias_initializer,
					regularizer=self.bias_regularizer,
					constraint=self.bias_constraint)
		else:
			self.bias = None

		if self.usePeephole:
			self.input_gate_peephole_weights = self.add_weight(
					shape=(self.units,),
					name='input_gate_peephole_weights',
					initializer=self.kernel_initializer)
			self.forget_gate_peephole_weights = self.add_weight(
					shape=(self.units,),
					name='forget_gate_peephole_weights',
					initializer=self.kernel_initializer)
			self.output_gate_peephole_weights = self.add_weight(
					shape=(self.units,),
					name='output_gate_peephole_weights',
					initializer=self.kernel_initializer)

		if self.useLayerNorm:
			shape = (self.units,)
			self.gamma = self.add_weight(shape=shape,
										name='gamma',
										initializer=self.gamma_initializer,
										regularizer=self.gamma_regularizer,
										constraint=self.gamma_constraint)
			
			self.beta = self.add_weight(shape=shape,
										name='beta',
										initializer=self.beta_initializer,
										regularizer=self.beta_regularizer,
										constraint=self.beta_constraint)



		self.built = True

	def _compute_carry_and_output_fused(self, z, c_tm1):
		"""Computes carry and output using fused kernels."""
		z0, z1, z2, z3 = z
		if self.usePeephole:
			i = self.recurrent_activation(z0 +
						self.input_gate_peephole_weights * c_tm1)
			f = self.recurrent_activation(z1 +
						self.forget_gate_peephole_weights * c_tm1)
			c = f * c_tm1 + i * self.activation(z2)
			o = self.recurrent_activation(z3 + self.output_gate_peephole_weights * c)
		else:
			i = self.recurrent_activation(z0)
			f = self.recurrent_activation(z1)
			c = f * c_tm1 + i * self.activation(z2)
			o = self.recurrent_activation(z3)
		return c, o

	def call(self, inputs, states, training=None):
		h_tm1 = states[0]	# previous memory state
		c_tm1 = states[1]	# previous carry state

		if 0. < self.dropout < 1.:
			dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
			inputs = inputs * dp_mask[0]
		z = K.dot(inputs, self.kernel)
		z += K.dot(h_tm1, self.recurrent_kernel)
		if self.use_bias:
			z = K.bias_add(z, self.bias)

		z = array_ops.split(z, num_or_size_splits=4, axis=1)

		c, o = self._compute_carry_and_output_fused(z, c_tm1)
		h = o * self.activation(c)

		states = [h, c]
		output = h

		if self.useResidualConnection:
			output = output + inputs

		if self.useLayerNorm:
			std = K.std(output, axis=-1, keepdims=True)
			mean = K.mean(output, axis=-1, keepdims=True)
			output = (output - mean) / (std + self.epsilon)
			
			output = (output - self.beta) * self.gamma

		return output, states

	def get_config(self):
		config = {
				'units':				self.units,
				'depth':				self.depth,
				'useResidualConnection':self.useResidualConnection,
				'useLayerNorm':			self.useLayerNorm,
				'usePeephole':			self.usePeephole,

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

	def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
		if inputs is not None:
			batch_size = array_ops.shape(inputs)[0]
			dtype = inputs.dtype
		return list(_generate_zero_filled_state(batch_size, self.state_size, dtype))


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
	"""Generate a zero filled tensor with shape [batch_size, state_size]."""
	if batch_size_tensor is None or dtype is None:
		raise ValueError(
				'batch_size and dtype cannot be None while constructing initial state: '
				'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

	def create_zeros(unnested_state_size):
		flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
		init_state_size = [batch_size_tensor] + flat_dims
		return array_ops.zeros(init_state_size, dtype=dtype)

	if nest.is_sequence(state_size):
		return nest.map_structure(create_zeros, state_size)
	else:
		return create_zeros(state_size)


