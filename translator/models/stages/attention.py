from keras.models import Model
from keras import layers, regularizers, activations, initializers, constraints
from keras.legacy import interfaces
from keras import backend as K
import warnings

from ... import constants as CONST

def sqrtScaleValues(x):
	from keras import backend as K
	scale = K.sqrt(K.cast(CONST.ATTENTION_UNITS, "float32"))
	return x/scale

def hideFutureSteps(x):
	from keras import backend as K

	m = K.arange(K.shape(x)[1])
	m1 = K.tile(K.expand_dims(m, 0), [K.shape(x)[1], 1])
	m2 = K.tile(K.expand_dims(m, 1), [1, K.shape(x)[1]])
	mask = K.cast(K.greater_equal(m2, m1), "float32")
	mask = K.tile(K.expand_dims(mask, 0), [K.shape(x)[0], 1, 1])

	lowValue = K.cast(K.less(m2, m1), "float32") * K.cast(-2**15, "float32")
	lowValue = K.tile(K.expand_dims(lowValue, 0), [K.shape(x)[0], 1, 1])

	return (x * mask) + lowValue

def mergeMultihead(x):
	batchSize = K.shape(x)[0]
	timeSteps = K.shape(x)[1]
	
	xOut = K.reshape(x, (batchSize, timeSteps, CONST.NUM_ATTENTION_HEADS, -1))
	xOut = K.permute_dimensions(xOut, (0, 2, 1, 3))
	xOut = K.reshape(xOut, (batchSize * CONST.NUM_ATTENTION_HEADS, timeSteps, -1))
	return xOut

def splitMultihead(x):
	timeSteps = K.shape(x)[1]
	headVectorSize = K.shape(x)[2]

	xOut = K.reshape(x, (-1, CONST.NUM_ATTENTION_HEADS, timeSteps, headVectorSize))
	xOut = K.permute_dimensions(xOut, (0, 2, 1, 3))
	xOut = K.reshape(xOut, (-1, timeSteps, headVectorSize * CONST.NUM_ATTENTION_HEADS))
	return xOut

def meanMultihead(x):
	xOut = K.reshape(x, (-1, CONST.NUM_ATTENTION_HEADS, K.shape(x)[1], K.shape(x)[2]))
	xOut = K.mean(xOut, axis=1)
	return xOut


def basicAttentionStage(count=0):
	query = layers.Input(batch_shape=(None,None,CONST.MODEL_BASE_UNITS))
	key = layers.Input(batch_shape=(None,None,CONST.MODEL_BASE_UNITS))

	#key query pair
	queryAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=regularizers.l2(CONST.L2_REGULARISATION)))(query)
	keyAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=regularizers.l2(CONST.L2_REGULARISATION)))(key)
	
	#### attention!
	#generate alphas
	alphas = layers.Dot(axes=2)([queryAttentionIn, keyAttentionIn])
	alphas = layers.Lambda(sqrtScaleValues)(alphas)
	alphas = layers.TimeDistributed(layers.Activation("softmax"))(alphas)
	#create weighted encoder context
	valuePerm = layers.Permute((2,1))(key)
	contextOut = layers.Dot(axes=2)([alphas, valuePerm])

	if CONST.BATCH_NORMALIZATION:
		contextOut = layers.TimeDistributed(layers.BatchNormalization(**CONST.BATCH_NORMALIZATION_ARGUMENTS))(contextOut)

	attentionModel = Model(inputs=[query, key], outputs=[contextOut, alphas], name="attention_stage_"+str(count))
	return attentionModel


def multiHeadAttentionStage(count=0, hideFuture=False, feedForward=True):
	query = layers.Input(batch_shape=(None,None,CONST.MODEL_BASE_UNITS))
	key = layers.Input(batch_shape=(None,None,CONST.MODEL_BASE_UNITS))

	queryAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=regularizers.l2(CONST.L2_REGULARISATION)))(query)
	keyAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=regularizers.l2(CONST.L2_REGULARISATION)))(key)
	valueAttentionIn = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=regularizers.l2(CONST.L2_REGULARISATION)))(key)

	queryAttentionIn = layers.Lambda(mergeMultihead)(queryAttentionIn)
	keyAttentionIn = layers.Lambda(mergeMultihead)(keyAttentionIn)
	valueAttentionIn = layers.Lambda(mergeMultihead)(valueAttentionIn)

	#generate alphas
	alphas = layers.Dot(axes=2)([queryAttentionIn, keyAttentionIn])
	alphas = layers.Lambda(sqrtScaleValues)(alphas)
	if hideFuture:
		alphas = layers.Lambda(hideFutureSteps)(alphas)
	alphas = layers.TimeDistributed(layers.Activation("softmax"))(alphas)

	#create weighted encoder context
	valuePerm = layers.Permute((2, 1))(valueAttentionIn)
	contextOut = layers.Dot(axes=2)([alphas, valuePerm])

	contextOut = layers.Lambda(splitMultihead)(contextOut)
	alphas = layers.Lambda(meanMultihead)(alphas)

	contextOut = layers.Add()([query, contextOut])
	contextOut = layers.Reshape(target_shape=(-1, CONST.MODEL_BASE_UNITS))(contextOut)
	if CONST.BATCH_NORMALIZATION:
		contextOut = layers.TimeDistributed(layers.BatchNormalization(**CONST.BATCH_NORMALIZATION_ARGUMENTS))(contextOut)

	if feedForward:
		contextOutFF = layers.TimeDistributed(layers.Dense(CONST.FEED_FORWARD_UNITS, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=regularizers.l2(CONST.L2_REGULARISATION)))(contextOut)
		if CONST.BATCH_NORMALIZATION:
			contextOutFF = layers.TimeDistributed(layers.BatchNormalization(**CONST.BATCH_NORMALIZATION_ARGUMENTS))(contextOutFF)
		contextOutFF = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=regularizers.l2(CONST.L2_REGULARISATION)))(contextOutFF)
		contextOut = layers.Add()([contextOut, contextOutFF])
		if CONST.BATCH_NORMALIZATION:
			contextOut = layers.TimeDistributed(layers.BatchNormalization(**CONST.BATCH_NORMALIZATION_ARGUMENTS))(contextOut)

	attentionModel = Model(inputs=[query, key], outputs=[contextOut, alphas], name="multihead_attention_stage_"+str(count))
	return attentionModel


class AttLSTMCell(layers.LSTMCell):
	def __init__(self, units, **kwargs):
		super(AttLSTMCell, self).__init__(units, **kwargs)
	
	def call(self, inputs, states, training=None):
		attKey = states.pop()



		h, states = super(AttLSTMCell, self).call(inputs, states, training)
		states.append(attKey)
		return h, states


class AttLSTM(layers.RNN):
	"""Long Short-Term Memory layer - Hochreiter 1997.

	# Arguments
		units: Positive integer, dimensionality of the output space.
		activation: Activation function to use
			(see [activations](../activations.md)).
			Default: hyperbolic tangent (`tanh`).
			If you pass `None`, no activation is applied
			(ie. "linear" activation: `a(x) = x`).
		recurrent_activation: Activation function to use
			for the recurrent step
			(see [activations](../activations.md)).
			Default: hard sigmoid (`hard_sigmoid`).
			If you pass `None`, no activation is applied
			(ie. "linear" activation: `a(x) = x`).
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
			This is recommended in [Jozefowicz et al. (2015)](
			http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf).
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
		implementation: Implementation mode, either 1 or 2.
			Mode 1 will structure its operations as a larger number of
			smaller dot products and additions, whereas mode 2 will
			batch them into fewer, larger operations. These modes will
			have different performance profiles on different hardware and
			for different applications.
		return_sequences: Boolean. Whether to return the last output
			in the output sequence, or the full sequence.
		return_state: Boolean. Whether to return the last state
			in addition to the output. The returned elements of the
			states list are the hidden state and the cell state, respectively.
		go_backwards: Boolean (default False).
			If True, process the input sequence backwards and return the
			reversed sequence.
		stateful: Boolean (default False). If True, the last state
			for each sample at index i in a batch will be used as initial
			state for the sample of index i in the following batch.
		unroll: Boolean (default False).
			If True, the network will be unrolled,
			else a symbolic loop will be used.
			Unrolling can speed-up a RNN,
			although it tends to be more memory-intensive.
			Unrolling is only suitable for short sequences.

	# References
		- [Long short-term memory](
		  http://www.bioinf.jku.at/publications/older/2604.pdf)
		- [Learning to forget: Continual prediction with LSTM](
		  http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
		- [Supervised sequence labeling with recurrent neural networks](
		  http://www.cs.toronto.edu/~graves/preprint.pdf)
		- [A Theoretically Grounded Application of Dropout in
		   Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
	"""

	@interfaces.legacy_recurrent_support
	def __init__(self, units,
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
				 activity_regularizer=None,
				 kernel_constraint=None,
				 recurrent_constraint=None,
				 bias_constraint=None,
				 dropout=0.,
				 recurrent_dropout=0.,
				 implementation=1,
				 return_sequences=False,
				 return_state=False,
				 go_backwards=False,
				 stateful=False,
				 unroll=False,
				 **kwargs):
		if implementation == 0:
			warnings.warn('`implementation=0` has been deprecated, '
						  'and now defaults to `implementation=1`.'
						  'Please update your layer call.')
		if K.backend() == 'theano' and (dropout or recurrent_dropout):
			warnings.warn(
				'RNN dropout is no longer supported with the Theano backend '
				'due to technical limitations. '
				'You can either set `dropout` and `recurrent_dropout` to 0, '
				'or use the TensorFlow backend.')
			dropout = 0.
			recurrent_dropout = 0.

		cell = AttLSTMCell(units,
						activation=activation,
						recurrent_activation=recurrent_activation,
						use_bias=use_bias,
						kernel_initializer=kernel_initializer,
						recurrent_initializer=recurrent_initializer,
						unit_forget_bias=unit_forget_bias,
						bias_initializer=bias_initializer,
						kernel_regularizer=kernel_regularizer,
						recurrent_regularizer=recurrent_regularizer,
						bias_regularizer=bias_regularizer,
						kernel_constraint=kernel_constraint,
						recurrent_constraint=recurrent_constraint,
						bias_constraint=bias_constraint,
						dropout=dropout,
						recurrent_dropout=recurrent_dropout,
						implementation=implementation)
		super(AttLSTM, self).__init__(cell,
								   return_sequences=return_sequences,
								   return_state=return_state,
								   go_backwards=go_backwards,
								   stateful=stateful,
								   unroll=unroll,
								   **kwargs)
		self.activity_regularizer = regularizers.get(activity_regularizer)

	def call(self, inputs, mask=None, training=None, initial_state=None):
		self.cell._dropout_mask = None
		self.cell._recurrent_dropout_mask = None
		return super(AttLSTM, self).call(inputs,
									  mask=mask,
									  training=training,
									  initial_state=initial_state)

	@property
	def units(self):
		return self.cell.units

	@property
	def activation(self):
		return self.cell.activation

	@property
	def recurrent_activation(self):
		return self.cell.recurrent_activation

	@property
	def use_bias(self):
		return self.cell.use_bias

	@property
	def kernel_initializer(self):
		return self.cell.kernel_initializer

	@property
	def recurrent_initializer(self):
		return self.cell.recurrent_initializer

	@property
	def bias_initializer(self):
		return self.cell.bias_initializer

	@property
	def unit_forget_bias(self):
		return self.cell.unit_forget_bias

	@property
	def kernel_regularizer(self):
		return self.cell.kernel_regularizer

	@property
	def recurrent_regularizer(self):
		return self.cell.recurrent_regularizer

	@property
	def bias_regularizer(self):
		return self.cell.bias_regularizer

	@property
	def kernel_constraint(self):
		return self.cell.kernel_constraint

	@property
	def recurrent_constraint(self):
		return self.cell.recurrent_constraint

	@property
	def bias_constraint(self):
		return self.cell.bias_constraint

	@property
	def dropout(self):
		return self.cell.dropout

	@property
	def recurrent_dropout(self):
		return self.cell.recurrent_dropout

	@property
	def implementation(self):
		return self.cell.implementation

	def get_config(self):
		config = {'units': self.units,
				  'activation': activations.serialize(self.activation),
				  'recurrent_activation':
					  activations.serialize(self.recurrent_activation),
				  'use_bias': self.use_bias,
				  'kernel_initializer':
					  initializers.serialize(self.kernel_initializer),
				  'recurrent_initializer':
					  initializers.serialize(self.recurrent_initializer),
				  'bias_initializer': initializers.serialize(self.bias_initializer),
				  'unit_forget_bias': self.unit_forget_bias,
				  'kernel_regularizer':
					  regularizers.serialize(self.kernel_regularizer),
				  'recurrent_regularizer':
					  regularizers.serialize(self.recurrent_regularizer),
				  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
				  'activity_regularizer':
					  regularizers.serialize(self.activity_regularizer),
				  'kernel_constraint': constraints.serialize(self.kernel_constraint),
				  'recurrent_constraint':
					  constraints.serialize(self.recurrent_constraint),
				  'bias_constraint': constraints.serialize(self.bias_constraint),
				  'dropout': self.dropout,
				  'recurrent_dropout': self.recurrent_dropout,
				  'implementation': self.implementation}
		base_config = super(AttLSTM, self).get_config()
		del base_config['cell']
		return dict(list(base_config.items()) + list(config.items()))

	@classmethod
	def from_config(cls, config):
		if 'implementation' in config and config['implementation'] == 0:
			config['implementation'] = 1
		return cls(**config)

