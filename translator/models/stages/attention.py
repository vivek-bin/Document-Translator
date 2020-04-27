from tensorflow.keras.models import Model
from tensorflow.keras import layers, regularizers, activations, initializers, constraints
#from tensorflow.keras.legacy import interfaces
from tensorflow.keras import backend as K
import warnings

from ... import constants as CONST
from .normalize import LayerNormalization

def sqrtScaleValues(x):
	from tensorflow.keras import backend as K
	scale = K.sqrt(K.cast(CONST.ATTENTION_UNITS, K.dtype(x)))
	return x/scale

def hideFutureSteps(x):
	from tensorflow.keras import backend as K

	m = K.arange(0, K.shape(x)[1])
	m1 = K.tile(K.expand_dims(m, 0), [K.shape(x)[1], 1])
	m2 = K.tile(K.expand_dims(m, 1), [1, K.shape(x)[1]])
	mask = K.cast(K.greater_equal(m2, m1), K.dtype(x))
	mask = K.tile(K.expand_dims(mask, 0), [K.shape(x)[0], 1, 1])

	lowValue = K.cast(K.less(m2, m1), K.dtype(x)) * K.cast(-2**15, K.dtype(x))
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

	if CONST.LAYER_NORMALIZATION:
		contextOut = LayerNormalization(**CONST.LAYER_NORMALIZATION_ARGUMENTS)(contextOut)

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
	if CONST.LAYER_NORMALIZATION:
		contextOut = LayerNormalization(**CONST.LAYER_NORMALIZATION_ARGUMENTS)(contextOut)

	if feedForward:
		contextOutFF = layers.TimeDistributed(layers.Dense(CONST.FEED_FORWARD_UNITS, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=regularizers.l2(CONST.L2_REGULARISATION)))(contextOut)
		if CONST.LAYER_NORMALIZATION:
			contextOutFF = LayerNormalization(**CONST.LAYER_NORMALIZATION_ARGUMENTS)(contextOutFF)
		contextOutFF = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=regularizers.l2(CONST.L2_REGULARISATION)))(contextOutFF)
		contextOut = layers.Add()([contextOut, contextOutFF])
		if CONST.LAYER_NORMALIZATION:
			contextOut = LayerNormalization(**CONST.LAYER_NORMALIZATION_ARGUMENTS)(contextOut)

	attentionModel = Model(inputs=[query, key], outputs=[contextOut, alphas], name="multihead_attention_stage_"+str(count))
	return attentionModel
