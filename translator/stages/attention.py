from keras.models import Model
from keras import layers

from .. import constants as CONST

def sqrtScaleValues(x):
	from keras import backend as K
	scale = K.sqrt(K.cast(CONST.ATTENTION_UNITS, "float32"))
	return x/scale

def sqrtScaleHideFuture(x):
	from keras import backend as K

	m = K.arange(K.shape(x)[1])
	m1 = K.tile(K.expand_dims(m, 0), [K.shape(x)[1], 1])
	m2 = K.tile(K.expand_dims(m, 1), [1, K.shape(x)[1]])
	mask = K.cast(K.greater_equal(m2, m1), "float32")
	mask = K.tile(K.expand_dims(mask, 0), [K.shape(x)[0], 1, 1])

	lowValue = K.cast(K.less(m2, m1), "float32") * K.cast(-2**15, "float32")
	lowValue = K.tile(K.expand_dims(lowValue, 0), [K.shape(x)[0], 1, 1])

	scale = K.sqrt(K.cast(CONST.ATTENTION_UNITS, "float32"))

	return (x/scale) * mask + lowValue


def dotAttentionFunc(inputs=[], hideFuture=False):
	query = inputs[0]
	key = inputs[1]
	value = inputs[2]

	#generate alphas
	alphas = layers.Dot(axes=2)([query, key])
	alphas = layers.Lambda(sqrtScaleHideFuture if hideFuture else sqrtScaleValues)(alphas)
	alphas = layers.TimeDistributed(layers.Activation("softmax"))(alphas)

	#create weighted encoder context
	valuePerm = layers.Permute((2,1))(value)
	contextOut = layers.Dot(axes=2)([alphas, valuePerm])

	return [contextOut, alphas]


def basicAttentionStage(inputSize=CONST.NUM_LSTM_UNITS):
	query = layers.Input(batch_shape=(None,None,inputSize))
	key = layers.Input(batch_shape=(None,None,inputSize))

	queryNorm = query
	keyNorm = key
	queryNorm = layers.TimeDistributed(layers.BatchNormalization())(queryNorm)
	keyNorm = layers.TimeDistributed(layers.BatchNormalization())(keyNorm)

	#key query pair
	queryAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS, activation=CONST.DENSE_ACTIVATION))(queryNorm)
	keyAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS, activation=CONST.DENSE_ACTIVATION))(keyNorm)
	
	# attention!
	[contextOut, alphas] = dotAttentionFunc([queryAttentionIn, keyAttentionIn, keyNorm])

	attentionModel = Model(inputs=[query, key], outputs=[contextOut, alphas], name="attention_stage")
	return attentionModel


def multiHeadAttentionStage(inputSize, h=CONST.NUM_ATTENTION_HEADS, count=0, hideFuture=False, feedForward=True):
	query = layers.Input(batch_shape=(None,None,inputSize))
	key = layers.Input(batch_shape=(None,None,inputSize))

	queryNorm = query
	keyNorm = key
	queryNorm = layers.TimeDistributed(layers.BatchNormalization())(queryNorm)
	keyNorm = layers.TimeDistributed(layers.BatchNormalization())(keyNorm)

	contextList = []
	alphasList = []

	for i in range(h):
		#key query pair
		queryAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS//h, activation=CONST.DENSE_ACTIVATION))(queryNorm)
		keyAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS//h, activation=CONST.DENSE_ACTIVATION))(keyNorm)
		valueAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS//h, activation=CONST.DENSE_ACTIVATION))(keyNorm)
		
		# attention!
		[contextOut, alphas] = dotAttentionFunc([queryAttentionIn, keyAttentionIn, valueAttentionIn], hideFuture=hideFuture)

		alphasList.append(alphas)
		contextList.append(contextOut)
	
	alphas = layers.Average()(alphasList)

	contextOut = layers.Concatenate()(contextList)
	contextOut = layers.TimeDistributed(layers.BatchNormalization())(contextOut)
	contextOut = layers.TimeDistributed(layers.Dense(inputSize, activation=CONST.DENSE_ACTIVATION))(contextOut)
	contextOut = layers.Add()([queryNorm, contextOut])

	if feedForward:
		contextOut = layers.TimeDistributed(layers.BatchNormalization())(contextOut)
		contextOutFF = layers.TimeDistributed(layers.Dense(CONST.FEED_FORWARD_UNITS, activation=CONST.DENSE_ACTIVATION))(contextOut)
		contextOutFF = layers.TimeDistributed(layers.BatchNormalization())(contextOutFF)
		contextOutFF = layers.TimeDistributed(layers.Dense(inputSize, activation=CONST.DENSE_ACTIVATION))(contextOutFF)
		contextOut = layers.Add()([contextOut, contextOutFF])

	attentionModel = Model(inputs=[query, key], outputs=[contextOut, alphas], name="multihead_attention_stage_"+str(count))
	return attentionModel