from keras.models import Model
from keras import layers

from .. import constants as CONST


def dotAttention(count=0, hideFuture=False):
	query = layers.Input(batch_shape=(None,None,None))
	key = layers.Input(batch_shape=(None,None,None))
	value = layers.Input(batch_shape=(None,None,None))

	def sqrtScaleValues(x):
		from .. import constants as CONST
		from keras import backend as K
		scale = K.sqrt(K.cast(CONST.ATTENTION_UNITS, "float32"))
		return x/scale

	def sqrtScaleHideFuture(x):
		from .. import constants as CONST
		from keras import backend as K
		from tensorflow.linalg import LinearOperatorLowerTriangular as tril

		scale = K.sqrt(K.cast(CONST.ATTENTION_UNITS, "float32"))
		mask = K.variable(tril(K.ones_like(x)))
		return (x * mask)/scale

	#generate alphas
	alphas = layers.Dot(axes=2)([query, key])
	alphas = layers.Lambda(sqrtScaleHideFuture if hideFuture else sqrtScaleValues)(alphas)
	alphas = layers.TimeDistributed(layers.Activation("softmax"))(alphas)

	#create weighted encoder context
	valuePerm = layers.Permute((2,1))(value)
	contextOut = layers.Dot(axes=2)([alphas, valuePerm])

	attention = Model(inputs=[query, key, value], outputs=[contextOut, alphas], name="attention_head_"+str(count))
	return attention


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
	[contextOut, alphas] = dotAttention()([queryAttentionIn, keyAttentionIn, keyNorm])

	attentionModel = Model(inputs=[query, key], outputs=[contextOut, alphas], name="attention_stage")
	return attentionModel


def multiHeadAttentionStage(inputSize, h=CONST.NUM_ATTENTION_HEADS, count=0, hideFuture=False):
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
		[contextOut, alphas] = dotAttention(count=count*h+i, hideFuture=hideFuture)([queryAttentionIn, keyAttentionIn, valueAttentionIn])

		alphasList.append(alphas)
		contextList.append(contextOut)
	
	alphas = layers.Average()(alphasList)

	contextOut = layers.Concatenate()(contextList)
	contextOut = layers.TimeDistributed(layers.BatchNormalization())(contextOut)
	contextOut = layers.TimeDistributed(layers.Dense(inputSize, activation=CONST.DENSE_ACTIVATION))(contextOut)
	contextOut = layers.Add()([queryNorm, contextOut])

	attentionModel = Model(inputs=[query, key], outputs=[contextOut, alphas], name="multihead_attention_stage_"+str(count))
	return attentionModel
