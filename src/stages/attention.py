from keras.models import Model
from keras import layers

from .. import constants as CONST


def dotAttention(count=0):
	query = layers.Input(batch_shape=(None,None,None))
	key = layers.Input(batch_shape=(None,None,None))
	value = layers.Input(batch_shape=(None,None,None))

	def sqrtScaleValues(x):
		from .. import constants as CONST
		import math
		return x/math.sqrt(float(CONST.ATTENTION_UNITS))
	
	#generate alphas
	alphas = layers.dot([query, key], axes=2)
	alphas = layers.Lambda(sqrtScaleValues)(alphas)
	alphas = layers.TimeDistributed(layers.Activation("softmax"))(alphas)

	#create weighted encoder context
	valuePerm = layers.Permute((2,1))(value)
	contextOut = layers.dot([alphas, valuePerm], axes=2)

	attention = Model(inputs=[query, key, value], outputs=[contextOut, alphas], name="attention_head_"+str(count))
	return attention


def basicAttentionStage():
	query = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS))
	key = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS))

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


def multiHeadAttentionStage(h, count=0):
	query = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS))
	key = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS))

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
		[contextOut, alphas] = dotAttention(count=count*h+i)([queryAttentionIn, keyAttentionIn, valueAttentionIn])

		alphasList.append(alphas)
		contextList.append(contextOut)
	
	alphas = layers.Average()(alphasList)

	contextOut = layers.Concatenate()(contextList)
	contextOut = layers.TimeDistributed(layers.BatchNormalization())(contextOut)
	contextOut = layers.TimeDistributed(layers.Dense(int(query.shape[-1]), activation=CONST.DENSE_ACTIVATION))(contextOut)
	contextOut = layers.Add()([queryNorm, contextOut])

	attentionModel = Model(inputs=[query, key], outputs=[contextOut, alphas], name="multihead_attention_stage_"+str(count))
	return attentionModel
