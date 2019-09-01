from keras.models import Model
from keras import layers

from .. import constants as CONST


def recurrentOutputStage(outputVocabularySize, contextSize, name=""):
	decoderEmbedding = layers.Input(batch_shape=(None,None,CONST.EMBEDDING_SIZE))
	decoderOut = layers.Input(batch_shape=(None,None,contextSize))
	contextOut = layers.Input(batch_shape=(None,None,contextSize))

	contextOutNorm = contextOut
	decoderOutNorm = decoderOut
	contextOutNorm = layers.TimeDistributed(layers.BatchNormalization())(contextOutNorm)
	decoderOutNorm = layers.TimeDistributed(layers.BatchNormalization())(decoderOutNorm)
	
	
	decoderOutFinal = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(decoderOutNorm)
	contextFinal = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(contextOutNorm)
	prevWordFinal = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(decoderEmbedding)

	#combine
	wordOut = layers.Add()([contextFinal, decoderOutFinal, prevWordFinal])
	wordOut = layers.TimeDistributed(layers.BatchNormalization())(wordOut)
	wordOut = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(wordOut)
	wordOut = layers.TimeDistributed(layers.BatchNormalization())(wordOut)

	#word prediction
	wordOut = layers.TimeDistributed(layers.Dense(outputVocabularySize, activation="softmax"))(wordOut)

	outputStage = Model(inputs=[contextOut, decoderOut, decoderEmbedding], outputs=[wordOut], name="output"+name)
	return outputStage


def simpleOutputStage(outputVocabularySize, contextSize, name=""):
	contextOut = layers.Input(batch_shape=(None,None,contextSize))

	contextOutNorm = layers.TimeDistributed(layers.BatchNormalization())(contextOut)
	contextFinal = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(contextOutNorm)
	contextFinal = layers.TimeDistributed(layers.BatchNormalization())(contextFinal)
	wordOut = layers.TimeDistributed(layers.Dense(outputVocabularySize, activation="softmax"))(contextFinal)

	outputStage = Model(inputs=[contextOut], outputs=[wordOut], name="output"+name)
	return outputStage

	