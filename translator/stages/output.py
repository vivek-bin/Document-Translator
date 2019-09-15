from keras.models import Model
from keras import layers

from .. import constants as CONST


def recurrentOutputStage(outputVocabularySize, contextSize, name=""):
	decoderEmbedding = layers.Input(batch_shape=(None,None,CONST.EMBEDDING_SIZE))
	decoderOut = layers.Input(batch_shape=(None,None,contextSize))
	contextOut = layers.Input(batch_shape=(None,None,contextSize))
	
	decoderOutFinal = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(decoderOut)
	contextFinal = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(contextOut)
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

	contextFinal = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(contextOut)
	contextFinal = layers.TimeDistributed(layers.BatchNormalization())(contextFinal)
	wordOut = layers.TimeDistributed(layers.Dense(outputVocabularySize, activation="softmax"))(contextFinal)

	outputStage = Model(inputs=[contextOut], outputs=[wordOut], name="output"+name)
	return outputStage

	