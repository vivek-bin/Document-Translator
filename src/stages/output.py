from keras.models import Model
from keras import layers

from .. import constants as CONST


def outputStage(OUTPUT_VOCABULARY_COUNT):
	decoderEmbedding = layers.Input(batch_shape=(None,None,CONST.WORD_EMBEDDING_SIZE + CONST.CHAR_EMBEDDING_SIZE*CONST.CHAR_INPUT_SIZE*2))
	decoderOut = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS))
	contextOut = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS))

	contextOutNorm = contextOut
	decoderOutNorm = decoderOut
	contextOutNorm = layers.TimeDistributed(layers.BatchNormalization())(contextOutNorm)
	decoderOutNorm = layers.TimeDistributed(layers.BatchNormalization())(decoderOutNorm)
	
	
	decoderOutFinal = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(decoderOutNorm)
	contextFinal = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(contextOutNorm)
	prevWordFinal = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(decoderEmbedding)

	#combine
	wordOut = layers.Add()([contextFinal, decoderOutFinal, prevWordFinal])
	wordOut = layers.TimeDistributed(layers.BatchNormalization())(wordOut)
	wordOut = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(wordOut)
	wordOut = layers.TimeDistributed(layers.BatchNormalization())(wordOut)

	#word prediction
	wordOut = layers.TimeDistributed(layers.Dense(OUTPUT_VOCABULARY_COUNT, activation="softmax"))(wordOut)

	outputStage = Model(inputs=[contextOut, decoderOut, decoderEmbedding], outputs=[wordOut], name="output")
	return outputStage
