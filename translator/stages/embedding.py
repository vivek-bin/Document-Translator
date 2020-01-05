from keras.models import Model
from keras import layers
from keras import backend as K
import numpy as np

from .. import constants as CONST

def addPositionalEncoding(x):
	from keras import backend as K
	return x + K.variable(CONST.MAX_POSITIONAL_EMBEDDING)[0:K.shape(x)[1], 0:K.shape(x)[2]]


def embeddingStage(VOCABULARY_COUNT, name, addPositionalEmbedding=False):
	#word embedding
	wordInput = layers.Input(batch_shape=(None, None))
	embedding = layers.Embedding(input_dim=VOCABULARY_COUNT, output_dim=CONST.EMBEDDING_SIZE, name="word_embedding")(wordInput)

	#interface with the rest of the model
	embedding = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION))(embedding)
	embedding = layers.TimeDistributed(layers.BatchNormalization())(embedding)

	if addPositionalEmbedding:
		embedding = layers.Lambda(addPositionalEncoding)(embedding)
		embedding = layers.TimeDistributed(layers.BatchNormalization())(embedding)
	
	embeddingModel = Model(inputs=[wordInput], outputs=[embedding], name="embedding_"+name)
	return embeddingModel


def wordCharEmbeddingStage(VOCABULARY_COUNT, CHAR_VOCABULARY_COUNT, name, addPositionalEmbedding=False):
	#word embedding
	wordInput = layers.Input(batch_shape=(None, None))
	wordEmbedding = layers.Embedding(input_dim=VOCABULARY_COUNT, output_dim=CONST.EMBEDDING_SIZE, name="word_embedding")(wordInput)
	#char embedding
	charForwardInput = layers.Input(batch_shape=(None, None, None))
	charBackwardInput = layers.Input(batch_shape=(None, None, None))
	charInput = layers.Concatenate()([charForwardInput, charBackwardInput])
	charInput = layers.Reshape(target_shape=(-1,))(charInput)
	charEmbedding = layers.Embedding(input_dim=CHAR_VOCABULARY_COUNT, output_dim=CONST.CHAR_EMBEDDING_SIZE, name="char_embedding")(charInput)
	charEmbedding = layers.Reshape(target_shape=(-1, CONST.CHAR_INPUT_SIZE * 2 * CONST.CHAR_EMBEDDING_SIZE))(charEmbedding)
	#final input embedding
	embedding = layers.Concatenate()([wordEmbedding, charEmbedding])

	#interface with the rest of the model
	embedding = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION))(embedding)
	embedding = layers.TimeDistributed(layers.BatchNormalization())(embedding)

	if addPositionalEmbedding:
		embedding = layers.Lambda(addPositionalEncoding)(embedding)
		embedding = layers.TimeDistributed(layers.BatchNormalization())(embedding)
	
	embeddingModel = Model(inputs=[wordInput, charForwardInput, charBackwardInput], outputs=[embedding], name="embedding_"+name)
	return embeddingModel

