from keras.models import Model
from keras import layers
from keras import backend as K
import numpy as np

from .. import constants as CONST

def positionalEncoding(x):
	from keras import backend as K
	
	positionEncoding = K.variable(CONST.MAX_POSITIONAL_EMBEDDING)[0:K.shape(x)[1], 0:CONST.EMBEDDING_SIZE]
	positionEncoding = K.tile(K.expand_dims(positionEncoding, 0), [K.shape(x)[0],1,1])
	#positionEncoding = K.reshape(positionEncoding,(K.shape(x)[0], K.shape(x)[1], CONST.EMBEDDING_SIZE))
	return positionEncoding


def embeddingStage(VOCABULARY_COUNT, name, addPositionalEmbedding=False):
	#word embedding
	wordInput = layers.Input(batch_shape=(None, None))
	embedding = layers.Embedding(input_dim=VOCABULARY_COUNT, output_dim=CONST.EMBEDDING_SIZE)(wordInput)

	if addPositionalEmbedding:
		positionEmbedding = layers.Lambda(positionalEncoding)(embedding)
		embedding = layers.Add()([embedding, positionEmbedding])

	#embedding = layers.TimeDistributed(layers.BatchNormalization())(embedding)
	
	#interface with the rest of the model
	embedding = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION))(embedding)
	embedding = layers.TimeDistributed(layers.BatchNormalization())(embedding)
	
	embeddingModel = Model(inputs=[wordInput], outputs=[embedding], name="embedding_"+name)
	return embeddingModel


def wordCharEmbeddingStage(VOCABULARY_COUNT, CHAR_VOCABULARY_COUNT, name, addPositionalEmbedding=False):
	#word embedding
	wordInput = layers.Input(batch_shape=(None, None))
	wordEmbedding = layers.Embedding(input_dim=VOCABULARY_COUNT, output_dim=CONST.WORD_EMBEDDING_SIZE)(wordInput)
	#char embedding
	charForwardInput = layers.Input(batch_shape=(None, None, None))
	charBackwardInput = layers.Input(batch_shape=(None, None, None))
	charInput = layers.Concatenate()([charForwardInput, charBackwardInput])
	charInput = layers.Reshape(target_shape=(-1,))(charInput)
	charEmbedding = layers.Embedding(input_dim=CHAR_VOCABULARY_COUNT, output_dim=CONST.CHAR_EMBEDDING_SIZE)(charInput)
	charEmbedding = layers.Reshape(target_shape=(-1, CONST.CHAR_INPUT_SIZE * 2 * CONST.CHAR_EMBEDDING_SIZE))(charEmbedding)
	#final input embedding
	embedding = layers.Concatenate()([wordEmbedding, charEmbedding])

	if addPositionalEmbedding:
		positionEmbedding = layers.Lambda(positionalEncoding)(embedding)
		embedding = layers.Add()([embedding, positionEmbedding])

	#embedding = layers.TimeDistributed(layers.BatchNormalization())(embedding)
	
	#interface with the rest of the model
	embedding = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION))(embedding)
	embedding = layers.TimeDistributed(layers.BatchNormalization())(embedding)
	
	embeddingModel = Model(inputs=[wordInput, charForwardInput, charBackwardInput], outputs=[embedding], name="embedding_"+name)
	return embeddingModel

