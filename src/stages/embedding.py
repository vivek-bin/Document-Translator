from keras.models import Model
from keras import layers
from keras import backend as K
import numpy as np

from .. import constants as CONST

def positionalEncoding(x):
	from keras import backend as K
	from .. import constants as CONST
	
	sequenceLength = CONST.MAX_WORDS + 2 #int(x.shape[1])
	batchSize = CONST.BATCH_SIZE #int(x.shape[0])
	
	positionEncoding = CONST.MAX_POSITIONAL_EMBEDDING[None,:sequenceLength]
	return K.tile(K.variable(positionEncoding), [batchSize,1,1])



def embeddingStage(VOCABULARY_COUNT, name, addPositionalEmbedding=False):
	#word embedding
	wordInput = layers.Input(batch_shape=(None, None))
	embedding = layers.Embedding(input_dim=VOCABULARY_COUNT, output_dim=CONST.EMBEDDING_SIZE)(wordInput)

	if addPositionalEmbedding:
		positionEmbedding = layers.Lambda(positionalEncoding)(embedding)
		embedding = layers.Add()([embedding, positionEmbedding])

	embedding = layers.TimeDistributed(layers.BatchNormalization())(embedding)
	embeddingModel = Model(inputs=[wordInput], outputs=[embedding], name="embedding_"+name)
	return embeddingModel


def wordCharEmbeddingStage(VOCABULARY_COUNT, CHAR_VOCABULARY_COUNT, name, addPositionalEmbedding=False):
	#word embedding
	wordInput = layers.Input(batch_shape=(None, None))
	wordEmbedding = layers.Embedding(input_dim=VOCABULARY_COUNT, output_dim=CONST.WORD_EMBEDDING_SIZE)(wordInput)
	#char embedding
	charInput = layers.Input(batch_shape=(None, None))
	charEmbedding = layers.Embedding(input_dim=CHAR_VOCABULARY_COUNT, output_dim=CONST.CHAR_EMBEDDING_SIZE)(charInput)
	charEmbedding = layers.Reshape(target_shape=(-1, CONST.CHAR_INPUT_SIZE * 2 * CONST.CHAR_EMBEDDING_SIZE))(charEmbedding)
	#final input embedding
	embedding = layers.concatenate([wordEmbedding, charEmbedding])

	if addPositionalEmbedding:
		positionEmbedding = layers.Lambda(positionalEncoding)(embedding)
		embedding = layers.Add()([embedding, positionEmbedding])

	embedding = layers.TimeDistributed(layers.BatchNormalization())(embedding)
	embeddingModel = Model(inputs=[wordInput, charInput], outputs=[embedding], name="embedding_"+name)
	return embeddingModel

