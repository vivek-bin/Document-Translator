from keras.models import Model
from keras import layers
from keras import backend as K
from keras.engine.base_layer import InputSpec

from ... import constants as CONST


class SharedOutput(layers.Dense):
	def __init__(self, tied_embedding, **kwargs):
		self.embedding = next((l for l in tied_embedding.layers if l.name == "word_embedding"))
		
		units = self.embedding.input_dim
		super(SharedOutput, self).__init__(units, use_bias=False, **kwargs)

	def build(self, _):
		self.kernel = K.transpose(self.embedding.weights[0])

		self.input_spec = InputSpec(min_ndim=2, axes={-1: self.embedding.output_dim})
		self.built = True

def recurrentOutputStage(outputVocabularySize=None, sharedEmbedding=None, name=""):
	decoderEmbedding = layers.Input(batch_shape=(None,None,CONST.MODEL_BASE_UNITS))
	decoderOut = layers.Input(batch_shape=(None,None,CONST.MODEL_BASE_UNITS))
	contextOut = layers.Input(batch_shape=(None,None,CONST.MODEL_BASE_UNITS))
	
	decoderOutFinal = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION))(decoderOut)
	contextFinal = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION))(contextOut)
	prevWordFinal = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION))(decoderEmbedding)

	#combine
	wordOut = layers.Add()([contextFinal, decoderOutFinal, prevWordFinal])
	wordOut = layers.TimeDistributed(layers.BatchNormalization())(wordOut)
	wordOut = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(wordOut)
	wordOut = layers.TimeDistributed(layers.BatchNormalization())(wordOut)

	#word prediction
	if CONST.SHARED_INPUT_OUTPUT_EMBEDDINGS:
		assert sharedEmbedding
		wordOut = layers.TimeDistributed(SharedOutput(sharedEmbedding, activation="softmax"))(wordOut)
	else:
		assert outputVocabularySize
		wordOut = layers.TimeDistributed(layers.Dense(outputVocabularySize, activation="softmax"))(wordOut)

	outputStage = Model(inputs=[contextOut, decoderOut, decoderEmbedding], outputs=[wordOut], name="output"+name)
	return outputStage


def simpleOutputStage(outputVocabularySize=None, sharedEmbedding=None, name=""):
	contextOut = layers.Input(batch_shape=(None,None,CONST.MODEL_BASE_UNITS))

	contextFinal = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(contextOut)
	contextFinal = layers.TimeDistributed(layers.BatchNormalization())(contextFinal)

	#word prediction
	if CONST.SHARED_INPUT_OUTPUT_EMBEDDINGS:
		assert sharedEmbedding
		wordOut = layers.TimeDistributed(SharedOutput(sharedEmbedding, activation="softmax"))(contextFinal)
	else:
		assert outputVocabularySize
		wordOut = layers.TimeDistributed(layers.Dense(outputVocabularySize, activation="softmax"))(contextFinal)

	outputStage = Model(inputs=[contextOut], outputs=[wordOut], name="output"+name)
	return outputStage

	