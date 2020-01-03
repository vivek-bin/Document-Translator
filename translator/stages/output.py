from keras.models import Model
from keras import layers
from keras import backend as K
from keras.engine.base_layer import InputSpec

from .. import constants as CONST


class SharedOutput(layers.Dense):
	def __init__(self, units,
				 kernel_initializer='glorot_uniform',
				 tied_embedding=None,
				 **kwargs):
		self.embedding = tied_embedding.layers[0]
		units = self.embedding.input_dim
		self.vocabSize = self.embedding.output_dim

		super(SharedOutput, self).__init__(units, use_bias=False, kernel_initializer=kernel_initializer, bias_initializer=None, **kwargs)

	def build(self):
		self.kernel = K.transpose(self.embedding.weights[0])

		self.input_spec = InputSpec(min_ndim=2, axes={-1: self.vocabSize})
		self.built = True

def recurrentOutputStage(outputVocabularySize, name=""):
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
	wordOut = layers.TimeDistributed(layers.Dense(outputVocabularySize, activation="softmax"))(wordOut)

	outputStage = Model(inputs=[contextOut, decoderOut, decoderEmbedding], outputs=[wordOut], name="output"+name)
	return outputStage


def simpleOutputStage(outputVocabularySize, name=""):
	contextOut = layers.Input(batch_shape=(None,None,CONST.MODEL_BASE_UNITS))

	contextFinal = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(contextOut)
	contextFinal = layers.TimeDistributed(layers.BatchNormalization())(contextFinal)

	#word prediction
	wordOut = layers.TimeDistributed(layers.Dense(outputVocabularySize, activation="softmax"))(contextFinal)

	outputStage = Model(inputs=[contextOut], outputs=[wordOut], name="output"+name)
	return outputStage

	