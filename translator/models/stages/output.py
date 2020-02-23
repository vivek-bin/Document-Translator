from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.regularizers import l2

from ... import constants as CONST
from .normalize import LayerNormalization


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
	
	decoderOutFinal = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=l2(CONST.L2_REGULARISATION)))(decoderOut)
	contextFinal = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=l2(CONST.L2_REGULARISATION)))(contextOut)
	prevWordFinal = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=l2(CONST.L2_REGULARISATION)))(decoderEmbedding)

	#combine
	wordOut = layers.Add()([contextFinal, decoderOutFinal, prevWordFinal])
	if CONST.LAYER_NORMALIZATION:
		wordOut = LayerNormalization(**CONST.LAYER_NORMALIZATION_ARGUMENTS)(wordOut)
	wordOut = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=l2(CONST.L2_REGULARISATION)))(wordOut)
	if CONST.LAYER_NORMALIZATION:
		wordOut = LayerNormalization(**CONST.LAYER_NORMALIZATION_ARGUMENTS)(wordOut)

	#word prediction
	if CONST.SHARED_INPUT_OUTPUT_EMBEDDINGS:
		assert sharedEmbedding
		wordOut = layers.TimeDistributed(SharedOutput(sharedEmbedding, activation="softmax", bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=l2(CONST.L2_REGULARISATION)))(wordOut)
	else:
		assert outputVocabularySize
		wordOut = layers.TimeDistributed(layers.Dense(outputVocabularySize, activation="softmax", bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=l2(CONST.L2_REGULARISATION)))(wordOut)

	outputStage = Model(inputs=[contextOut, decoderOut, decoderEmbedding], outputs=[wordOut], name="output"+name)
	return outputStage


def simpleOutputStage(outputVocabularySize=None, sharedEmbedding=None, name=""):
	contextOut = layers.Input(batch_shape=(None,None,CONST.MODEL_BASE_UNITS))

	contextFinal = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=l2(CONST.L2_REGULARISATION)))(contextOut)
	if CONST.LAYER_NORMALIZATION:
		contextFinal = LayerNormalization(**CONST.LAYER_NORMALIZATION_ARGUMENTS)(contextFinal)

	#word prediction
	if CONST.SHARED_INPUT_OUTPUT_EMBEDDINGS:
		assert sharedEmbedding
		wordOut = layers.TimeDistributed(SharedOutput(sharedEmbedding, activation="softmax", bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=l2(CONST.L2_REGULARISATION)))(contextFinal)
	else:
		assert outputVocabularySize
		wordOut = layers.TimeDistributed(layers.Dense(outputVocabularySize, activation="softmax", bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=l2(CONST.L2_REGULARISATION)))(contextFinal)

	outputStage = Model(inputs=[contextOut], outputs=[wordOut], name="output"+name)
	return outputStage

	