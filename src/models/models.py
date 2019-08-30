from keras.models import Model
from keras import layers
import json

from .. import constants as CONST
from ..stages import *

def getEncodingSizes():
	## get vocabulary sizes to build model
	with open(CONST.ENCODING_PATH+"fr_word.json", "r") as f:
		INPUT_VOCABULARY_COUNT = len(json.load(f))
	with open(CONST.ENCODING_PATH+"en_word.json", "r") as f:
		OUTPUT_VOCABULARY_COUNT = len(json.load(f))
	with open(CONST.ENCODING_PATH+"fr_char.json", "r") as f:
		INPUT_CHAR_VOCABULARY_COUNT = len(json.load(f))
	with open(CONST.ENCODING_PATH+"en_char.json", "r") as f:
		OUTPUT_CHAR_VOCABULARY_COUNT = len(json.load(f))

	return (INPUT_VOCABULARY_COUNT, OUTPUT_VOCABULARY_COUNT, INPUT_CHAR_VOCABULARY_COUNT, OUTPUT_CHAR_VOCABULARY_COUNT)

def translationLSTMAttModel():
	(INPUT_VOCABULARY_COUNT, OUTPUT_VOCABULARY_COUNT, INPUT_CHAR_VOCABULARY_COUNT, OUTPUT_CHAR_VOCABULARY_COUNT) = getEncodingSizes()

	################################
	#training model creation start
	################################
	######ENCODER EMBEDDING
	encoderWordInput = layers.Input(batch_shape=(None, None))
	encoderCharInput = layers.Input(batch_shape=(None, None))		# forward and backwards
	encoderEmbedding_SHARED = wordCharEmbeddingStage(INPUT_VOCABULARY_COUNT, INPUT_CHAR_VOCABULARY_COUNT, name="encoder")
	encoderEmbedding = encoderEmbedding_SHARED([encoderWordInput, encoderCharInput])

	######DECODER EMBEDDING
	decoderWordInput = layers.Input(batch_shape=(None, None))
	decoderCharInput = layers.Input(batch_shape=(None, None))		# forward and backwards
	decoderEmbedding_SHARED = wordCharEmbeddingStage(OUTPUT_VOCABULARY_COUNT, OUTPUT_CHAR_VOCABULARY_COUNT, name="decoder")
	decoderEmbedding = decoderEmbedding_SHARED([decoderWordInput, decoderCharInput])

	######ENCODER PROCESSING STAGE
	encoderOut_SHARED = layers.Bidirectional(layers.LSTM(CONST.NUM_LSTM_UNITS//2, return_sequences=True, return_state=True, activation=CONST.LSTM_ACTIVATION))
	encoderOut, encoderForwardH, encoderForwardC, encoderBackwardH, encoderBackwardC = encoderOut_SHARED(encoderEmbedding)
	encoderH = layers.Concatenate()([encoderForwardH, encoderBackwardH])
	encoderC = layers.Concatenate()([encoderForwardC, encoderBackwardC])
	######DECODER PROCESSING STAGE
	decoderOut_SHARED = layers.LSTM(CONST.NUM_LSTM_UNITS, return_state=True, return_sequences=True, activation=CONST.LSTM_ACTIVATION)
	decoderOut, decoderH, decoderC = decoderOut_SHARED([decoderEmbedding, encoderH, encoderC])

	######ATTENTION STAGE
	attentionLayer_SHARED = multiHeadAttentionStage(CONST.NUM_LSTM_UNITS)
	[contextOut, alphas] = attentionLayer_SHARED([decoderOut, encoderOut])
	
	######FINAL PREDICTION STAGE
	outputStage_SHARED = outputStage(OUTPUT_VOCABULARY_COUNT, CONST.NUM_LSTM_UNITS)
	wordOut = outputStage_SHARED([contextOut, decoderOut, decoderEmbedding])

	trainingModel = Model(inputs=[encoderWordInput, encoderCharInput, decoderWordInput, decoderCharInput], outputs=wordOut)
	


	################################
	#sampling model creation start
	################################
	###########
	#first step prediction model
	samplingModelInit = Model(inputs=[encoderWordInput, encoderCharInput, decoderWordInput, decoderCharInput], outputs=[wordOut, encoderOut, decoderH, decoderC, alphas])

	###########
	#next steps prediction model
	preprocessedEncoder = layers.Input(batch_shape=(None, None, CONST.NUM_LSTM_UNITS))			#since Bi-LSTM
	previousDecoderH = layers.Input(batch_shape=(None, None, CONST.NUM_LSTM_UNITS))
	previousDecoderC = layers.Input(batch_shape=(None, None, CONST.NUM_LSTM_UNITS))

	decoderOut, decoderH, decoderC = decoderOut_SHARED([decoderEmbedding, previousDecoderH, previousDecoderC])
	[contextOut, alphas] = attentionLayer_SHARED([decoderOut, preprocessedEncoder])
	wordOut = outputStage_SHARED([contextOut, decoderOut, decoderEmbedding])
	
	samplingModelNext = Model(inputs=[preprocessedEncoder, previousDecoderH, previousDecoderC, decoderWordInput, decoderCharInput], outputs=[wordOut, preprocessedEncoder, decoderH, decoderC, alphas])

	return trainingModel, (samplingModelInit, samplingModelNext)



def translationTransformerModel():
	(INPUT_VOCABULARY_COUNT, OUTPUT_VOCABULARY_COUNT, INPUT_CHAR_VOCABULARY_COUNT, OUTPUT_CHAR_VOCABULARY_COUNT) = getEncodingSizes()

	################################
	#training model creation start
	################################
	######ENCODER STAGE
	######EMBEDDING
	encoderWordInput = layers.Input(batch_shape=(None, None))
	encoderCharInput = layers.Input(batch_shape=(None, None))		# forward and backwards
	encoderEmbedding_SHARED = wordCharEmbeddingStage(INPUT_VOCABULARY_COUNT, INPUT_CHAR_VOCABULARY_COUNT, name="encoder", addPositionalEmbedding=True)
	encoderEmbedding = encoderEmbedding_SHARED([encoderWordInput, encoderCharInput])

	######ATTENTION STAGE
	encoderContext = encoderEmbedding
	for i in range(CONST.ENCODER_ATTENTION_STAGES):
		encoderSelfAttentionLayer_SHARED = multiHeadAttentionStage(CONST.EMBEDDING_SIZE, count=i)
		[encoderContextNext, alphas] = encoderSelfAttentionLayer_SHARED([encoderContext, encoderContext])
		encoderContext = layers.Add()([encoderContextNext, encoderContext])


	######DECODER STAGE
	######EMBEDDING
	decoderWordInput = layers.Input(batch_shape=(None, None))
	decoderCharInput = layers.Input(batch_shape=(None, None))		# forward and backwards
	decoderEmbedding_SHARED = wordCharEmbeddingStage(OUTPUT_VOCABULARY_COUNT, OUTPUT_CHAR_VOCABULARY_COUNT, name="decoder", addPositionalEmbedding=True)
	decoderEmbedding = decoderEmbedding_SHARED([decoderWordInput, decoderCharInput])

	######ATTENTION STAGE
	decoderContext = decoderEmbedding
	alphasList = []
	decoderSelfAttentionLayer_SHARED = []
	decoderEncoderAttentionLayer_SHARED = []
	for i in range(CONST.DECODER_ATTENTION_STAGES):
		decoderSelfAttentionLayer_SHARED.append(multiHeadAttentionStage(CONST.EMBEDDING_SIZE, hideFuture=True, count=CONST.ENCODER_ATTENTION_STAGES + i*2))
		[decoderContextNext, alphas] = decoderSelfAttentionLayer_SHARED[-1]([decoderContext, decoderContext])
		decoderContext = layers.Add()([decoderContextNext, decoderContext])

		decoderEncoderAttentionLayer_SHARED.append(multiHeadAttentionStage(CONST.EMBEDDING_SIZE, count=CONST.ENCODER_ATTENTION_STAGES + i*2 + 1))
		[decoderContextNext, alphas] = decoderEncoderAttentionLayer_SHARED[-1]([decoderContext, encoderContext])
		decoderContext = layers.Add()([decoderContextNext, decoderContext])
		alphasList.append(alphas)
	alphas = layers.Average()(alphasList)
	
	######OUTPUT/PREDICTION STAGE
	outputStage_SHARED = simpleOutputStage(OUTPUT_VOCABULARY_COUNT, CONST.EMBEDDING_SIZE)
	wordOut = outputStage_SHARED(decoderContext)

	trainingModel = Model(inputs=[encoderWordInput, encoderCharInput, decoderWordInput, decoderCharInput], outputs=wordOut)
	


	################################
	#sampling model creation start
	################################
	###########
	#first step prediction model
	samplingModelInit = Model(inputs=[encoderWordInput, encoderCharInput, decoderWordInput, decoderCharInput], outputs=[wordOut, encoderContext, alphas])

	###########
	#next steps prediction model
	preprocessedEncoder = layers.Input(batch_shape=(None, None, CONST.EMBEDDING_SIZE))

	decoderContext = decoderEmbedding
	alphasList = []
	for i in range(CONST.DECODER_ATTENTION_STAGES):
		[decoderContextNext, alphas] = decoderSelfAttentionLayer_SHARED[i]([decoderContext, decoderContext])
		decoderContext = layers.Add()([decoderContextNext, decoderContext])

		[decoderContextNext, alphas] = decoderEncoderAttentionLayer_SHARED[i]([decoderContext, preprocessedEncoder])
		decoderContext = layers.Add()([decoderContextNext, decoderContext])
		alphasList.append(alphas)
	alphas = layers.Average()(alphasList)

	wordOut = outputStage_SHARED(decoderContext)
	
	samplingModelNext = Model(inputs=[preprocessedEncoder, decoderWordInput, decoderCharInput], outputs=[wordOut, preprocessedEncoder, alphas])

	return trainingModel, (samplingModelInit, samplingModelNext)



