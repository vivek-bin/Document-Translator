from keras.models import Model
from keras import layers
import json

from .. import constants as CONST
from ..stages import *


def translationLSTMAttModel():
	## get vocabulary sizes to build model
	with open(CONST.ENCODINGS+"fr_word.json", "r") as f:
		INPUT_VOCABULARY_COUNT = len(json.load(f))
	with open(CONST.ENCODINGS+"en_word.json", "r") as f:
		OUTPUT_VOCABULARY_COUNT = len(json.load(f))
	if CONST.INCLUDE_CHAR_EMBEDDING:
		with open(CONST.ENCODINGS+"fr_char.json", "r") as f:
			INPUT_CHAR_VOCABULARY_COUNT = len(json.load(f))
		with open(CONST.ENCODINGS+"en_char.json", "r") as f:
			OUTPUT_CHAR_VOCABULARY_COUNT = len(json.load(f))
	

	################################
	#training model creation start
	################################
	######ENCODER EMBEDDING
	encoderWordInput = layers.Input(batch_shape=(None, None))
	encoderInput = [encoderWordInput]
	if CONST.INCLUDE_CHAR_EMBEDDING:
		encoderCharInput = layers.Input(batch_shape=(None, None))		# forward and backwards
		encoderInput.append(encoderCharInput)
		encoderEmbedding_SHARED = wordCharEmbeddingStage(INPUT_VOCABULARY_COUNT, INPUT_CHAR_VOCABULARY_COUNT, name="encoder")
	else:
		encoderEmbedding_SHARED = embeddingStage(INPUT_VOCABULARY_COUNT, name="encoder")
	encoderEmbedding = encoderEmbedding_SHARED(encoderInput)

	
	######DECODER EMBEDDING
	decoderWordInput = layers.Input(batch_shape=(None, None))
	decoderInput = [decoderWordInput]
	if CONST.INCLUDE_CHAR_EMBEDDING:
		decoderCharInput = layers.Input(batch_shape=(None, None))		# forward and backwards
		decoderInput.append(decoderCharInput)
		decoderEmbedding_SHARED = wordCharEmbeddingStage(OUTPUT_VOCABULARY_COUNT, OUTPUT_CHAR_VOCABULARY_COUNT, name="decoder")
	else:
		decoderEmbedding_SHARED = embeddingStage(OUTPUT_VOCABULARY_COUNT, name="decoder")
	decoderEmbedding = decoderEmbedding_SHARED(decoderInput)
	decoderEmbedding = layers.TimeDistributed(layers.BatchNormalization())(decoderEmbedding)


	######ENCODER PROCESSING STAGE
	encoderStates = []
	encoderOut = encoderEmbedding
	for _ in range(CONST.DECODER_ENCODER_DEPTH):
		encoderOut = layers.TimeDistributed(layers.BatchNormalization())(encoderOut)
		encoderOut_SHARED = layers.Bidirectional(layers.LSTM(CONST.NUM_LSTM_UNITS//2, return_sequences=True, return_state=True, activation=CONST.LSTM_ACTIVATION))
		encoderOut, forwardH, forwardC, backwardH, backwardC = encoderOut_SHARED(encoderOut)
		encoderStates.append(layers.Concatenate()([forwardH, backwardH]))
		encoderStates.append(layers.Concatenate()([forwardC, backwardC]))

	######DECODER PROCESSING STAGE
	decoderStates = []
	decoderOut = decoderEmbedding
	for i in range(CONST.DECODER_ENCODER_DEPTH):
		initialState = encoderStates[i*2:(i+1)*2]
		decoderOut = layers.TimeDistributed(layers.BatchNormalization())(decoderOut)
		decoderOut_SHARED = layers.LSTM(CONST.NUM_LSTM_UNITS, return_sequences=True, return_state=True, activation=CONST.LSTM_ACTIVATION)
		decoderOut, forwardH, forwardC = decoderOut_SHARED([decoderOut] + initialState)
		decoderStates.append(forwardH)
		decoderStates.append(forwardC)
		

	######ATTENTION STAGE
	attentionLayer_SHARED = multiHeadAttentionStage(CONST.NUM_LSTM_UNITS)
	[contextOut, alphas] = attentionLayer_SHARED([decoderOut, encoderOut])
	
	######FINAL PREDICTION STAGE
	outputStage_SHARED = recurrentOutputStage(OUTPUT_VOCABULARY_COUNT, CONST.NUM_LSTM_UNITS)
	wordOut = outputStage_SHARED([contextOut, decoderOut, decoderEmbedding])

	trainingModel = Model(inputs=encoderInput + decoderInput, outputs=wordOut, name="AttLSTM")
	

	################################
	#sampling model creation start
	################################
	###########
	#first step prediction model
	samplingModelInit = Model(inputs=encoderInput + decoderInput, outputs=[wordOut, alphas, encoderOut] + decoderStates, name="AttLSTMSamplingInit")

	###########
	#next steps prediction model
	preprocessedEncoder = layers.Input(batch_shape=(None, None, CONST.NUM_LSTM_UNITS))
	previousStates = []
	for _ in range(CONST.DECODER_ENCODER_DEPTH):
		previousStates.append(layers.Input(batch_shape=(None, CONST.NUM_LSTM_UNITS)))			#H
		previousStates.append(layers.Input(batch_shape=(None, CONST.NUM_LSTM_UNITS)))			#C

	#shared decoder
	decoderStates = []
	decoderOut = decoderEmbedding
	for _ in range(CONST.DECODER_ENCODER_DEPTH):
		initialState = previousStates[i*2:(i+1)*2]
		decoderOut = layers.TimeDistributed(layers.BatchNormalization())(decoderOut)
		decoderOut, forwardH, forwardC = decoderOut_SHARED([decoderOut] + initialState)
		decoderStates.append(forwardH)
		decoderStates.append(forwardC)

	[contextOut, alphas] = attentionLayer_SHARED([decoderOut, preprocessedEncoder])
	wordOut = outputStage_SHARED([contextOut, decoderOut, decoderEmbedding])
	
	samplingModelNext = Model(inputs=[preprocessedEncoder] + previousStates + decoderInput, outputs=[wordOut, alphas] + decoderStates, name="AttLSTMSamplingNext")

	return trainingModel, (samplingModelInit, samplingModelNext)



def translationTransformerModel():
	## get vocabulary sizes to build model
	with open(CONST.ENCODINGS+"fr_word.json", "r") as f:
		INPUT_VOCABULARY_COUNT = len(json.load(f))
	with open(CONST.ENCODINGS+"en_word.json", "r") as f:
		OUTPUT_VOCABULARY_COUNT = len(json.load(f))
	if CONST.INCLUDE_CHAR_EMBEDDING:
		with open(CONST.ENCODINGS+"fr_char.json", "r") as f:
			INPUT_CHAR_VOCABULARY_COUNT = len(json.load(f))
		with open(CONST.ENCODINGS+"en_char.json", "r") as f:
			OUTPUT_CHAR_VOCABULARY_COUNT = len(json.load(f))



	################################
	#training model creation start
	################################
	######ENCODER STAGE
	######EMBEDDING
	encoderWordInput = layers.Input(batch_shape=(None, None))
	encoderInput = [encoderWordInput]
	if CONST.INCLUDE_CHAR_EMBEDDING:
		encoderCharInput = layers.Input(batch_shape=(None, None))		# forward and backwards
		encoderInput.append(encoderCharInput)
		encoderEmbedding_SHARED = wordCharEmbeddingStage(INPUT_VOCABULARY_COUNT, INPUT_CHAR_VOCABULARY_COUNT, name="encoder", addPositionalEmbedding=True)
	else:
		encoderEmbedding_SHARED = embeddingStage(INPUT_VOCABULARY_COUNT, name="encoder", addPositionalEmbedding=True)
	encoderEmbedding = encoderEmbedding_SHARED(encoderInput)


	######ATTENTION STAGE
	encoderContext = encoderEmbedding
	for i in range(CONST.ENCODER_ATTENTION_STAGES):
		encoderSelfAttentionLayer_SHARED = multiHeadAttentionStage(CONST.EMBEDDING_SIZE, count=i)
		[encoderContext, alphas] = encoderSelfAttentionLayer_SHARED([encoderContext, encoderContext])


	######DECODER STAGE
	######EMBEDDING
	decoderWordInput = layers.Input(batch_shape=(None, None))
	decoderInput = [decoderWordInput]
	if CONST.INCLUDE_CHAR_EMBEDDING:
		decoderCharInput = layers.Input(batch_shape=(None, None))		# forward and backwards
		decoderInput.append(decoderCharInput)
		decoderEmbedding_SHARED = wordCharEmbeddingStage(OUTPUT_VOCABULARY_COUNT, OUTPUT_CHAR_VOCABULARY_COUNT, name="decoder", addPositionalEmbedding=True)
	else:
		decoderEmbedding_SHARED = embeddingStage(OUTPUT_VOCABULARY_COUNT, name="decoder", addPositionalEmbedding=True)
	decoderEmbedding = decoderEmbedding_SHARED(decoderInput)


	######ATTENTION STAGE
	decoderContext = decoderEmbedding
	alphasList = []
	decoderSelfAttentionLayer_SHARED = []
	decoderEncoderAttentionLayer_SHARED = []
	for i in range(CONST.DECODER_ATTENTION_STAGES):
		decoderSelfAttentionLayer_SHARED.append(multiHeadAttentionStage(CONST.EMBEDDING_SIZE, hideFuture=True, count=CONST.ENCODER_ATTENTION_STAGES + i*2, feedForward=False))
		[decoderContext, _] = decoderSelfAttentionLayer_SHARED[-1]([decoderContext, decoderContext])

		decoderEncoderAttentionLayer_SHARED.append(multiHeadAttentionStage(CONST.EMBEDDING_SIZE, count=CONST.ENCODER_ATTENTION_STAGES + i*2 + 1))
		[decoderContext, alphas] = decoderEncoderAttentionLayer_SHARED[-1]([decoderContext, encoderContext])
		alphasList.append(alphas)
	alphas = layers.Average()(alphasList)
	
	######OUTPUT/PREDICTION STAGE
	outputStage_SHARED = simpleOutputStage(OUTPUT_VOCABULARY_COUNT, CONST.EMBEDDING_SIZE)
	wordOut = outputStage_SHARED(decoderContext)

	trainingModel = Model(inputs=encoderInput + decoderInput, outputs=wordOut, name="Transformer")
	


	################################
	#sampling model creation start
	################################
	###########
	#first step prediction model
	samplingModelInit = Model(inputs=encoderInput + decoderInput, outputs=[wordOut, alphas, encoderContext], name="TransformerSamplingInit")

	###########
	#next steps prediction model
	preprocessedEncoder = layers.Input(batch_shape=(None, None, CONST.EMBEDDING_SIZE))

	decoderContext = decoderEmbedding
	alphasList = []
	for i in range(CONST.DECODER_ATTENTION_STAGES):
		[decoderContext, _] = decoderSelfAttentionLayer_SHARED[i]([decoderContext, decoderContext])
		[decoderContext, alphas] = decoderEncoderAttentionLayer_SHARED[i]([decoderContext, preprocessedEncoder])
		alphasList.append(alphas)
	alphas = layers.Average()(alphasList)

	wordOut = outputStage_SHARED(decoderContext)
	
	samplingModelNext = Model(inputs=[preprocessedEncoder] + decoderInput, outputs=[wordOut, alphas], name="TransformerSamplingNext")

	return trainingModel, (samplingModelInit, samplingModelNext)



