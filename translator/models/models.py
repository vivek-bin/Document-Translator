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
		encoderCharForwardInput = layers.Input(batch_shape=(None, None, None))
		encoderCharBackwardInput = layers.Input(batch_shape=(None, None, None))
		encoderInput.append(encoderCharForwardInput)
		encoderInput.append(encoderCharBackwardInput)
		encoderEmbedding_SHARED = wordCharEmbeddingStage(INPUT_VOCABULARY_COUNT, INPUT_CHAR_VOCABULARY_COUNT, name="encoder")
	else:
		encoderEmbedding_SHARED = embeddingStage(INPUT_VOCABULARY_COUNT, name="encoder")
	encoderEmbedding = encoderEmbedding_SHARED(encoderInput)

	
	######DECODER EMBEDDING
	decoderWordInput = layers.Input(batch_shape=(None, None))
	decoderInput = [decoderWordInput]
	if CONST.INCLUDE_CHAR_EMBEDDING:
		decoderCharForwardInput = layers.Input(batch_shape=(None, None, None))
		decoderCharBackwardInput = layers.Input(batch_shape=(None, None, None))
		decoderInput.append(decoderCharForwardInput)
		decoderInput.append(decoderCharBackwardInput)
		decoderEmbedding_SHARED = wordCharEmbeddingStage(OUTPUT_VOCABULARY_COUNT, OUTPUT_CHAR_VOCABULARY_COUNT, name="decoder")
	else:
		decoderEmbedding_SHARED = embeddingStage(OUTPUT_VOCABULARY_COUNT, name="decoder")
	decoderEmbedding = decoderEmbedding_SHARED(decoderInput)


	######ENCODER PROCESSING STAGE
	encoderStates = []
	encoderOut = encoderEmbedding
	for i in range(CONST.DECODER_ENCODER_DEPTH):
		encoderLSTM = layers.Bidirectional(layers.LSTM(CONST.MODEL_BASE_UNITS//2, return_sequences=True, return_state=True, activation=CONST.LSTM_ACTIVATION, recurrent_activation=CONST.LSTM_RECURRENT_ACTIVATION))
		encoderBatchNorm = layers.TimeDistributed(layers.BatchNormalization())
    
		encoderOutNext, forwardH, forwardC, backwardH, backwardC = encoderLSTM(encoderOut)
		if CONST.RECURRENT_LAYER_RESIDUALS:
			encoderOut = layers.Add()([encoderOut, encoderOutNext])
		else:
			encoderOut = encoderOutNext
		encoderOut = encoderBatchNorm(encoderOut)

		encoderStates.append(layers.Concatenate()([forwardH, backwardH]))
		encoderStates.append(layers.Concatenate()([forwardC, backwardC]))

	######DECODER PROCESSING STAGE
	decoderStates = []
	decoderOut = decoderEmbedding
	decoderLSTM_SHARED = []
	decoderBatchNorm_SHARED = []
	for i in range(CONST.DECODER_ENCODER_DEPTH):
		decoderLSTM_SHARED.append(layers.LSTM(CONST.MODEL_BASE_UNITS, return_sequences=True, return_state=True, activation=CONST.LSTM_ACTIVATION, recurrent_activation=CONST.LSTM_RECURRENT_ACTIVATION))
		decoderBatchNorm_SHARED.append(layers.TimeDistributed(layers.BatchNormalization()))

		initialState = encoderStates[i*2:(i+1)*2]
		decoderOutNext, forwardH, forwardC = decoderLSTM_SHARED[i]([decoderOut] + initialState)
		if CONST.RECURRENT_LAYER_RESIDUALS:
			decoderOut = layers.Add()([decoderOut, decoderOutNext])
		else:
			decoderOut = decoderOutNext
		decoderOut = decoderBatchNorm_SHARED[i](decoderOut)

		decoderStates.append(forwardH)
		decoderStates.append(forwardC)
		

	######ATTENTION STAGE
	attentionLayer_SHARED = multiHeadAttentionStage()
	[contextOut, alphas] = attentionLayer_SHARED([decoderOut, encoderOut])
	
	######FINAL PREDICTION STAGE
	if CONST.SHARED_INPUT_OUTPUT_EMBEDDINGS:
		outputStage_SHARED = recurrentOutputStage(sharedEmbedding=decoderEmbedding_SHARED)
	else:
		outputStage_SHARED = recurrentOutputStage(outputVocabularySize=OUTPUT_VOCABULARY_COUNT)
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
	preprocessedEncoder = layers.Input(batch_shape=(None, None, CONST.MODEL_BASE_UNITS))
	previousStates = []
	for _ in range(CONST.DECODER_ENCODER_DEPTH):
		previousStates.append(layers.Input(batch_shape=(None, CONST.MODEL_BASE_UNITS)))			#H
		previousStates.append(layers.Input(batch_shape=(None, CONST.MODEL_BASE_UNITS)))			#C

	#shared decoder
	decoderStates = []
	decoderOut = decoderEmbedding
	for i in range(CONST.DECODER_ENCODER_DEPTH):
		initialState = previousStates[i*2:(i+1)*2]
		decoderOutNext, forwardH, forwardC = decoderLSTM_SHARED[i]([decoderOut] + initialState)
		if CONST.RECURRENT_LAYER_RESIDUALS:
			decoderOut = layers.Add()([decoderOut, decoderOutNext])
		else:
			decoderOut = decoderOutNext
		decoderOut = decoderBatchNorm_SHARED[i](decoderOut)

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
		encoderCharForwardInput = layers.Input(batch_shape=(None, None, None))
		encoderCharBackwardInput = layers.Input(batch_shape=(None, None, None))
		encoderInput.append(encoderCharForwardInput)
		encoderInput.append(encoderCharBackwardInput)
		encoderEmbedding_SHARED = wordCharEmbeddingStage(INPUT_VOCABULARY_COUNT, INPUT_CHAR_VOCABULARY_COUNT, name="encoder", addPositionalEmbedding=True)
	else:
		encoderEmbedding_SHARED = embeddingStage(INPUT_VOCABULARY_COUNT, name="encoder", addPositionalEmbedding=True)
	encoderEmbedding = encoderEmbedding_SHARED(encoderInput)


	######ATTENTION STAGE
	encoderContext = encoderEmbedding
	for i in range(CONST.ENCODER_ATTENTION_STAGES):
		encoderSelfAttentionLayer = multiHeadAttentionStage(count=i)
		[encoderContext, _] = encoderSelfAttentionLayer([encoderContext, encoderContext])


	######DECODER STAGE
	######EMBEDDING
	decoderWordInput = layers.Input(batch_shape=(None, None))
	decoderInput = [decoderWordInput]
	if CONST.INCLUDE_CHAR_EMBEDDING:
		decoderCharForwardInput = layers.Input(batch_shape=(None, None, None))
		decoderCharBackwardInput = layers.Input(batch_shape=(None, None, None))
		decoderInput.append(decoderCharForwardInput)
		decoderInput.append(decoderCharBackwardInput)
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
		decoderSelfAttentionLayer_SHARED.append(multiHeadAttentionStage(hideFuture=True, count=CONST.ENCODER_ATTENTION_STAGES + i*2, feedForward=False))
		[decoderContext, _] = decoderSelfAttentionLayer_SHARED[-1]([decoderContext, decoderContext])

		decoderEncoderAttentionLayer_SHARED.append(multiHeadAttentionStage(count=CONST.ENCODER_ATTENTION_STAGES + i*2 + 1))
		[decoderContext, alphas] = decoderEncoderAttentionLayer_SHARED[-1]([decoderContext, encoderContext])
		alphasList.append(alphas)
	alphas = layers.Average()(alphasList)
	
	######OUTPUT/PREDICTION STAGE
	if CONST.SHARED_INPUT_OUTPUT_EMBEDDINGS:
		outputStage_SHARED = simpleOutputStage(sharedEmbedding=decoderEmbedding_SHARED)
	else:
		outputStage_SHARED = simpleOutputStage(outputVocabularySize=OUTPUT_VOCABULARY_COUNT)
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
	preprocessedEncoder = layers.Input(batch_shape=(None, None, CONST.MODEL_BASE_UNITS))

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



