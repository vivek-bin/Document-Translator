from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import json

from .. import constants as CONST
from .stages import *


def translationLSTMAttModel(startLang, endLang):
	## get vocabulary sizes to build model
	with open(CONST.ENCODINGS+startLang+"_word.json", "r") as f:
		INPUT_VOCABULARY_COUNT = len(json.load(f))
	with open(CONST.ENCODINGS+endLang+"_word.json", "r") as f:
		OUTPUT_VOCABULARY_COUNT = len(json.load(f))
	if CONST.INCLUDE_CHAR_EMBEDDING:
		with open(CONST.ENCODINGS+startLang+"_char.json", "r") as f:
			INPUT_CHAR_VOCABULARY_COUNT = len(json.load(f))
		with open(CONST.ENCODINGS+endLang+"_char.json", "r") as f:
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
	### bi-directional layer
	if CONST.ENCODER_BIDIREC_DEPTH > 0:
		encoderCell = EncoderLSTMCell(units=CONST.MODEL_BASE_UNITS, depth=CONST.ENCODER_BIDIREC_DEPTH, use_residual_connection=CONST.RECURRENT_LAYER_RESIDUALS, use_layer_norm=CONST.LAYER_NORMALIZATION, use_peephole=CONST.PEEPHOLE, activation=CONST.LSTM_ACTIVATION, recurrent_activation=CONST.LSTM_RECURRENT_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=l2(CONST.L2_REGULARISATION), recurrent_regularizer=l2(CONST.L2_REGULARISATION))
		encoderBiLSTM = layers.Bidirectional(layers.RNN(cell=encoderCell, return_sequences=True, return_state=True))
		encoderAllOut = encoderBiLSTM(encoderOut)
		encoderOut = encoderAllOut[0]
		encoderStates.extend([layers.Add()([encoderAllOut[1+i], encoderAllOut[1+i+CONST.ENCODER_BIDIREC_DEPTH]]) for i in range(2*CONST.ENCODER_BIDIREC_DEPTH)])
		encoderOut = layers.TimeDistributed(layers.Dense(CONST.MODEL_BASE_UNITS, activation=CONST.DENSE_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=l2(CONST.L2_REGULARISATION)))(encoderOut)
	
	### uni-directional layer
	if CONST.DECODER_ENCODER_DEPTH > CONST.ENCODER_BIDIREC_DEPTH:
		encoderCell = EncoderLSTMCell(units=CONST.MODEL_BASE_UNITS, depth=CONST.DECODER_ENCODER_DEPTH-CONST.ENCODER_BIDIREC_DEPTH, use_residual_connection=CONST.RECURRENT_LAYER_RESIDUALS, use_layer_norm=CONST.LAYER_NORMALIZATION, use_peephole=CONST.PEEPHOLE, activation=CONST.LSTM_ACTIVATION, recurrent_activation=CONST.LSTM_RECURRENT_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=l2(CONST.L2_REGULARISATION), recurrent_regularizer=l2(CONST.L2_REGULARISATION))
		encoderUniLSTM = layers.RNN(cell=encoderCell, return_sequences=True, return_state=True)
		encoderAllOut = encoderUniLSTM(encoderOut)
		encoderOut = encoderAllOut[0]
		encoderStates.extend(encoderAllOut[1:])


	######DECODER PROCESSING STAGE
	alphaPlaceholder = layers.Reshape((1, -1))(encoderStates[0])
	decoderInitialStates = encoderStates + [encoderOut, alphaPlaceholder]
	decoderCell = DecoderLSTMCell(units=CONST.MODEL_BASE_UNITS, depth=CONST.DECODER_ENCODER_DEPTH, use_residual_connection=CONST.RECURRENT_LAYER_RESIDUALS, use_layer_norm=CONST.LAYER_NORMALIZATION, use_peephole=CONST.PEEPHOLE, activation=CONST.LSTM_ACTIVATION, recurrent_activation=CONST.LSTM_RECURRENT_ACTIVATION, bias_initializer=CONST.BIAS_INITIALIZER, kernel_regularizer=l2(CONST.L2_REGULARISATION), recurrent_regularizer=l2(CONST.L2_REGULARISATION))
	decoderLSTM = layers.RNN(cell=decoderCell, return_sequences=True, return_state=True)
	decoderAllOut = decoderLSTM(decoderEmbedding, initial_state=decoderInitialStates)
	decoderOut = decoderAllOut[0]
	decoderStates = decoderAllOut[1:]
	alphas = decoderStates[-1]
	
	######FINAL PREDICTION STAGE
	if CONST.SHARED_INPUT_OUTPUT_EMBEDDINGS:
		outputStage_SHARED = recurrentOutputStage(sharedEmbedding=decoderEmbedding_SHARED)
	else:
		outputStage_SHARED = recurrentOutputStage(outputVocabularySize=OUTPUT_VOCABULARY_COUNT)
	wordOut = outputStage_SHARED([decoderOut, decoderEmbedding])

	trainingModel = Model(inputs=encoderInput + decoderInput, outputs=wordOut, name="AttLSTM-"+startLang+"-to-"+endLang)
	

	################################
	#sampling model creation start
	################################
	###########
	#first step prediction model
	samplingModelInit = Model(inputs=encoderInput + decoderInput, outputs=[wordOut, alphas, encoderOut] + decoderStates, name="AttLSTMSamplingInit")

	###########
	#next steps prediction model
	preprocessedEncoder = layers.Input(batch_shape=(None, None, CONST.MODEL_BASE_UNITS))
	previousStates = [layers.Input(batch_shape=(None, CONST.MODEL_BASE_UNITS))]
	for _ in range(CONST.DECODER_ENCODER_DEPTH):
		previousStates.append(layers.Input(batch_shape=(None, CONST.MODEL_BASE_UNITS)))			#H
		previousStates.append(layers.Input(batch_shape=(None, CONST.MODEL_BASE_UNITS)))			#C

	alphaPlaceholder = layers.Reshape((1, -1))(previousStates[0])
	decoderPreviousStates = previousStates + [preprocessedEncoder, alphaPlaceholder]
	decoderAllOut = decoderLSTM(decoderEmbedding, initial_state=decoderPreviousStates)
	decoderOut = decoderAllOut[0]
	decoderStates = decoderAllOut[1:]
	alphas = decoderStates[-1]
	wordOut = outputStage_SHARED([decoderOut, decoderEmbedding])
	
	samplingModelNext = Model(inputs=[preprocessedEncoder] + previousStates + decoderInput, outputs=[wordOut, alphas] + decoderStates, name="AttLSTMSamplingNext")

	return trainingModel, (samplingModelInit, samplingModelNext)



def translationTransformerModel(startLang, endLang):
	## get vocabulary sizes to build model
	with open(CONST.ENCODINGS+startLang+"_word.json", "r") as f:
		INPUT_VOCABULARY_COUNT = len(json.load(f))
	with open(CONST.ENCODINGS+endLang+"_word.json", "r") as f:
		OUTPUT_VOCABULARY_COUNT = len(json.load(f))
	if CONST.INCLUDE_CHAR_EMBEDDING:
		with open(CONST.ENCODINGS+startLang+"_char.json", "r") as f:
			INPUT_CHAR_VOCABULARY_COUNT = len(json.load(f))
		with open(CONST.ENCODINGS+endLang+"_char.json", "r") as f:
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

	trainingModel = Model(inputs=encoderInput + decoderInput, outputs=wordOut, name="Transformer-"+startLang+"-to-"+endLang)
	


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



