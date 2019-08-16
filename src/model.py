import numpy as np
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop
import json
import keras.backend as K

import prepData as PD
import constants as CONST


def getFrToEngData():
	fr, en = PD.loadEncodedData()
	inputData = fr + en
	outputData = [en[0]]

	trainIn = [x[:CONST.TRAIN_SPLIT] for x in inputData]
	testIn = [x[CONST.TRAIN_SPLIT:] for x in inputData]
	
	trainOut = [x[:CONST.TRAIN_SPLIT] for x in outputData]
	testOut = [x[CONST.TRAIN_SPLIT:] for x in outputData]

	return (trainIn, trainOut), (testIn, testOut)


def translationLSTMAttModel():
	## get vocabulary sizes to build model
	with open(CONST.ENCODING_PATH+"fr_word.json", "r") as f:
		INPUT_VOCABULARY_COUNT = len(json.load(f))
	with open(CONST.ENCODING_PATH+"en_word.json", "r") as f:
		OUTPUT_VOCABULARY_COUNT = len(json.load(f))
	with open(CONST.ENCODING_PATH+"fr_char.json", "r") as f:
		INPUT_CHAR_VOCABULARY_COUNT = len(json.load(f))
	with open(CONST.ENCODING_PATH+"en_char.json", "r") as f:
		OUTPUT_CHAR_VOCABULARY_COUNT = len(json.load(f))


	################################
	#training model creation start
	################################
	# ######ENCODER EMBEDDING
	# encoderWordInput = layers.Input(batch_shape=(None,CONST.INPUT_SEQUENCE_LENGTH))
	# encoderCharInput = layers.Input(batch_shape=(None,CONST.INPUT_SEQUENCE_LENGTH * CONST.CHAR_INPUT_SIZE * 2))		# forward and backwards
	encoderWordInput = layers.Input(batch_shape=(None, None))
	encoderCharInput = layers.Input(batch_shape=(None, None))		# forward and backwards
	encoderEmbedding_SHARED = embeddingStage(INPUT_VOCABULARY_COUNT, INPUT_CHAR_VOCABULARY_COUNT)
	encoderEmbedding = encoderEmbedding_SHARED([encoderWordInput, encoderCharInput])

	# ######DECODER EMBEDDING
	# decoderWordInput = layers.Input(batch_shape=(None,CONST.OUTPUT_SEQUENCE_LENGTH))
	# decoderCharInput = layers.Input(batch_shape=(None,CONST.OUTPUT_SEQUENCE_LENGTH * CONST.CHAR_INPUT_SIZE * 2))		# forward and backwards
	decoderWordInput = layers.Input(batch_shape=(None, None))
	decoderCharInput = layers.Input(batch_shape=(None, None))		# forward and backwards
	decoderEmbedding = embeddingStage(OUTPUT_VOCABULARY_COUNT, OUTPUT_CHAR_VOCABULARY_COUNT)([decoderWordInput, decoderCharInput])

	######ENCODER PROCESSING STAGE
	encoderOut, encoderForwardH, encoderForwardC, _, _ = layers.Bidirectional(layers.LSTM(CONST.NUM_LSTM_UNITS, return_sequences=True, return_state=True))(encoderEmbedding)

	######ATTENTION STAGE
	attentionLayer_SHARED = attentionStage()
	[contextOut, decoderOut, decoderH, decoderC, alphas] = attentionLayer_SHARED([encoderOut, encoderForwardH, encoderForwardC, decoderEmbedding])
	
	######FINAL PREDICTION STAGE
	outputStage_SHARED = outputStage(OUTPUT_VOCABULARY_COUNT)
	wordOut = outputStage_SHARED([contextOut, decoderOut, decoderEmbedding])

	trainingModel = Model(inputs=[encoderWordInput, encoderCharInput, decoderWordInput, decoderCharInput], outputs=wordOut)
	trainingModel.compile(optimizer=RMSprop(lr=8e-4), loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])
	


	################################
	#sampling model creation start
	################################
	###########
	#first step prediction model
	samplingModelInit = Model(inputs=[encoderWordInput, encoderCharInput, decoderWordInput, decoderCharInput], outputs=[wordOut, encoderOut,  decoderH, decoderC, alphas])

	###########
	#next steps prediction model
	preprocessedEncoder = layers.Input(batch_shape=(None, None, CONST.NUM_LSTM_UNITS*2))			#since Bi-LSTM
	previousDecoderH = layers.Input(batch_shape=(None, None, CONST.NUM_LSTM_UNITS))
	previousDecoderC = layers.Input(batch_shape=(None, None, CONST.NUM_LSTM_UNITS))

	[contextOut, decoderOut, decoderH, decoderC, alphas] = attentionLayer_SHARED([preprocessedEncoder, previousDecoderH, previousDecoderC, decoderEmbedding])	
	wordOut = outputStage_SHARED([contextOut, decoderOut, decoderEmbedding])
	
	samplingModelNext = Model(inputs=[preprocessedEncoder, previousDecoderH, previousDecoderC, decoderWordInput, decoderCharInput], outputs=[wordOut, preprocessedEncoder, decoderH, decoderC, alphas])


	return trainingModel, (samplingModelInit, samplingModelNext)
	


def embeddingStage(VOCABULARY_COUNT, CHAR_VOCABULARY_COUNT):
	#word embedding
	wordInput = layers.Input(batch_shape=(None, None))
	wordEmbedding = layers.Embedding(input_dim=VOCABULARY_COUNT, output_dim=CONST.WORD_EMBEDDING_SIZE)(wordInput)
	#char embedding
	charInput = layers.Input(batch_shape=(None, None))
	charEmbedding = layers.Embedding(input_dim=CHAR_VOCABULARY_COUNT, output_dim=CONST.CHAR_EMBEDDING_SIZE)(charInput)
	charEmbedding = layers.Reshape(target_shape=(-1, CONST.CHAR_INPUT_SIZE * 2 * CONST.CHAR_EMBEDDING_SIZE))(charEmbedding)
	#final input embedding
	embedding = layers.concatenate([wordEmbedding, charEmbedding])

	embeddingModel = Model(inputs=[wordInput, charInput], outputs=[embedding])
	return embeddingModel


def attentionStage():
	decoderEmbedding = layers.Input(batch_shape=(None,None,CONST.WORD_EMBEDDING_SIZE + CONST.CHAR_EMBEDDING_SIZE*CONST.CHAR_INPUT_SIZE*2))
	encoderOut = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS*2))
	decoderInitialH = layers.Input(batch_shape=(None,CONST.NUM_LSTM_UNITS))
	decoderInitialC = layers.Input(batch_shape=(None,CONST.NUM_LSTM_UNITS))

	#build decoder context
	decoderOut, decoderH, decoderC = layers.LSTM(CONST.NUM_LSTM_UNITS, return_state=True, return_sequences=True)(decoderEmbedding, initial_state=[decoderInitialH, decoderInitialC])
	
	#key query pair
	decoderAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS))(decoderOut)
	encoderAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS))(encoderOut)
	
	#generate alphas
	alphas = layers.dot([decoderAttentionIn, encoderAttentionIn],axes=2)
	alphas = layers.TimeDistributed(layers.Activation("softmax"))(alphas)

	#create weighted encoder context
	permEncoderOut = layers.Permute((2,1))(encoderOut)
	contextOut = layers.dot([alphas, permEncoderOut], axes=2)

	attentionModel = Model(inputs=[encoderOut, decoderInitialH, decoderInitialC, decoderEmbedding], outputs=[contextOut, decoderOut, decoderH, decoderC, alphas], name="attention")
	return attentionModel


def outputStage(OUTPUT_VOCABULARY_COUNT):
	decoderEmbedding = layers.Input(batch_shape=(None,None,CONST.WORD_EMBEDDING_SIZE + CONST.CHAR_EMBEDDING_SIZE*CONST.CHAR_INPUT_SIZE*2))
	decoderOut = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS))
	contextOut = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS*2))

	#prepare different inputs for prediction
	decoderOutFinal = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE))(decoderOut)
	contextFinal = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE))(contextOut)
	prevWordFinal = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE))(decoderEmbedding)

	#combine
	wordOut = layers.Add()([contextFinal, decoderOutFinal, prevWordFinal])
	wordOut = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE))(wordOut)

	#word prediction
	wordOut = layers.TimeDistributed(layers.Dense(OUTPUT_VOCABULARY_COUNT, activation="softmax"))(wordOut)

	outputStage = Model(inputs=[contextOut, decoderOut, decoderEmbedding], outputs=[wordOut], name="output")
	return outputStage


def saveModels(trainModel, samplingModels, modelName):
	# serialize model to JSON
	with open(CONST.MODEL_PATH + modelName + "_train.json", "w") as json_file:
		json_file.write(trainModel.to_json())
	with open(CONST.MODEL_PATH + modelName + "_sampInit.json", "w") as json_file:
		json_file.write(samplingModels[0].to_json())
	with open(CONST.MODEL_PATH + modelName + "_sampNext.json", "w") as json_file:
		json_file.write(samplingModels[1].to_json())
	
	# serialize weights to HDF5
	trainModel.save_weights(CONST.MODEL_PATH + modelName + ".h5")
	print("Saved model to disk")
	

if __name__ == "__main__":
	(xTrain, yTrain), (xTest, yTest) = getFrToEngData()

	trainingModel, samplingModels = translationLSTMAttModel()

	saveModels(trainingModel, samplingModels, "AttLSTM")

	trainingModel.summary()


	history = trainingModel.fit(x=xTrain, y=yTrain, epochs=50, batch_size=128, validation_split=0.2)

	saveModels(trainingModel, samplingModels, "AttLSTMTrained")

	# scores = trainingModel.evaluate(xTest, yTest, verbose=0)
	# print("%s: %.2f%%" % (trainingModel.metrics_names[1], scores[1]*100))


