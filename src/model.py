import numpy as np
from keras.models import load_model
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
import json
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import os

import constants as CONST


def loadEncodedData(fileName):
	data = np.load(CONST.PROCESSED_DATA + fileName + ".npz")
	wordData = data["encoded"]
	charForwardData = data["charForwardEncoded"]
	charBackwardData = data["charBackwardEncoded"]
	data = None

	wordData = wordData[:CONST.DATA_COUNT].copy()
	charForwardData = charForwardData[:CONST.DATA_COUNT].copy()
	charBackwardData = charBackwardData[:CONST.DATA_COUNT].copy()

	charData = np.concatenate((charForwardData, charBackwardData), axis=2)
	charData = np.reshape(charData, (charData.shape[0], -1))

	return wordData, charData
	

def getFrToEngData():
	fr = loadEncodedData("frEncodedData")
	en = loadEncodedData("enEncodedData")

	inputData = fr + en
	outputData = en[0]
	outputData = np.pad(outputData,((0,0),(0,1)), mode='constant')[:,1:]
	outputData = [np.expand_dims(outputData,axis=-1)]					#for sparse categorical

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
	######ENCODER EMBEDDING
	encoderWordInput = layers.Input(batch_shape=(None, None))
	encoderCharInput = layers.Input(batch_shape=(None, None))		# forward and backwards
	encoderEmbedding_SHARED = embeddingStage(INPUT_VOCABULARY_COUNT, INPUT_CHAR_VOCABULARY_COUNT)
	encoderEmbedding = encoderEmbedding_SHARED([encoderWordInput, encoderCharInput])

	######DECODER EMBEDDING
	decoderWordInput = layers.Input(batch_shape=(None, None))
	decoderCharInput = layers.Input(batch_shape=(None, None))		# forward and backwards
	decoderEmbedding_SHARED = embeddingStage(OUTPUT_VOCABULARY_COUNT, OUTPUT_CHAR_VOCABULARY_COUNT)
	decoderEmbedding = decoderEmbedding_SHARED([decoderWordInput, decoderCharInput])

	######ENCODER PROCESSING STAGE
	encoderOut_SHARED = layers.Bidirectional(layers.LSTM(CONST.NUM_LSTM_UNITS, return_sequences=True, return_state=True, activation=CONST.LSTM_ACTIVATION))
	encoderOut, encoderForwardH, encoderForwardC, _, _ = encoderOut_SHARED(encoderEmbedding)
	######DECODER PROCESSING STAGE
	decoderOut_SHARED = layers.LSTM(CONST.NUM_LSTM_UNITS, return_state=True, return_sequences=True, activation=CONST.LSTM_ACTIVATION)
	decoderOut, decoderH, decoderC = decoderOut_SHARED([decoderEmbedding, encoderForwardH, encoderForwardC])

	######ATTENTION STAGE
	attentionLayer_SHARED = multiHeadAttentionStage(CONST.NUM_ATTENTION_HEADS)
	[contextOut, alphas] = attentionLayer_SHARED([decoderOut, encoderOut])
	
	######FINAL PREDICTION STAGE
	outputStage_SHARED = outputStage(OUTPUT_VOCABULARY_COUNT)
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
	preprocessedEncoder = layers.Input(batch_shape=(None, None, CONST.NUM_LSTM_UNITS*2))			#since Bi-LSTM
	previousDecoderH = layers.Input(batch_shape=(None, None, CONST.NUM_LSTM_UNITS))
	previousDecoderC = layers.Input(batch_shape=(None, None, CONST.NUM_LSTM_UNITS))

	decoderOut, decoderH, decoderC = decoderOut_SHARED([decoderEmbedding, previousDecoderH, previousDecoderC])
	[contextOut, alphas] = attentionLayer_SHARED([decoderOut, preprocessedEncoder])
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
	embedding = layers.TimeDistributed(layers.BatchNormalization())(embedding)

	embeddingModel = Model(inputs=[wordInput, charInput], outputs=[embedding])
	return embeddingModel


def attentionStage():
	query = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS))
	key = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS*2))

	queryNorm = query
	keyNorm = key
	queryNorm = layers.TimeDistributed(layers.BatchNormalization())(queryNorm)
	keyNorm = layers.TimeDistributed(layers.BatchNormalization())(keyNorm)

	def sqrtScaleValues(x):
		import constants as CONST
		import math
		return x/math.sqrt(float(CONST.ATTENTION_UNITS))

	
	#key query pair
	queryAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS, activation=CONST.DENSE_ACTIVATION))(queryNorm)
	keyAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS, activation=CONST.DENSE_ACTIVATION))(keyNorm)
	

	#generate alphas
	alphas = layers.dot([queryAttentionIn, keyAttentionIn], axes=2)
	alphas = layers.Lambda(sqrtScaleValues)(alphas)
	alphas = layers.TimeDistributed(layers.Activation("softmax"))(alphas)

	#create weighted encoder context
	permKeyNorm = layers.Permute((2,1))(keyNorm)
	contextOut = layers.dot([alphas, permKeyNorm], axes=2)

	attentionModel = Model(inputs=[query, key], outputs=[contextOut, alphas], name="attention")
	return attentionModel


def multiHeadAttentionStage(h):
	query = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS))
	key = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS*2))

	queryNorm = query
	keyNorm = key
	queryNorm = layers.TimeDistributed(layers.BatchNormalization())(queryNorm)
	keyNorm = layers.TimeDistributed(layers.BatchNormalization())(keyNorm)

	# sqrtScale = K.sqrt(K.cast(CONST.ATTENTION_UNITS//h, K.floatx()))
	def sqrtScaleValues(x):
		import constants as CONST
		import math
		return x/math.sqrt(float(CONST.ATTENTION_UNITS//h))

	contextList = []
	alphasList = []

	for _ in range(h):
		#key query pair
		queryAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS//h, activation=CONST.DENSE_ACTIVATION))(queryNorm)
		keyAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS//h, activation=CONST.DENSE_ACTIVATION))(keyNorm)
		valueAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS//h, activation=CONST.DENSE_ACTIVATION))(keyNorm)
		
		#generate alphas
		alphas = layers.dot([queryAttentionIn, keyAttentionIn], axes=2)
		alphas = layers.Lambda(sqrtScaleValues)(alphas)
		alphas = layers.TimeDistributed(layers.Activation("softmax"))(alphas)

		#create weighted encoder context
		valueAttentionIn = layers.Permute((2,1))(valueAttentionIn)
		contextOut = layers.dot([alphas, valueAttentionIn], axes=2)

		alphasList.append(alphas)
		contextList.append(contextOut)
	
	alphas = layers.Average()(alphasList)

	contextOut = layers.Concatenate()(contextList)
	contextOut = layers.TimeDistributed(layers.BatchNormalization())(contextOut)
	contextOut = layers.TimeDistributed(layers.Dense(int(query.shape[-1]), activation=CONST.DENSE_ACTIVATION))(contextOut)
	contextOut = layers.Add()([queryNorm, contextOut])

	attentionModel = Model(inputs=[query, key], outputs=[contextOut, alphas], name="multi_attention")
	return attentionModel


def outputStage(OUTPUT_VOCABULARY_COUNT):
	decoderEmbedding = layers.Input(batch_shape=(None,None,CONST.WORD_EMBEDDING_SIZE + CONST.CHAR_EMBEDDING_SIZE*CONST.CHAR_INPUT_SIZE*2))
	decoderOut = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS))
	contextOut = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS))

	contextOutNorm = contextOut
	decoderOutNorm = decoderOut
	contextOutNorm = layers.TimeDistributed(layers.BatchNormalization())(contextOutNorm)
	decoderOutNorm = layers.TimeDistributed(layers.BatchNormalization())(decoderOutNorm)
	
	
	decoderOutFinal = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(decoderOutNorm)
	contextFinal = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(contextOutNorm)
	prevWordFinal = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(decoderEmbedding)

	#combine
	wordOut = layers.Add()([contextFinal, decoderOutFinal, prevWordFinal])
	wordOut = layers.TimeDistributed(layers.BatchNormalization())(wordOut)
	wordOut = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE, activation=CONST.DENSE_ACTIVATION))(wordOut)
	wordOut = layers.TimeDistributed(layers.BatchNormalization())(wordOut)

	#word prediction
	wordOut = layers.TimeDistributed(layers.Dense(OUTPUT_VOCABULARY_COUNT, activation="softmax"))(wordOut)

	outputStage = Model(inputs=[contextOut, decoderOut, decoderEmbedding], outputs=[wordOut], name="output")
	return outputStage


def saveModels(trainingModel, modelName, samplingModels=False):
	# serialize model to JSON
	with open(CONST.MODEL_PATH + modelName + "_train.json", "w") as json_file:
		json_file.write(trainingModel.to_json())
	if samplingModels:
		with open(CONST.MODEL_PATH + modelName + "_sampInit.json", "w") as json_file:
			json_file.write(samplingModels[0].to_json())
		with open(CONST.MODEL_PATH + modelName + "_sampNext.json", "w") as json_file:
			json_file.write(samplingModels[1].to_json())
	
	print("Saved model structure")
	

def evaluateModel(model, xTest, yTest):
	scores = model.evaluate(xTest, yTest, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def getLastCheckpoint():
	m = [x for x in os.listdir(CONST.MODEL_PATH) if x.startswith(CONST.MODEL_CHECKPOINT_NAME_START) and x.endswith(CONST.MODEL_CHECKPOINT_NAME_END)]
	
	return sorted(m)[-1] if m else False


def loadModel():
	#get model
	trainingModel, samplingModels = translationLSTMAttModel()
	trainingModel.compile(optimizer=Adam(lr=CONST.LEARNING_RATE, decay=CONST.LEARNING_RATE_DECAY), loss=CONST.LOSS_FUNCTION, metrics=[CONST.EVALUATION_METRIC])
	trainingModel.summary()

	#load checkpoint if available
	checkPointName = getLastCheckpoint()
	if checkPointName:
		trainingModel.load_weights(CONST.MODEL_PATH + checkPointName)
		samplingModels[0].load_weights(CONST.MODEL_PATH + checkPointName)
		samplingModels[1].load_weights(CONST.MODEL_PATH + checkPointName, by_name=True)

		tempModel = load_model(CONST.MODEL_PATH + checkPointName)
		trainingModel._make_train_function()
		weight_values = K.batch_get_value(getattr(tempModel.optimizer, 'weights'))
		trainingModel.optimizer.set_weights(weight_values)


	return trainingModel, samplingModels


def trainModel():
	#get model
	trainingModel, _ = loadModel()

	# load all data
	(xTrain, yTrain), (_, _) = getFrToEngData()

	# start training
	initialEpoch = int(getLastCheckpoint()[len(CONST.MODEL_CHECKPOINT_NAME_START):][:4]) if getLastCheckpoint() else 0
	callbacks = []
	callbacks.append(ModelCheckpoint(CONST.MODEL_PATH + CONST.MODEL_CHECKPOINT_NAME, monitor=CONST.EVALUATION_METRIC, mode='max',save_best_only=True))
	_ = trainingModel.fit(x=xTrain, y=yTrain, epochs=CONST.NUM_EPOCHS, batch_size=CONST.BATCH_SIZE, validation_split=CONST.VALIDATION_SPLIT, callbacks=callbacks, initial_epoch=initialEpoch)

	trainingModel.save(CONST.MODEL_PATH + "AttLSTMTrained.h5")



def main():
	trainModel()

if __name__ == "__main__":
	main()


