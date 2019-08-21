import numpy as np
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop
import json
import keras.backend as K
from keras.callbacks import ModelCheckpoint

import prepData as PD
import constants as CONST


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
	encoderOut_SHARED = layers.Bidirectional(layers.LSTM(CONST.NUM_LSTM_UNITS, return_sequences=True, return_state=True))
	encoderOut, encoderForwardH, encoderForwardC, _, _ = encoderOut_SHARED(encoderEmbedding)
	######DECODER PROCESSING STAGE
	decoderOut_SHARED = layers.LSTM(CONST.NUM_LSTM_UNITS, return_state=True, return_sequences=True)
	decoderOut, decoderH, decoderC = decoderOut_SHARED(decoderEmbedding, initial_state=[encoderForwardH, encoderForwardC])

	######ATTENTION STAGE
	attentionLayer_SHARED = attentionStage()
	[contextOut, alphas] = attentionLayer_SHARED([encoderOut, decoderOut])
	
	######FINAL PREDICTION STAGE
	outputStage_SHARED = outputStage(OUTPUT_VOCABULARY_COUNT)
	wordOut = outputStage_SHARED([contextOut, decoderOut, decoderEmbedding])

	trainingModel = Model(inputs=[encoderWordInput, encoderCharInput, decoderWordInput, decoderCharInput], outputs=wordOut)
	


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

	decoderOut, decoderH, decoderC = decoderOut_SHARED(decoderEmbedding, initial_state=[previousDecoderH, previousDecoderC])
	[contextOut, alphas] = attentionLayer_SHARED([preprocessedEncoder, decoderOut])
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
	#embedding = layers.BatchNormalization()(embedding)

	embeddingModel = Model(inputs=[wordInput, charInput], outputs=[embedding])
	return embeddingModel


def attentionStage():
	encoderOut = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS*2))
	decoderOut = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS))

	encoderOutNorm = encoderOut
	decoderOutNorm = decoderOut
	#encoderOutNorm = layers.BatchNormalization()(encoderOutNorm)
	#decoderOutNorm = layers.BatchNormalization()(decoderOutNorm)
	#key query pair
	decoderAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS))(decoderOutNorm)
	encoderAttentionIn = layers.TimeDistributed(layers.Dense(CONST.ATTENTION_UNITS))(encoderOutNorm)
	
	#generate alphas
	alphas = layers.dot([decoderAttentionIn, encoderAttentionIn],axes=2)
	alphas = layers.TimeDistributed(layers.Activation("softmax"))(alphas)

	#create weighted encoder context
	permEncoderOut = layers.Permute((2,1))(encoderOutNorm)
	contextOut = layers.dot([alphas, permEncoderOut], axes=2)

	attentionModel = Model(inputs=[encoderOut, decoderOut], outputs=[contextOut, alphas], name="attention")
	return attentionModel


def outputStage(OUTPUT_VOCABULARY_COUNT):
	decoderEmbedding = layers.Input(batch_shape=(None,None,CONST.WORD_EMBEDDING_SIZE + CONST.CHAR_EMBEDDING_SIZE*CONST.CHAR_INPUT_SIZE*2))
	decoderOut = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS))
	contextOut = layers.Input(batch_shape=(None,None,CONST.NUM_LSTM_UNITS*2))

	contextOutNorm = contextOut
	decoderOutNorm = decoderOut
	#contextOutNorm = layers.BatchNormalization()(contextOutNorm)
	#decoderOutNorm = layers.BatchNormalization()(decoderOutNorm)
	#prepare different inputs for prediction
	decoderOutFinal = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE))(decoderOutNorm)
	contextFinal = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE))(contextOutNorm)
	prevWordFinal = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE))(decoderEmbedding)

	#combine
	wordOut = layers.Add()([contextFinal, decoderOutFinal, prevWordFinal])
	#wordOut = layers.BatchNormalization()(wordOut)
	wordOut = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE))(wordOut)
	#wordOut = layers.BatchNormalization()(wordOut)

	#word prediction
	wordOut = layers.TimeDistributed(layers.Dense(OUTPUT_VOCABULARY_COUNT, activation="softmax"))(wordOut)

	outputStage = Model(inputs=[contextOut, decoderOut, decoderEmbedding], outputs=[wordOut], name="output")
	return outputStage


def saveModels(trainingModel, modelName, samplingModels=False, saveWeights=True):
	# serialize model to JSON
	with open(CONST.MODEL_PATH + modelName + "_train.json", "w") as json_file:
		json_file.write(trainingModel.to_json())
	if samplingModels:
		with open(CONST.MODEL_PATH + modelName + "_sampInit.json", "w") as json_file:
			json_file.write(samplingModels[0].to_json())
		with open(CONST.MODEL_PATH + modelName + "_sampNext.json", "w") as json_file:
			json_file.write(samplingModels[1].to_json())
	
	if saveWeights:
		# serialize weights to HDF5
		trainingModel.save_weights(CONST.MODEL_PATH + modelName + ".h5")
		
	print("Saved model to disk")
	

def main():
	trainingModel, samplingModels = translationLSTMAttModel()
	saveModels(trainingModel, samplingModels=samplingModels, modelName="AttLSTM", saveWeights=False)
	trainingModel.summary()
	trainingModel.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["sparse_categorical_accuracy"])

	(xTrain, yTrain), (_, _) = PD.getFrToEngData()
	trainingCallbacks = [ModelCheckpoint(CONST.MODEL_PATH + "AttLSTMTrained-{epoch:02d}-{sparse_categorical_accuracy:.2f}.hdf5", monitor='sparse_categorical_accuracy', mode='max')]
	_ = trainingModel.fit(x=xTrain, y=yTrain, epochs=5, batch_size=4, validation_split=0.2, callbacks=trainingCallbacks)

	saveModels(trainingModel, modelName="AttLSTMTrained")

	# scores = trainingModel.evaluate(xTest, yTest, verbose=0)
	# print("%s: %.2f%%" % (trainingModel.metrics_names[1], scores[1]*100))


if __name__ == "__main__":
	main()


