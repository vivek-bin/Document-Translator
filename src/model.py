import numpy as np
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop
import json

from recurrent_advanced import AttLSTMCond
import prepData as PD
import constants as CONST


def getData():
	fr, en = PD.loadEncodedData()
	frTrain = (x[:CONST.TRAIN_SPLIT] for x in fr)
	frTest = (x[CONST.TRAIN_SPLIT:] for x in fr)
	
	enTrain = (x[:CONST.TRAIN_SPLIT] for x in en)
	enTest = (x[CONST.TRAIN_SPLIT:] for x in en)

	return (frTrain, enTrain), (frTest, enTest)


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
	######ENCODER
	#word embedding
	encoderWordInput = layers.Input(batch_shape=(None,CONST.INPUT_SEQUENCE_LENGTH))
	encoderWordEmbedding = layers.Embedding(input_dim=INPUT_VOCABULARY_COUNT, output_dim=CONST.WORD_EMBEDDING_SIZE, input_length=CONST.INPUT_SEQUENCE_LENGTH)(encoderWordInput)
	#char embedding
	encoderCharInput = layers.Input(batch_shape=(None,CONST.INPUT_SEQUENCE_LENGTH * CONST.CHAR_INPUT_SIZE * 2))		# forward and backwards
	encoderCharEmbedding = layers.Embedding(input_dim=INPUT_CHAR_VOCABULARY_COUNT, output_dim=CONST.CHAR_EMBEDDING_SIZE, input_length=CONST.INPUT_SEQUENCE_LENGTH * CONST.CHAR_INPUT_SIZE * 2)(encoderCharInput)
	encoderCharEmbedding = layers.Reshape(target_shape=(CONST.INPUT_SEQUENCE_LENGTH, CONST.CHAR_INPUT_SIZE * 2 * CONST.CHAR_EMBEDDING_SIZE))(encoderCharEmbedding)
	#final input embedding
	encoderEmbedding = layers.Concatenate()(encoderWordEmbedding, encoderCharEmbedding)
	
	encoderOut = layers.Bidirectional(layers.LSTM(CONST.NUM_LSTM_UNITS, return_sequence=True))(encoderEmbedding)
	
	######DECODER
	#word embedding
	decoderWordInput = layers.Input(batch_shape=(None,CONST.OUTPUT_SEQUENCE_LENGTH))
	decoderWordEmbedding = layers.Embedding(input_dim=OUTPUT_VOCABULARY_COUNT, output_dim=CONST.WORD_EMBEDDING_SIZE, input_length=CONST.OUTPUT_SEQUENCE_LENGTH)(decoderWordInput)
	#char embedding
	decoderCharInput = layers.Input(batch_shape=(None,CONST.OUTPUT_SEQUENCE_LENGTH * CONST.CHAR_INPUT_SIZE * 2))		# forward and backwards
	decoderCharEmbedding = layers.Embedding(input_dim=OUTPUT_CHAR_VOCABULARY_COUNT, output_dim=CONST.CHAR_EMBEDDING_SIZE, input_length=CONST.OUTPUT_SEQUENCE_LENGTH * CONST.CHAR_INPUT_SIZE * 2)(decoderCharInput)
	decoderCharEmbedding = layers.Reshape(target_shape=(CONST.OUTPUT_SEQUENCE_LENGTH, CONST.CHAR_INPUT_SIZE * 2 * CONST.CHAR_EMBEDDING_SIZE))(decoderCharEmbedding)
	#final input embedding
	decoderEmbedding = layers.Concatenate()(decoderWordEmbedding, decoderCharEmbedding)

	initialDecodeState = layers.Lambda(lambda x: x[:,-1,:])(encoderOut)
	initialDecodeState = layers.Flatten(axis=1)(initialDecodeState)
	
	######ATTENTION
	attentionLayer_SHARED = AttLSTMCond(CONST.NUM_LSTM_UNITS, return_extra_variables=True, return_states=True, return_sequences=True)
	[proj_h, x_att, alphas, h_state] = attentionLayer_SHARED([decoderEmbedding,encoderOut,initialDecodeState])
	
	decoderState_SHARED = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE))
	decoderState = decoderState_SHARED(proj_h)
	contextVector_SHARED = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE))
	contextVector = contextVector_SHARED(x_att)
	prevPrediction_SHARED = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE))
	prevPrediction = prevPrediction_SHARED(decoderEmbedding)
	
	######FINAL PREDICTION STAGE
	wordOut = layers.Add(activation="relu")([contextVector,decoderState,prevPrediction])
	wordOut_SHARED_1 = layers.TimeDistributed(layers.Dense(CONST.WORD_EMBEDDING_SIZE))
	wordOut = wordOut_SHARED_1(wordOut)
	wordOut_SHARED_2 = layers.TimeDistributed(layers.Dense(OUTPUT_VOCABULARY_COUNT, activation="softmax"))
	wordOut = wordOut_SHARED_2(wordOut)
	
	trainingModel = Model(inputs=[encoderWordInput, encoderCharInput, decoderWordInput, decoderCharInput],outputs=[wordOut])
	trainingModel.compile(optimizer=RMSprop(lr=8e-4),loss="categorical_crossentropy",metrics=["acc"])
	


	################################
	#sampling model creation start
	################################

	###########
	#first step prediction model creation start
	samplingModelInit = Model(inputs=[encoderWordInput, encoderCharInput, decoderWordInput, decoderCharInput],outputs=[wordOut,encoderOut,h_state,alphas])
	samplingModelInit.compile(optimizer=RMSprop(lr=8e-4),loss="categorical_crossentropy",metrics=["acc"])

	###########
	#next steps prediction model creation start
	preprocessedEncoder = layers.Input(shape=(None,CONST.NUM_LSTM_UNITS*2))			#since Bi-LSTM
	previousAttState = layers.Input(shape=(None,CONST.NUM_LSTM_UNITS))

	[proj_h, x_att, alphas, h_state] = attentionLayer_SHARED([decoderEmbedding,preprocessedEncoder,previousAttState])

	decoderState = decoderState_SHARED(proj_h)
	contextVector = contextVector_SHARED(x_att)
	prevPrediction = prevPrediction_SHARED(decoderEmbedding)
	
	wordOut = layers.Add(activation="relu")([contextVector,decoderState,prevPrediction])
	wordOut = wordOut_SHARED_1(wordOut)
	wordOut = wordOut_SHARED_2(wordOut)
	
	samplingModelNext = Model(inputs=[preprocessedEncoder,previousAttState, decoderWordInput, decoderCharInput], outputs=[wordOut,preprocessedEncoder,h_state,alphas])
	samplingModelNext.compile(optimizer=RMSprop(lr=8e-4), loss="categorical_crossentropy", metrics=["acc"])


	return trainingModel, [samplingModelInit, samplingModelNext]
	

def customAttentionLayer():
	########### attention implementation
	decoderEmbedding = layers.Input(shape=(None,CONST.NUM_LSTM_UNITS*2))
	encoderOut = layers.Input(shape=(None,CONST.NUM_LSTM_UNITS*2))


	decoderLSTM, decoderStates = layers.LSTM(CONST.NUM_LSTM_UNITS, return_states=True, return_sequences=True)(decoderEmbedding)
	decoderToAtt = layers.RepeatVector(CONST.INPUT_SEQUENCE_LENGTH)(decoderLSTM)
	attentionInput = layers.Concatenate([encoderOut,decoderToAtt])
	attentionFC = layers.Dense(CONST.NUM_LSTM_UNITS)(attentionInput)
	attentionFC = layers.Dense(1)(attentionFC)
	attentionAlpha = layers.Activation("softmax")(attentionFC)
	attentionAlpha = layers.Flatten()(attentionAlpha)
	attentionAlpha = layers.RepeatVector(CONST.NUM_LSTM_UNITS)(attentionAlpha)
	contextOut = layers.Multiply()([encoderOut,attentionAlpha])

	return contextOut, decoderStates


def saveModel(model,modelName):
	# serialize model to JSON
	model_json = model.to_json()
	with open(CONST.MODEL_PATH + modelName + ".json", "w") as json_file:
		json_file.write(model_json)
	
	# serialize weights to HDF5
	model.save_weights(CONST.MODEL_PATH + modelName + ".h5")
	print("Saved model to disk")
	

if __name__ == "__main__":
	(xTrain, yTrain), (xTest, yTest) = getData()
	trainingModel, samplingModel = translationLSTMAttModel()

	trainingModel.summary()
	samplingModel[0].summary()
	samplingModel[1].summary()


	# history = trainingModel.fit(xTrain, yTrain,
	# 				epochs=50,
	# 				batch_size=128,
	# 				validation_split=0.2)

	# saveModel(trainingModel,"training_model")
	# saveModel(samplingModel,"sampling_model")

	# scores = trainingModel.evaluate(xTest, yTest, verbose=0)
	# print("%s: %.2f%%" % (trainingModel.metrics_names[1], scores[1]*100))


