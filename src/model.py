import numpy as np
from keras.models import Model
from keras import layers
from keras.optimizers import RMSprop


from recurrent_advanced import AttLSTMCond
import prepData as PD
import constants as CONST


TRAIN_SPLIT = 500000

PATH = CONST.PATH
MODEL_PATH = PATH + "models\\"


def getData():
	x = []
	y = []
	
	#x,y = PD.loadVectorizedData()
	
	xTrain = x[:TRAIN_SPLIT]
	xTest = x[TRAIN_SPLIT:]
	yTrain = y[:TRAIN_SPLIT]
	yTest = y[TRAIN_SPLIT:]
	
	return xTrain,yTrain,xTest,yTest


def translationLSTMAttModel():
	################################
	#training model creation start
	################################
	encoderInput = layers.Input(batch_shape=(None,CONST.INPUT_SENTENCE_LENGTH))
	encoderEmbedding = layers.Embedding(input_dim=CONST.INPUT_WORDS_COUNT, output_dim=CONST.EMBEDDING_SIZE, input_length=CONST.INPUT_SENTENCE_LENGTH)(encoderInput)
	encoderOut = layers.Bidirectional(layers.LSTM(CONST.NUM_LSTM_UNITS, return_sequence=True))(encoderEmbedding)
	
	decoderInput = layers.Input(batch_shape=(None,CONST.OUTPUT_SENTENCE_LENGTH))
	decoderEmbedding = layers.Embedding(input_dim=CONST.OUTPUT_WORDS_COUNT, output_dim=CONST.EMBEDDING_SIZE, input_length=CONST.OUTPUT_SENTENCE_LENGTH)(decoderInput)
	
	initialDecodeState = layers.Lambda(lambda x: x[:,-1,:])(encoderOut)
	initialDecodeState = layers.Flatten(axis=1)(initialDecodeState)
	
	attentionLayer_SHARED = AttLSTMCond(CONST.NUM_LSTM_UNITS, return_extra_variables=True, return_states=True, return_sequences=True)
	[proj_h, x_att, alphas, h_state] = attentionLayer_SHARED([decoderEmbedding,encoderOut,initialDecodeState])
	
	decoderState_SHARED = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE))
	decoderState = decoderState_SHARED(proj_h)
	contextVector_SHARED = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE))
	contextVector = contextVector_SHARED(x_att)
	prevPrediction_SHARED = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE))
	prevPrediction = prevPrediction_SHARED(decoderEmbedding)
	
	wordOut = layers.Add(activation='relu')([contextVector,decoderState,prevPrediction])
	wordOut_SHARED_1 = layers.TimeDistributed(layers.Dense(CONST.EMBEDDING_SIZE))
	wordOut = wordOut_SHARED_1(wordOut)
	wordOut_SHARED_2 = layers.TimeDistributed(layers.Dense(CONST.OUTPUT_WORDS_COUNT, activation="softmax"))
	wordOut = wordOut_SHARED_2(wordOut)
	
	trainingModel = Model(inputs=[encoderInput,decoderInput],outputs=[wordOut])
	trainingModel.compile(optimizer=RMSprop(lr=8e-4),loss='categorical_crossentropy',metrics=['acc'])
	


	################################
	#sampling model creation start
	################################

	###########
	#first step prediction model creation start
	samplingModelInit = Model(inputs=[encoderInput,decoderInput],outputs=[wordOut,encoderOut,h_state,alphas])
	samplingModelInit.compile(optimizer=RMSprop(lr=8e-4),loss='categorical_crossentropy',metrics=['acc'])

	###########
	#next steps prediction model creation start
	preprocessedEncoder = layers.Input(shape=(None,CONST.NUM_LSTM_UNITS*2))			#since Bi-LSTM
	previousAttState = layers.Input(shape=(None,CONST.NUM_LSTM_UNITS))

	[proj_h, x_att, alphas, h_state] = attentionLayer_SHARED([decoderEmbedding,preprocessedEncoder,previousAttState])

	decoderState = decoderState_SHARED(proj_h)
	contextVector = contextVector_SHARED(x_att)
	prevPrediction = prevPrediction_SHARED(decoderEmbedding)
	
	wordOut = layers.Add(activation='relu')([contextVector,decoderState,prevPrediction])
	wordOut = wordOut_SHARED_1(wordOut)
	wordOut = wordOut_SHARED_2(wordOut)
	
	samplingModelNext = Model(inputs=[preprocessedEncoder,previousAttState,decoderInput],outputs=[wordOut,preprocessedEncoder,h_state,alphas])
	samplingModelNext.compile(optimizer=RMSprop(lr=8e-4),loss='categorical_crossentropy',metrics=['acc'])


	return trainingModel, [samplingModelInit, samplingModelNext]
	

def customAttentionLayer():
	###########attention implementation
	decoderEmbedding = layers.Input(shape=(None,CONST.NUM_LSTM_UNITS*2))
	encoderOut = layers.Input(shape=(None,CONST.NUM_LSTM_UNITS*2))


	decoderLSTM, decoderStates = layers.LSTM(CONST.NUM_LSTM_UNITS, return_states=True)(decoderEmbedding)
	decoderToAtt = layers.RepeatVector(CONST.INPUT_SENTENCE_LENGTH)(decoderLSTM)
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
	with open(MODEL_PATH + modelName + ".json", "w") as json_file:
		json_file.write(model_json)
	
	# serialize weights to HDF5
	model.save_weights(MODEL_PATH + modelName + ".h5")
	print("Saved model to disk")
	

if __name__ == "__main__":
	xTrain,yTrain,xTest,yTest = getData()
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


