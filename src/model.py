import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop


from recurrent_advanced import AttLSTMCond
import prepData as PD


TRAIN_SPLIT = 500000

PATH = "D:\\ML\\translate\\data\\"
PATH = PD.PATH
MODEL_PATH = PATH + "models\\"


def getData():
	x,y = PD.loadVectorizedData()
		
	xTrain = x[:TRAIN_SPLIT]
	xTest = x[TRAIN_SPLIT:]
	yTrain = y[:TRAIN_SPLIT]
	yTest = y[TRAIN_SPLIT:]
	
	return xTrain,yTrain,xTest,yTest


def trainLSTMAttModel(xTrain,yTrain):

	INPUT_SENTENCE_LENGTH = 64
	OUTPUT_SENTENCE_LENGTH = 1
	INPUT_WORDS_COUNT = 10000
	OUTPUT_WORDS_COUNT = 10000
	EMBEDDING_SIZE = 128
	ENCODER_LSTM_UNITS = 128
	DECODER_LSTM_UNITS = 128
	ATTENTION_SIZE = 128
	NEXT_WORD_SIZE = 128

	encoderInput = layers.Input(batch_shape=(None,INPUT_SENTENCE_LENGTH))
	encoderEmbedding = layers.Embedding(input_dim=INPUT_WORDS_COUNT, output_dim=EMBEDDING_SIZE, input_length=INPUT_SENTENCE_LENGTH)(encoderInput)
	encoderOut = layers.Bidirectional(layers.LSTM(ENCODER_LSTM_UNITS, return_sequence=True))(encoderEmbedding)
	
	decoderInput = layers.Input(batch_shape=(None,OUTPUT_SENTENCE_LENGTH))
	decoderEmbedding = layers.Embedding(input_dim=OUTPUT_WORDS_COUNT, output_dim=EMBEDDING_SIZE, input_length=OUTPUT_SENTENCE_LENGTH)(decoderInput)
	
	initialDecodeState = layers.Lambda(lambda x: x[:,-1,:])(encoderOut)
	initialDecodeState = layers.Flatten(axis=1)(initialDecodeState)
	
	attentionLayer = AttLSTMCond(ENCODER_LSTM_UNITS, return_extra_variables=True, return_states=True, return_sequences=True)
	[proj_h, x_att, alphas, h_state] = attentionLayer([decoderEmbedding,encoderOut,initialDecodeState])
	
	############attention implementation
	#decoderLSTM, decoderStates = layers.LSTM(DECODER_LSTM_UNITS, return_states=True)(decoderEmbedding)
	#decoderToAtt = layers.RepeatVector(INPUT_SENTENCE_LENGTH)(decoderLSTM)
	#attentionInput = layers.Concat([encoderOut,decoderToAtt])
	#attentionFC = layers.Dense(ATTENTION_SIZE)(attentionInput)
	#attentionFC = layers.Dense(1)(attentionFC)
	#attentionAlpha = layers.Activation("softmax")(attentionFC)
	#attentionAlpha = layers.Flaten()(attentionAlpha)
	#attentionAlpha = layers.RepeatVector(ENCODER_LSTM_UNITS)(attentionAlpha)
	#contextOut = layers.Multiply([encoderOut,attentionAlpha])

	
	decoderState = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE))(proj_h)
	contextVector = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE))(x_att)
	prevPrediction = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE))(decoderEmbedding)
		
	
	
	wordOut = layers.Add(activation='relu')([contextVector,decoderState,prevPrediction])
	wordOut = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE))(wordOut)
	wordOut = layers.TimeDistributed(layers.Dense(OUTPUT_WORDS_COUNT, activation="softmax"))(wordOut)
	
	trainingModel = Model(inputs=[encoderInput,decoderInput],outputs=[wordOut])
	
	
	
	
	#model.summary()
	trainingModel.compile(	optimizer=RMSprop(lr=8e-4),
					loss='categorical_crossentropy',
					metrics=['acc'])
	
	
	history = trainingModel.fit(xTrain, yTrain,
					epochs=40,
					batch_size=128,
					validation_split=0.2)
	
	trainingModel.summary()
	saveModel(trainingModel, "training_model")
	
	return trainingModel
	

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
	model = trainConvModel(xTrain,yTrain)

	#scores = model.evaluate(xTest, yTest, verbose=0)
	#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


