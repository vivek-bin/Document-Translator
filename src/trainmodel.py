import numpy as np
from keras.models import load_model
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
import json
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import os

from . import constants as CONST
from .models.models import *



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
	c = os.listdir(CONST.MODEL_PATH)
	c = [x for x in c if x.startswith(CONST.MODEL_CHECKPOINT_NAME_START) and x.endswith(CONST.MODEL_CHECKPOINT_NAME_END)]
	return sorted(c)[-1] if c else False


def loadModel(modelNum):
	#get model
	if modelNum == 1:
		trainingModel, samplingModels = translationLSTMAttModel()
	else:
		trainingModel, samplingModels = translationTransformerModel()
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


def loadEncodedData(language):
	data = np.load(CONST.PROCESSED_DATA + language + "EncodedData.npz")
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
	
def getTrainingData(startLang, endLang):
	inData = loadEncodedData(startLang)
	outData = loadEncodedData(endLang)

	inputData = inData + outData
	outputData = outData[0]
	outputData = np.pad(outputData,((0,0),(0,1)), mode='constant')[:,1:]
	outputData = [np.expand_dims(outputData,axis=-1)]					#for sparse categorical

	trainIn = [x[:CONST.TRAIN_SPLIT] for x in inputData]
	testIn = [x[CONST.TRAIN_SPLIT:] for x in inputData]
	
	trainOut = [x[:CONST.TRAIN_SPLIT] for x in outputData]
	testOut = [x[CONST.TRAIN_SPLIT:] for x in outputData]

	return (trainIn, trainOut), (testIn, testOut)


def trainModel(modelNum):
	#get model
	trainingModel, _ = loadModel(modelNum)

	# load all data
	(xTrain, yTrain), (_, _) = getTrainingData(startLang="fr", endLang="en")

	# start training
	initialEpoch = int(getLastCheckpoint()[len(CONST.MODEL_CHECKPOINT_NAME_START):][:4]) if getLastCheckpoint() else 0
	callbacks = []
	callbacks.append(ModelCheckpoint(CONST.MODEL_PATH + CONST.MODEL_CHECKPOINT_NAME, monitor=CONST.EVALUATION_METRIC, mode='max',save_best_only=True))
	_ = trainingModel.fit(x=xTrain, y=yTrain, epochs=CONST.NUM_EPOCHS, batch_size=CONST.BATCH_SIZE, validation_split=CONST.VALIDATION_SPLIT, callbacks=callbacks, initial_epoch=initialEpoch)

	trainingModel.save(CONST.MODEL_PATH + CONST.MODEL_CHECKPOINT_NAME_START + "9999" + CONST.MODEL_CHECKPOINT_NAME_END)



def main():
	trainModel()

if __name__ == "__main__":
	main()


