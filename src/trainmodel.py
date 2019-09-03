import numpy as np
from keras.models import load_model
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
import json
import keras.backend as K
from keras.callbacks import ModelCheckpoint
import os
from keras.utils import Sequence

from . import constants as CONST
from .models.models import *

class DataPartitioningSequence(Sequence):
	def __init__(self, x, y, batchSize, numPartitions, initialEpoch=0):
		self.x = x
		self.y = y
		self.epoch = initialEpoch
		self.batchSize = batchSize
		self.numPartitions = numPartitions
		self.partitionSize = int(np.ceil(len(self.x[0])/self.numPartitions))
		self.partitionOffset = self.partitionSize * ((self.epoch-1) % self.numPartitions)

	def __len__(self):
		return int(np.ceil(len(self.x[0][self.partitionOffset:self.partitionOffset+self.partitionSize]) / self.batchSize))

	def __getitem__(self, idx):
		batchIdx = self.partitionOffset + idx * self.batchSize
		xBatch = [a[batchIdx:batchIdx + self.batchSize] for a in self.x]
		yBatch = [a[batchIdx:batchIdx + self.batchSize] for a in self.y]

		return xBatch, yBatch

	def on_epoch_end(self):
		self.epoch += 1
		self.partitionOffset = self.partitionSize * ((self.epoch-1) % self.numPartitions)


def saveModels(trainingModel, modelName, samplingModels=False):
	# serialize model to JSON
	with open(CONST.MODELS + modelName + "_train.json", "w") as json_file:
		json_file.write(trainingModel.to_json())
	if samplingModels:
		with open(CONST.MODELS + modelName + "_sampInit.json", "w") as json_file:
			json_file.write(samplingModels[0].to_json())
		with open(CONST.MODELS + modelName + "_sampNext.json", "w") as json_file:
			json_file.write(samplingModels[1].to_json())
	
	print("Saved model structure")
	
def evaluateModel(model, xTest, yTest):
	scores = model.evaluate(xTest, yTest, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


def getLastCheckpoint(modelName):
	c = os.listdir(CONST.MODELS)
	c = [x for x in c if x.startswith(modelName) and x.endswith(CONST.MODEL_NAME_SUFFIX)]
	return sorted(c)[-1] if c else False

def getLastEpoch(modelName):
	lastCheckpoint = getLastCheckpoint(modelName)
	if lastCheckpoint:
		epoch = lastCheckpoint[len(modelName)+1:][:4]
		try:
			return int(epoch)
		except ValueError:
			pass
	
	return 0

def loadModel(modelNum, loadForTraining=True):
	#get model
	if modelNum == 1:
		trainingModel, samplingModels = translationLSTMAttModel()
	else:
		trainingModel, samplingModels = translationTransformerModel()
	if loadForTraining:
		trainingModel.compile(optimizer=Adam(lr=CONST.LEARNING_RATE, decay=CONST.LEARNING_RATE_DECAY/CONST.DATA_PARTITIONS), loss=CONST.LOSS_FUNCTION, metrics=[CONST.EVALUATION_METRIC])
		trainingModel.summary()

	#load checkpoint if available
	checkPointName = getLastCheckpoint(trainingModel.name)
	if checkPointName:
		samplingModels[0].load_weights(CONST.MODELS + checkPointName)
		samplingModels[1].load_weights(CONST.MODELS + checkPointName, by_name=True)

		if loadForTraining:
			trainingModel.load_weights(CONST.MODELS + checkPointName)
			tempModel = load_model(CONST.MODELS + checkPointName, custom_objects={"CONST": CONST})
			trainingModel._make_train_function()
			weight_values = K.batch_get_value(getattr(tempModel.optimizer, 'weights'))
			trainingModel.optimizer.set_weights(weight_values)


	return trainingModel, samplingModels


def loadEncodedData(language):
	npData = np.load(CONST.PROCESSED_DATA + language + "EncodedData.npz")
	wordData = npData["encoded"]
	if CONST.INCLUDE_CHAR_EMBEDDING:
		charForwardData = npData["charForwardEncoded"]
		charBackwardData = npData["charBackwardEncoded"]
	npData = None

	wordData = wordData[:CONST.DATA_COUNT].copy()
	data = [wordData]
	if CONST.INCLUDE_CHAR_EMBEDDING:
		charForwardData = charForwardData[:CONST.DATA_COUNT].copy()
		charBackwardData = charBackwardData[:CONST.DATA_COUNT].copy()

		charData = np.concatenate((charForwardData, charBackwardData), axis=2)
		charData = np.reshape(charData, (charData.shape[0], -1))
		data.append(charData)

	return data
	
def getTrainingData(startLang, endLang):
	inData = loadEncodedData(startLang)
	outData = loadEncodedData(endLang)

	inputData = inData + outData
	outputData = outData[0]
	outputData = np.pad(outputData,((0,0),(0,1)), mode='constant')[:,1:]
	outputData = [np.expand_dims(outputData,axis=-1)]					#for sparse categorical
	
	
	trainingSplit = int(CONST.TRAIN_SPLIT_PCT * len(inputData[0]))
	validationSplit = int(trainingSplit*CONST.VALIDATION_SPLIT_PCT)
	
	trainIn = [x[:trainingSplit-validationSplit] for x in inputData]
	trainOut = [x[:trainingSplit-validationSplit] for x in outputData]

	valIn = [x[trainingSplit-validationSplit:trainingSplit] for x in inputData]
	valOut = [x[trainingSplit-validationSplit:trainingSplit] for x in outputData]

	testIn = [x[trainingSplit:] for x in inputData]
	testOut = [x[trainingSplit:] for x in outputData]

	return (trainIn, trainOut), (valIn, valOut), (testIn, testOut)


def trainModel(modelNum):
	#get model
	trainingModel, _ = loadModel(modelNum)

	# load all data
	(xTrain, yTrain), (xVal, yVal), (_, _) = getTrainingData(startLang="fr", endLang="en")

	# start training
	initialEpoch = getLastEpoch(trainingModel.name)
	callbacks = []
	callbacks.append(ModelCheckpoint(CONST.MODELS + trainingModel.name + CONST.MODEL_CHECKPOINT_NAME_SUFFIX, monitor=CONST.EVALUATION_METRIC, mode='max', save_best_only=True))

	trainingDataGenerator = DataPartitioningSequence(x=xTrain, y=yTrain, batchSize=CONST.BATCH_SIZE, numPartitions=CONST.DATA_PARTITIONS, initialEpoch=initialEpoch)
	_ = trainingModel.fit_generator(trainingDataGenerator, validation_data=(xVal, yVal), epochs=CONST.NUM_EPOCHS*CONST.DATA_PARTITIONS, callbacks=callbacks, initial_epoch=initialEpoch)
	#_ = trainingModel.fit(x=xTrain, y=yTrain, epochs=CONST.NUM_EPOCHS, batch_size=CONST.BATCH_SIZE, validation_data=(xVal, yVal), callbacks=callbacks, initial_epoch=initialEpoch)

	trainingModel.save(CONST.MODELS + trainingModel.name + CONST.MODEL_TRAINED_NAME_SUFFIX)



def main():
	trainModel(1)

if __name__ == "__main__":
	main()


