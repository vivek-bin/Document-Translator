import numpy as np
from keras.models import load_model
from keras.models import Model
from keras import layers
from keras.optimizers import Adam
import json
import keras.backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
import os
from keras.utils import Sequence
import time
import h5py

from . import constants as CONST
from . import preparedata as PD
from .models.models import *
from .processing import fileaccess as FA

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
		currentPartitionSize = min(self.partitionSize, len(self.x[0]) - self.partitionOffset)
		return int(np.ceil(currentPartitionSize / self.batchSize))

	def __getitem__(self, idx):
		batchIdx = self.partitionOffset + idx * self.batchSize
		xBatch = [a[batchIdx:batchIdx + self.batchSize] for a in self.x]
		yBatch = [a[batchIdx:batchIdx + self.batchSize] for a in self.y]

		return xBatch, yBatch

	def on_epoch_end(self):
		self.epoch += 1
		self.partitionOffset = self.partitionSize * ((self.epoch-1) % self.numPartitions)

		currentPartitionSize = min(self.partitionSize, len(self.x[0]) - self.partitionOffset)
		print("Training on data from {0} to {1}".format(self.partitionOffset, self.partitionOffset + currentPartitionSize))

class DataPartitionLoadingSequence(Sequence):
	def __init__(self, startLang, endLang, batchSize, numPartitions, training=False, validation=False, testing=False, initialEpoch=0):
		'''
			training flag -> training data generator
			validation flag -> validation data generator
			testing flag -> testing data generator
			**if multiple flags from these are set, the first True from the above order is considered**
		'''
		assert type(training) == bool
		assert type(validation) == bool
		assert type(testing) == bool
		assert training or validation or testing

		self.startLang = startLang
		self.endLang = endLang
		self.epoch = initialEpoch
		self.batchSize = batchSize
		self.numPartitions = numPartitions
		self.partitionSize = int(np.ceil(FA.lenProcessedData(startLang)/self.numPartitions))

		if training:
			start = 0
			end = CONST.TRAIN_SPLIT_PCT * (1 - CONST.VALIDATION_SPLIT_PCT)
		elif validation:
			start = CONST.TRAIN_SPLIT_PCT * (1 - CONST.VALIDATION_SPLIT_PCT)
			end = CONST.TRAIN_SPLIT_PCT
		elif testing:
			start = CONST.TRAIN_SPLIT_PCT
			end = 1
		self.dataPartStart = int(start * self.partitionSize)
		self.dataPartEnd = int(end * self.partitionSize)

		self.startLangText = []
		self.endLangText = []
		self.loadData()

	def __len__(self):
		assert len(self.endLangText) == len(self.startLangText)
		assert len(self.endLangText) > 0

		return int(np.ceil(len(self.endLangText) / self.batchSize))

	def __getitem__(self, idx):
		assert len(self.endLangText) == len(self.startLangText)
		assert len(self.endLangText) > 0

		startLangBatchText = self.startLangText[idx * self.batchSize:idx * self.batchSize + self.batchSize]
		endLangBatchText = self.endLangText[idx * self.batchSize:idx * self.batchSize + self.batchSize]

		startLangEncoded = []
		startLangEncoded.append(PD.encodeWords(startLangBatchText, self.startLang))
		if CONST.INCLUDE_CHAR_EMBEDDING:
			startLangEncoded.append(PD.encodeCharsForward(startLangBatchText, self.startLang))
			startLangEncoded.append(PD.encodeCharsBackward(startLangBatchText, self.startLang))

		endLangEncoded = []
		endLangEncoded.append(PD.encodeWords(endLangBatchText, self.endLang))
		if CONST.INCLUDE_CHAR_EMBEDDING:
			endLangEncoded.append(PD.encodeCharsForward(endLangBatchText, self.endLang))
			endLangEncoded.append(PD.encodeCharsBackward(endLangBatchText, self.endLang))

		xBatch = startLangEncoded + endLangEncoded
		yBatch = endLangEncoded[0]
		yBatch = np.pad(yBatch,((0,0),(0,1)), mode='constant')[:,1:]
		yBatch = [np.expand_dims(yBatch,axis=-1)]

		return xBatch, yBatch

	def on_epoch_end(self):
		self.epoch += 1
		self.loadData()

	def loadData(self):
		# load next data partition
		partitionOffset = self.partitionSize * ((self.epoch-1) % self.numPartitions)

		self.startLangText = FA.readProcessedData(self.startLang, partitionOffset + self.dataPartStart, partitionOffset + self.dataPartEnd)
		self.endLangText = FA.readProcessedData(self.endLang, partitionOffset + self.dataPartStart, partitionOffset + self.dataPartEnd)
		data = list(zip(self.startLangText, self.endLangText))
		data.sort(key=lambda x:len(x[1]))
		self.startLangText = [ls for ls, le in data]
		self.endLangText = [le for ls, le in data]


def sparseCrossEntropyLoss(targets=None, outputs=None):
	batchSize = K.shape(outputs)[0]
	sequenceSize = K.shape(outputs)[1]
	vocabularySize = K.shape(outputs)[2]
	firstPositionShifter = K.repeat(K.expand_dims(K.arange(sequenceSize) * vocabularySize, 0), batchSize)
	secondPositionShifter = K.repeat(K.expand_dims(K.arange(batchSize) * sequenceSize * vocabularySize, 1), sequenceSize)

	shiftedtargets = K.cast(K.flatten(targets), "int32") + K.flatten(firstPositionShifter) + K.flatten(secondPositionShifter)

	relevantValues = K.gather(K.flatten(outputs), shiftedtargets)
	relevantValues = K.reshape(relevantValues, (batchSize, -1))
	relevantValues = K.clip(relevantValues, K.epsilon(), 1. - K.epsilon())
	cost = -K.sum(K.log(relevantValues), axis=-1) / K.cast(K.shape(relevantValues)[1], "float32")				# sum the cross entropy for all words in the sequence
	return cost


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
		#trainingModel.compile(optimizer=Adam(lr=CONST.LEARNING_RATE, decay=CONST.LEARNING_RATE_DECAY/CONST.DATA_PARTITIONS), loss=CONST.LOSS_FUNCTION, metrics=[CONST.EVALUATION_METRIC])
		trainingModel.compile(optimizer=Adam(lr=CONST.LEARNING_RATE, decay=CONST.LEARNING_RATE_DECAY/CONST.DATA_PARTITIONS), loss=sparseCrossEntropyLoss, metrics=[CONST.EVALUATION_METRIC])
		trainingModel.summary()

	#load checkpoint if available
	checkPointName = getLastCheckpoint(trainingModel.name)
	if checkPointName:
		samplingModels[0].load_weights(CONST.MODELS + checkPointName)
		samplingModels[1].load_weights(CONST.MODELS + checkPointName, by_name=True)

		if loadForTraining:
			trainingModel.load_weights(CONST.MODELS + checkPointName)
			savedOptimizerStates = h5py.File(CONST.MODELS + checkPointName, mode="r")["optimizer_weights"]
			optimizerWeightNames = [n.decode('utf8') for n in savedOptimizerStates.attrs['weight_names']]
			optimizerWeightValues = [savedOptimizerStates[n] for n in optimizerWeightNames]

			trainingModel._make_train_function()
			trainingModel.optimizer.set_weights(optimizerWeightValues)


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
	startLangData = loadEncodedData(startLang)
	endLangData = loadEncodedData(endLang)

	inputData = startLangData + endLangData
	outputData = endLangData[0]
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


def trainModel(modelNum, startLang="fr", endLang="en"):
	# get model
	trainingModel, _ = loadModel(modelNum)
	initialEpoch = getLastEpoch(trainingModel.name)

	# load all data
	# (xTrain, yTrain), (xVal, yVal), (_, _) = getTrainingData(startLang, endLang)
	# trainingDataGenerator = DataPartitioningSequence(x=xTrain, y=yTrain, batchSize=CONST.BATCH_SIZE, numPartitions=CONST.DATA_PARTITIONS, initialEpoch=initialEpoch)
	# validationDataGenerator = DataPartitioningSequence(x=xVal, y=yVal, batchSize=CONST.BATCH_SIZE, numPartitions=CONST.DATA_PARTITIONS, initialEpoch=initialEpoch)
	trainingDataGenerator = DataPartitionLoadingSequence(training=True, startLang=startLang, endLang=endLang, batchSize=CONST.BATCH_SIZE, numPartitions=CONST.DATA_PARTITIONS, initialEpoch=initialEpoch)
	validationDataGenerator = DataPartitionLoadingSequence(validation=True, startLang=startLang, endLang=endLang, batchSize=CONST.BATCH_SIZE, numPartitions=CONST.DATA_PARTITIONS, initialEpoch=initialEpoch)

	# prepare callbacks
	callbacks = []
	callbacks.append(ModelCheckpoint(CONST.MODELS + trainingModel.name + CONST.MODEL_CHECKPOINT_NAME_SUFFIX, monitor=CONST.EVALUATION_METRIC, mode='max', save_best_only=True))
	callbacks.append(TensorBoard(log_dir=CONST.LOGS + "tensorboard-log-{}".format(time.time()), histogram_freq=0, batch_size=CONST.BATCH_SIZE))#, write_graph=True, write_grads=True, write_images=True))

	# start training
	_ = trainingModel.fit_generator(trainingDataGenerator, validation_data=validationDataGenerator, epochs=CONST.NUM_EPOCHS*CONST.DATA_PARTITIONS, callbacks=callbacks, initial_epoch=initialEpoch)

	# save model after training
	trainingModel.save(CONST.MODELS + trainingModel.name + CONST.MODEL_TRAINED_NAME_SUFFIX)



def main():
	trainModel(1)

if __name__ == "__main__":
	main()


