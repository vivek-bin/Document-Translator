def matMulSample():
	import tensorflow as tf
	import numpy as np
	batch_size = 2
	sequence_size = 3
	embed_dim = 4

	M = tf.constant(np.arange(0,batch_size * sequence_size * embed_dim), shape=[batch_size, sequence_size, embed_dim])
	U = tf.constant(np.arange(batch_size * embed_dim), shape=[batch_size, embed_dim])

	prod = tf.einsum('bse,be->bs', M, U)

	with tf.Session():
		print("M")
		print(M.eval())
		print("")
		print("U")
		print(U.eval())
		print("")
		print("einsum result")
		print(prod.eval())
		print("")

		print("numpy, example 0")
		print(np.matmul(M.eval()[0], U.eval()[0]))
		print("")
		print("numpy, example 1")
		print(np.matmul(M.eval()[1], U.eval()[1]))


def repeatLayerTry():
	import keras

	model = keras.models.Sequential()
	model.add(keras.layers.GRU(23,input_shape=(7,13), return_sequences=True))
	model.add(keras.layers.RepeatVector(5))

	model.summary()


def dotLayerTry():
	import keras
	import numpy as np

	encoderOut = keras.layers.Input(shape=(2,4))
	decoderLSTM = keras.layers.Input(shape=(3,4))

	# permEncoderOut = keras.layers.Permute((2,1))(encoderOut)
	# permDecoderOut = keras.layers.Permute((2,1))(decoderLSTM)
	alpha = keras.layers.dot([decoderLSTM, encoderOut], axes=2)

	model = keras.models.Model(inputs=[encoderOut,decoderLSTM], outputs=[alpha])

	model.summary()

	enc = np.arange(8).reshape((1,2,4))
	dec = np.arange(-1,-13,-1).reshape((1,3,4))

	enc = np.array([1,2,3,5,7,11,13,17]).reshape((1,2,4))
	dec = np.array([19,23,29,31,37,41,43,47,53,59,61,67]).reshape((1,3,4))

	print(enc)
	print(dec)
	#print(model.predict([enc,dec])[0])
	print(model.predict([enc,dec]))


def multiPartModel():
	import keras

	models = []
	for i in range(5):
		ip = keras.layers.Input(batch_shape=(None,10))
		d = keras.layers.Dense(10)(ip)

		models.append(keras.models.Model(inputs=ip, outputs=d))

	ip = keras.layers.Input(batch_shape=(None,10))
	op = ip
	
	for i in range(5):
		op = models[i](op)

	fmodel = keras.models.Model(inputs=ip, outputs=op)
	fmodel.compile(optimizer=keras.optimizers.RMSprop(lr=8e-4),loss="categorical_crossentropy",metrics=["acc"])


	fmodel.summary()

	print(fmodel.layers[4].layers[1].name)
	return fmodel
	

def positionEncodingSpeed():
	from keras import backend as K
	import numpy as np
	scale = 1
	batchSize = 256 // scale
	sequenceLength = 1000 // scale
	embeddingSize = 512 // scale
	
	positionEncoding = np.array([[pos / np.power(10, 8. * i / embeddingSize) for i in range(embeddingSize)] for pos in range(sequenceLength)])
	positionEncoding[:, 0::2] = np.sin(positionEncoding[:, 0::2])
	positionEncoding[:, 1::2] = np.cos(positionEncoding[:, 1::2])

	positionEncoding = K.expand_dims(K.variable(positionEncoding), 0)
	positionEncoding = K.tile(positionEncoding, [batchSize,1,1])

	return positionEncoding	
		

def futureMask():
	from keras import backend as K
	import numpy as np

	batchShape = 3
	seqLen = 5

	m = np.arange(seqLen)
	m1 = np.tile(np.expand_dims(m, 0), [seqLen, 1])
	m2 = np.tile(np.expand_dims(m, 1), [1, seqLen])
	#mask = np.cast(np.greater_equal(m1, m2), "float32")
	mask = np.greater_equal(m2, m1).astype("float32")
	mask = np.tile(np.expand_dims(mask, 0), [batchShape, 1, 1])
	print(mask)


def readDictionaryPDF(bestWordOnly=True):
	from translator import constants as CONST
	from PyPDF2 import PdfFileReader

	with open(CONST.DICTIONARY_PATH, "rb") as pdfFileBinary:
		pdfFile = PdfFileReader(pdfFileBinary)
		pageTexts = []
		for i in range(3, pdfFile.numPages-1):
			footerLen = len("English-french (dictionnaire)English-french Dictionary\n" + str(i))
			pageTexts.append(pdfFile.getPage(i).extractText()[:-footerLen].split("\n"))

	dictList = [x.lower().split(":") for page in pageTexts for x in page if x]
	engToFrDict = {x[0].strip():[v.strip() for v in x[1].split(", ")] for x in dictList}
	
	frToEngDict = {}
	for key, valueList in engToFrDict.items():
		for value in valueList:
			try:
				frToEngDict[value].append(key)
			except KeyError:
				frToEngDict[value] = [key]
	
	print(engToFrDict["diagnosed"])
	print(frToEngDict["un à un"])
	
	if bestWordOnly:
		# already sorted as such in dictionary for eng->fr
		engToFrDict = {key:valueList[0] for key,valueList in engToFrDict.items()}

		# select shortest word as best
		frToEngDict = {key:[v for v in valueList if len(v) == min([len(v) for v in valueList])][0] for key, valueList in frToEngDict.items()}

	
	print(engToFrDict["diagnosed"])
	print(frToEngDict["un à un"])
	
	return engToFrDict, frToEngDict


def lossFuncShiftPos():
	from keras import backend as K

	targets = K.constant([[3, 1, 2],[2, 0, 3]], dtype="int32")
	outputs = K.constant([[[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]],[[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]])

	batchSize = K.shape(outputs)[0]
	sequenceSize = K.shape(outputs)[1]
	vocabularySize = K.shape(outputs)[2]
	
	firstPositionShifter = K.flatten(K.repeat(K.expand_dims(K.arange(sequenceSize) * vocabularySize, 0), batchSize))
	#secondPositionShifter = K.repeat_elements(K.arange(batchSize) * sequenceSize * vocabularySize, rep=sequenceSize, axis=-1)
	secondPositionShifter = K.flatten(K.repeat(K.expand_dims(K.arange(batchSize) * sequenceSize * vocabularySize, 1), sequenceSize))

	shiftedtargets = K.flatten(targets) + firstPositionShifter + secondPositionShifter

	relevantValues = K.reshape(K.gather(K.flatten(outputs), shiftedtargets), (batchSize, -1))
	print(K.eval(firstPositionShifter))
	print(K.eval(secondPositionShifter))
	print(K.eval(relevantValues))


def sampleEncoded():
	from translator import preparedata as PD
	from translator.processing import fileaccess as FA
	import numpy as np

	text = FA.readProcessedData("en", 100000, 100000 + 100)
	encText = PD.encodeWords(text, "en")

	encLen = [len([y for y in x if y]) for x in encText]
	encUnkLen = [len([y for y in x if y == 19894]) for x in encText]
	encLenUnkLenPair = list(zip(encLen, encUnkLen))
	uncRatio = [ul/l for l, ul in encLenUnkLenPair]
	maxUnk = np.argmax(encUnkLen)
	print(text[maxUnk][-1])
	print(text[maxUnk])
	print(encText[maxUnk])


	# for x in encLenUnkLenPair:
	# 	print(x)
	print(sum(uncRatio)/len(uncRatio))


def visualizeModel():
	from translator import constants as CONST
	with CONST.HiddenPrints():
		from translator import trainmodel as TM
	TM.visualizeModel(1)
	TM.visualizeModel(2)


def shiftMat(inputSize=16, outputSize=16, connectionSize=4, stepSize=2):
	from keras import backend as K

	inputSize = inputSize * 2 # double to circle over

	weights = K.reshape(K.arange(1, connectionSize*outputSize+1, dtype="int16"), shape=(connectionSize, outputSize))
	
	padding = K.zeros(shape=(inputSize - connectionSize + stepSize, outputSize), dtype="int16")

	weightsPad = K.concatenate([weights, padding], axis=0)
	weightsPad = K.permute_dimensions(weightsPad,(1,0))
	weightsPad = K.reshape(weightsPad, shape=(-1, outputSize))
	weightsPad = K.permute_dimensions(weightsPad,(1,0))
	weightsPad = K.reshape(weightsPad, shape=(inputSize, -1))

	print(weights.shape)
	print(padding.shape)
	print(weightsPad.shape)
	
	print(K.eval(weightsPad))

def sparseDenseLayer():
	import numpy as np
	from keras import backend as K
	from keras.legacy import interfaces
	from keras.datasets import mnist
	from keras.models import Sequential, load_model
	from keras.layers.core import Dense, Dropout, Activation
	from keras.utils import np_utils
	

	class SparseDense(Dense):
		@interfaces.legacy_dense_support
		def __init__(self, units,
					connections=None,
					step=None,
					**kwargs):
			super(SparseDense, self).__init__(units, **kwargs)
			self.connections = connections if connections is not None else units // 16
			self.step = units if units is not None else connections // 4


		def build(self, input_shape):
			print(input_shape)
			super(SparseDense, self).build(input_shape)
			self.weight_mask = np.zeros(shape=(input_shape[1], self.units))
			for i in range(self.units):
				for j in range(self.connections):
					self.weight_mask[((i*self.step)+j)%input_shape[1], i] = 1.

		def call(self, inputs):
			output = K.dot(inputs, self.kernel*self.weight_mask)
			if self.use_bias:
				output = K.bias_add(output, self.bias, data_format='channels_last')
			if self.activation is not None:
				output = self.activation(output)
			return output

	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	# let's print the shape before we reshape and normalize
	print("X_train shape", X_train.shape)
	print("y_train shape", y_train.shape)
	print("X_test shape", X_test.shape)
	print("y_test shape", y_test.shape)

	# building the input vector from the 28x28 pixels
	X_train = X_train.reshape(60000, 784).astype('float32')
	X_test = X_test.reshape(10000, 784).astype('float32')

	# normalizing the data to help with the training
	X_train /= 255
	X_test /= 255

	n_classes = 10
	Y_train = np_utils.to_categorical(y_train, n_classes)
	Y_test = np_utils.to_categorical(y_test, n_classes)

	models = []
	# building a linear stack of layers with the sequential model
	model = Sequential()
	model.add(Dense(256, input_shape=(784,), activation="relu"))
	model.add(Dense(10, activation="softmax"))
	models.append(model)
	
	# building a linear stack of layers with the sequential model
	model = Sequential()
	model.add(SparseDense(32, connections=8, step=2, input_shape=(784,), activation="relu"))
	model.add(Dropout(0.2))
	model.add(SparseDense(32, connections=8, step=2, activation="relu"))
	model.add(Dropout(0.2))
	model.add(SparseDense(32, connections=4, step=2, activation="relu"))
	model.add(Dropout(0.2))
	model.add(SparseDense(32, connections=4, step=2, activation="relu"))
	model.add(Dropout(0.2))
	model.add(Dense(10, activation="softmax"))
	models.append(model)

	#X_train = np.random.rand(100000, 784)
	#X_test = np.random.rand(10000, 784)
	#Y_train = np_utils.to_categorical((np.random.rand(100000)*10).astype("int32"), 10)
	#Y_test = np_utils.to_categorical((np.random.rand(10000)*10).astype("int32"), 10)
	# compiling the sequential model
	for model in models:
		model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
		model.summary()
	for model in models:
		_ = model.fit(X_train, Y_train, batch_size=128, epochs=30, verbose=2, validation_data=(X_test, Y_test))

def docextract():
	import translator.constants as CONST
	from translator.processing import projextract
	import os

	for dirName in sorted(os.listdir(CONST.PROJECT_TRANSLATIONS_MATCHED_PATH)):
		print(dirName)
		inputDir = CONST.PROJECT_TRANSLATIONS_MATCHED_PATH + dirName + "/"
		outputDir = CONST.PROJECT_TRANSLATIONS_EXTRACT_PATH + dirName + "/"

		projextract.extractAllGroupsInDirectory(inputDir, outputDir)

def trainingDataInfo():
	import translator.constants as CONST
	import translator.processing.fileaccess as FA
	for lang in ["en","fr"]:
		data = FA.readProcessedData(lang)
		nd = {}
		i = 0
		t = 0
		mn = 9999
		mx = -1
		for line in data:
			count = CONST.NUM_WORDPIECES(line)
			mx = count if count > mx else mx
			mn = count if count < mn else mn
			i += 1
			t += count
			try:
				nd[(count//10) * 10] += 1
			except KeyError:
				nd[(count//10) * 10] = 1
			
		for k in sorted(nd.keys()):
			print(str(k).zfill(3),":",nd[k])
		print("max={}, min={}, average={}".format(mx, mn, t/i))



trainingDataInfo()

