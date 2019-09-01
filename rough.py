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

futureMask()