import json
import numpy as np

from . import constants as CONST
from .trainmodel import loadModel
from .processing import preparedata as PD



def encoderEncodeData(data, language):
	data = encodeData(data, language)
	if CONST.INCLUDE_CHAR_EMBEDDING:
		data[1] = np.reshape(data[1], (data[1].shape[0], -1))

	return data


def decoderEncodeData(data, language, initial=False, onlyLastWord=True):
	data = encodeData(data, language)
	if initial:
		data = [x[:,0] for x in data]
	elif onlyLastWord:
		data = [x[:,1] for x in data]
	if CONST.INCLUDE_CHAR_EMBEDDING:
		data[1] = np.reshape(data[1], (data[1].shape[0], -1))

	return data


def encodeData(data, language):
	wordData = PD.encodeWords(data, language)
	data = [wordData]
	if CONST.INCLUDE_CHAR_EMBEDDING:
		charForwardData = PD.encodeCharsForward(data, language)
		charBackwardData = PD.encodeCharsBackward(data, language)
		charData = np.concatenate((charForwardData, charBackwardData), axis=2)
		data.append(charData)

	return data


def decodeWord(wordOut, alphas, language):
	#get output word encodings
	with open(CONST.ENCODINGS+language+"_word.json", "r") as f:
		wordEncoding = {w:i for i,w in enumerate(json.load(f))}

	outputWord = [wordEncoding[word] for word in np.argmax(wordOut, axis=-1)]

	return outputWord


def prepareSentences(wordLists):
	outputStrings = []
	line = ""
	for words in wordLists:
		line = ""
		for i, word in enumerate(words):
			if word == CONST.END_OF_SEQUENCE_TOKEN:
				break
			line += word + " "
			i=i

		outputStrings.append(line)

	return outputStrings
	

def translate(inputStrings, startLang="fr", endLang="en"):
	#get model
	_, (samplingModelInit, samplingModelNext) = loadModel(modelNum, loadOptimizerWeights=False)


	# prepare input sentence
	cleanData = PD.cleanText(inputStrings, startLang)
	encoderData = encoderEncodeData(cleanData, startLang)
	
	# prepare decoder input
	initialOutput = ["" for _ in inputStrings]
	decoderData = decoderEncodeData(initialOutput, endLang, initial=True)
	
	# decode first word
	[wordOut, preprocessedEncoder, decoderH, decoderC, alphas] = samplingModelInit(encoderData + decoderData)
	outputStringsNext = decodeWord(wordOut, alphas, endLang)

	predictedWords = [outputStringsNext]
	continueTranslation = False
	
	while continueTranslation and len(predictedWords) < CONST.MAX_TRANSLATION_LENGTH:
		# decode rest of the sentences
		decoderData = decoderEncodeData(outputStringsNext, endLang)
		[wordOut, preprocessedEncoder, decoderH, decoderC, alphas] = samplingModelNext([preprocessedEncoder, decoderH, decoderC] + decoderData)

		# decode predicted word
		outputStringsNext = decodeWord(wordOut, alphas, endLang)
		predictedWords.append(outputStringsNext)
		continueTranslation = False in [(CONST.END_OF_SEQUENCE_TOKEN in x) for x in zip(*predictedWords)]


	outputWordList = list(zip(*predictedWords))
	outputStrings = prepareSentences(outputWordList)


	return outputStrings





