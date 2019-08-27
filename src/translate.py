import json
import numpy as np

import constants as CONST
from model import loadModel
import prepData as PD



def encoderEncodeData(data, language):
	wordData, charData = encodeData(data, language)
	charData = np.reshape(charData, (charData.shape[0], -1))

	return wordData, charData


def decoderEncodeData(data, language, initial=False):
	wordData, charData = encodeData(data, language)
	if initial:
		wordData = wordData[:,0]
		charData = charData[:,0]
	else:
		wordData = wordData[:,1]
		charData = charData[:,1]
	charData = np.reshape(charData, (charData.shape[0], -1))

	return wordData, charData


def encodeData(data, language):
	wordData = PD.encodeWords(data, language)
	charForwardData = PD.encodeCharsForward(data, language)
	charBackwardData = PD.encodeCharsBackward(data, language)
	charData = np.concatenate((charForwardData, charBackwardData), axis=2)

	return wordData, charData


def decodeWord(wordOut, alphas, language):
	#get output word encodings
	with open(CONST.ENCODING_PATH+language+"_word.json", "r") as f:
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
	_, (samplingModelInit, samplingModelNext) = loadModel()


	# prepare input sentence
	cleanData = PD.cleanText(inputStrings, startLang)
	wordData, charData = encoderEncodeData(cleanData, startLang)
	
	# prepare decoder input
	initialOutput = ["" for _ in inputStrings]
	decoderWordData, decoderCharData = decoderEncodeData(initialOutput, endLang, initial=True)
	
	# decode first word
	[wordOut, preprocessedEncoder, decoderH, decoderC, alphas] = samplingModelInit([wordData, charData, decoderWordData, decoderCharData])
	outputStringsNext = decodeWord(wordOut, alphas, endLang)

	predictedWords = [outputStringsNext]
	continueTranslation = False
	
	while continueTranslation and len(predictedWords) < CONST.MAX_TRANSLATION_LENGTH:
		# decode rest of the sentences
		decoderWordData, decoderCharData = decoderEncodeData(outputStringsNext, endLang)
		[wordOut, preprocessedEncoder, decoderH, decoderC, alphas] = samplingModelNext([preprocessedEncoder, decoderH, decoderC, decoderWordData, decoderCharData])

		# decode predicted word
		outputStringsNext = decodeWord(wordOut, alphas, endLang)
		predictedWords.append(outputStringsNext)
		continueTranslation = False in [(CONST.END_OF_SEQUENCE_TOKEN in x) for x in zip(*predictedWords)]


	outputWordList = list(zip(*predictedWords))
	outputStrings = prepareSentences(outputWordList)


	return outputStrings





