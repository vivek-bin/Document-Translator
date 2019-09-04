import json
import numpy as np

from . import constants as CONST
from . import preparedata as PD
from .trainmodel import loadModel
from .processing import fileaccess as FA


class Translater:
	def __init__(self, startLang="fr", endLang="en", modelNum=1):
		self.startLang = startLang
		self.endLang = endLang
		self.modelNum = modelNum
		
		with open(CONST.ENCODINGS+self.endLang+"_word.json", "r") as f:
			self.endLangDecoding = {i:w for i,w in enumerate(json.load(f))}
		if CONST.INCLUDE_CHAR_EMBEDDING:
			with open(CONST.ENCODINGS+self.endLang+"_char.json", "r") as f:
				self.endLangCharDecoding = {i:w for i,w in enumerate(json.load(f))}

		_, (self.samplingModelInit, self.samplingModelNext) = loadModel(self.modelNum, loadForTraining=False)

		self.wordDictionary = FA.readDictionaryPDF()[self.startLang]


	def encodeData(self, data, language):
		wordData = PD.encodeWords(data, language)
		data = [wordData]
		if CONST.INCLUDE_CHAR_EMBEDDING:
			charForwardData = PD.encodeCharsForward(data, language)
			charBackwardData = PD.encodeCharsBackward(data, language)
			charData = np.concatenate((charForwardData, charBackwardData), axis=2)
			data.append(charData)

		return data


	def encoderEncodeData(self, data):
		data = self.encodeData(data, self.startLang)
		if CONST.INCLUDE_CHAR_EMBEDDING:
			data[1] = np.reshape(data[1], (data[1].shape[0], -1))

		return data


	def decoderEncodeData(self, data, initial=False, onlyLastWord=True):
		data = [x for xl in data for x in xl]
		data = self.encodeData(data, self.endLang)
		if initial:
			data = [x[:,0:1] for x in data]
		elif onlyLastWord:
			data = [x[:,1:2] for x in data]
		else:
			data = [x[:,:-1] for x in data]
		if CONST.INCLUDE_CHAR_EMBEDDING:
			data[1] = np.reshape(data[1], (data[1].shape[0], -1))

		return data


	def decodeWord(self, wordSoftmax, alpha, cleanString):
		#get output word encodings
		topPredictions = np.argsort(-wordSoftmax)[:CONST.BEAM_SIZE]
		score = wordSoftmax[topPredictions]
		originalWordParts = [CONST.START_OF_SEQUENCE_TOKEN] + cleanString[0].split(CONST.UNIT_SEP) + [CONST.END_OF_SEQUENCE_TOKEN]

		outputWord = []
		for wordIndex in topPredictions:
			word = None
			try:
				word = self.endLangDecoding[wordIndex]
			except KeyError:
				print("not found in dictionary; should not happen as should become UNK token if OOV")
			if not word or word == CONST.UNKNOWN_TOKEN:
				encoderPosition = np.argmax(alpha)
				originalWord = originalWordParts[encoderPosition]
				try:
					word = self.wordDictionary[originalWord]
				except KeyError:
					word = originalWord
				if word in ("SOS", "EOS"):
					word = ""

			outputWord.append([word])

		return outputWord, score


	def prepareSentences(self, wordList):
		outputString = " ".join(wordList)

		return outputString
	

	def sampleFirstWord(self, cleanString):
		encoderInput = self.encoderEncodeData(cleanString)
		decoderInput = self.decoderEncodeData([[""]], initial=True)
		print(self.endLangDecoding[decoderInput[0][0][0]])
		# sample first word
		outputs = self.samplingModelInit.predict_on_batch(encoderInput + decoderInput)
		wordOut = outputs[0]
		alphas = outputs[1]
		# prepare preproessed encoder for future sampling
		preprocessed = outputs[2:]
		preprocessed = [np.repeat(x, CONST.BEAM_SIZE, 0) for x in preprocessed]

		# decode predicted word
		wordSoftmax = wordOut[0,0]
		alpha = alphas[0,0]
		startingWord, initialScore = self.decodeWord(wordSoftmax, alpha, cleanString)
		
		return (startingWord, -initialScore, alpha), preprocessed[0], preprocessed[1:]									#negated scores for sorting later				


	def __call__(self, inputString):
		# prepare input sentence
		cleanString = PD.cleanText(inputString, self.startLang)
		(startingWord, cumulativeScore, firstAlpha), preprocessedEncoder, prevStates = self.sampleFirstWord(cleanString)
		
		predictedWords = startingWord
		nextDecoderInputWord = startingWord
		while len(predictedWords[0]) < CONST.MAX_TRANSLATION_LENGTH:
			# decode rest of the sentences
			decoderInput = self.decoderEncodeData(nextDecoderInputWord)
			outputs = self.samplingModelNext.predict_on_batch([preprocessedEncoder] + prevStates + decoderInput)
			wordOut = outputs[0]
			alphas = outputs[1]
			prevStates = outputs[2:]
			
			# decode all sampled words
			allScores = []
			allPredictions = []
			endOfSequence = []
			for i in range(CONST.BEAM_SIZE):
				wordSoftmax = wordOut[i,0]
				alpha = alphas[i,0]
				predictedWord, score = self.decodeWord(wordSoftmax, alpha, cleanString)
				if CONST.END_OF_SEQUENCE_TOKEN in [w for wl in predictedWord for w in wl]:
					endOfSequence.append(i)

				allScores.append(cumulativeScore[i] * score)						# cumulate scores with existing sequences
				allPredictions.append(predictedWord)

			# if any prediction contained END OF SEQUENCE, translation finished
			if endOfSequence:
				bestStringIndexFromEOS = int(np.where(cumulativeScore == np.min(cumulativeScore[endOfSequence]))[0][0])
				outputString = self.prepareSentences(predictedWords[bestStringIndexFromEOS])
				
				print(outputString) 
				#return outputString


			# get highest scoring sequences
			bestPredictions = np.argsort(np.concatenate(allScores))[:CONST.BEAM_SIZE]
			cumulativeScore = np.concatenate(allScores)[bestPredictions]

			# get best sequences and highest scoring words for next prediction 
			nextDecoderInputWord = []
			predictedWordsNext = []
			for b in bestPredictions:
				i = b // CONST.BEAM_SIZE
				j = b % CONST.BEAM_SIZE
				predictedWordsNext.append(predictedWords[i] + allPredictions[i][j])
				nextDecoderInputWord.append(allPredictions[i][j])
			# update states for next sampling according to selected best words
			stateIndices = bestPredictions // CONST.BEAM_SIZE
			prevStates = [p[stateIndices] for p in prevStates]

			predictedWords = predictedWordsNext

		bestSentenceIndex = np.argmin(cumulativeScore)
		outputString = self.prepareSentences(predictedWords[bestSentenceIndex])
		return outputString


