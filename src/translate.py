import json
import numpy as np

from . import constants as CONST
from .trainmodel import loadModel
from .processing import preparedata as PD



class Translater:
	def __init__(self, startLang="fr", endLang="en", modelNum=1):
		self.startLang = startLang
		self.endLang = endLang
		self.modelNum = modelNum
		
		with open(CONST.ENCODINGS+self.startLang+"_word.json", "r") as f:
			self.startLangEncoding = {w:i for i,w in enumerate(json.load(f))}
		with open(CONST.ENCODINGS+self.endLang+"_word.json", "r") as f:
			self.endLangEncoding = {w:i for i,w in enumerate(json.load(f))}
		if CONST.INCLUDE_CHAR_EMBEDDING:
			with open(CONST.ENCODINGS+self.startLang+"_char.json", "r") as f:
				self.startLangCharEncoding = {w:i for i,w in enumerate(json.load(f))}
			with open(CONST.ENCODINGS+self.endLang+"_char.json", "r") as f:
				self.endLangCharEncoding = {w:i for i,w in enumerate(json.load(f))}

		_, (self.samplingModelInit, self.samplingModelNext) = loadModel(self.modelNum, loadOptimizerWeights=False)

		self.wordDictionary = self.loadDictionary()

	def loadDictionary(self):
		d = {}

		return d

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
		data = self.encodeData(data, self.endLang)
		if initial:
			data = [x[:,0] for x in data]
		elif onlyLastWord:
			data = [x[:,1] for x in data]
		else:
			data = [x[:,:-1] for x in data]
		if CONST.INCLUDE_CHAR_EMBEDDING:
			data[1] = np.reshape(data[1], (data[1].shape[0], -1))

		return data


	def decodeWord(self, wordSoftmax, alpha, cleanString):
		#get output word encodings
		topPredictions = np.argsort(-wordSoftmax)[:CONST.BEAM_SIZE]
		outputWord = []
		score = []
		for wordIndex in topPredictions:
			word = None
			try:
				word = self.endLangEncoding[wordIndex]
			except KeyError:
				pass
			if not word:
				encoderPosition = np.argmax(alpha)
				originalWord = cleanString.split(CONST.UNIT_SEP)[encoderPosition]
				try:
					word = self.wordDictionary[originalWord]
				except KeyError:
					word = originalWord

			outputWord.append([word])
			score.append(wordSoftmax[wordIndex])

		return outputWord, score


	def prepareSentences(self, wordList):
		outputString = " ".join(wordList)

		return outputString
	
	def sampleFirstWord(self, cleanString):
		encoderInput = self.encoderEncodeData(cleanString)
		decoderInput = self.decoderEncodeData([""], initial=True)
		# decode first word
		outputs = self.samplingModelInit(encoderInput + decoderInput)
		wordOut = outputs[0]
		alphas = outputs[1]
		preprocessed = outputs[2:]
		preprocessed = [np.repeat(x, CONST.BEAM_SIZE, 0) for x in preprocessed]

		wordSoftmax = wordOut[0,0]
		alpha = alphas[0,0]
		startingWord, initialScore = self.decodeWord(wordSoftmax, alpha, cleanString)
		initialScore = [-s for s in initialScore]										#negated scores for sorting later
		return (startingWord, initialScore), preprocessed					


	def __call__(self, inputString):
		# prepare input sentence
		cleanString = PD.cleanText(inputString, self.startLang)
		(startingWord, cumulativeScore), preprocessed = self.sampleFirstWord(cleanString)
		
		predictedWords = [startingWord]
		continueTranslation = False
		nextDecoderInputWord = startingWord
		while continueTranslation and len(predictedWords) < CONST.MAX_TRANSLATION_LENGTH:
			# decode rest of the sentences
			decoderInput = self.decoderEncodeData(nextDecoderInputWord)
			outputs = self.samplingModelNext(preprocessed + decoderInput)
			wordOut = outputs[0]
			alphas = outputs[1]
			
			# decode predicted word
			allScores = []
			allPredictions = []
			for i in range(CONST.BEAM_SIZE):
				wordSoftmax = wordOut[i,0]
				alpha = alphas[i,0]
				predictedWord, score = self.decodeWord(wordSoftmax, alpha, cleanString)
				allScores.append([s*cumulativeScore[i] for s in score])						
				allPredictions.append(predictedWord)
			bestPredictions = np.argsort(np.array(allScores).reshape((-1)))[:CONST.BEAM_SIZE]

			nextDecoderInputWord = []
			cumulativeScore = []
			for b in bestPredictions:
				i = b / CONST.BEAM_SIZE
				j = b % CONST.BEAM_SIZE
				nextDecoderInputWord.append(allPredictions[i][j])
				cumulativeScore.append(allScores[i][j])

			predictedWords.append(nextDecoderInputWord)
			if [True for xl in nextDecoderInputWord if CONST.END_OF_SEQUENCE_TOKEN in xl]:			# if any prediction contained END OF SEQUENCE, end translation
				break

		bestSentenceIndex = np.argmin(cumulativeScore)
		bestSentenceList = [wordStep[bestSentenceIndex] for wordStep in predictedWords]
		outputStrings = self.prepareSentences(bestSentenceList)


		return outputStrings