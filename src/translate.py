import json
import numpy as np
from spellchecker import SpellChecker

from . import constants as CONST
from . import preparedata as PD
from .trainmodel import loadModel
from .processing import fileaccess as FA


class Translater:
	def __init__(self, startLang="fr", endLang="en", modelNum=1):
		self.startLang = startLang
		self.endLang = endLang
		self.modelNum = modelNum
		
		with open(CONST.ENCODINGS+self.startLang+"_word.json", "r") as f:
			self.startLangDecoding = {i:w for i,w in enumerate(json.load(f))}
		with open(CONST.ENCODINGS+self.endLang+"_word.json", "r") as f:
			self.endLangDecoding = {i:w for i,w in enumerate(json.load(f))}
		if CONST.INCLUDE_CHAR_EMBEDDING:
			with open(CONST.ENCODINGS+self.startLang+"_char.json", "r") as f:
				self.startLangCharDecoding = {i:w for i,w in enumerate(json.load(f))}
			with open(CONST.ENCODINGS+self.endLang+"_char.json", "r") as f:
				self.endLangCharDecoding = {i:w for i,w in enumerate(json.load(f))}

		_, (self.samplingModelInit, self.samplingModelNext) = loadModel(self.modelNum, loadForTraining=False)

		self.wordDictionary = FA.readDictionaryPDF()[self.startLang]
		self.spellChecker = SpellChecker(language=endLang, distance=1)


	def encodeData(self, data, language):
		wordData = PD.encodeWords(data, language)
		data = [wordData]
		if CONST.INCLUDE_CHAR_EMBEDDING:
			charForwardData = PD.encodeCharsForward(data, language)
			charBackwardData = PD.encodeCharsBackward(data, language)
			charData = np.concatenate((charForwardData, charBackwardData), axis=2)
			data.append(charData)

		return data


	def encoderEncodeData(self, inputString, mergeUnk=False):
		cleanString = PD.cleanText(inputString, self.startLang)
		data = self.encodeData(cleanString, self.startLang)

		if mergeUnk:
			# merge unknown tokens
			unkPosList = [[i for i, w in enumerate(e) if self.startLangDecoding[w] == CONST.UNKNOWN_TOKEN] for e in data[0]]
			unitSepPosList = [[i for i, ch in enumerate(s) if ch == CONST.UNIT_SEP] for s in cleanString]

			finalCleanStrings = []
			for cleanStr, unkPos, unitSepPos in zip(cleanString, unkPosList, unitSepPosList):
				prevUnk = unkPos[0]
				for curUnk in unkPos[1:]:
					if prevUnk + 1 == curUnk:
						i = unitSepPos[prevUnk]
						cleanStr = cleanStr[:i] + " " + cleanStr[i+1:]
					prevUnk = curUnk
				finalCleanStrings.append(cleanStr)

			data = self.encodeData(finalCleanStrings, self.startLang)

		if CONST.INCLUDE_CHAR_EMBEDDING:
			data[1] = np.reshape(data[1], (data[1].shape[0], -1))
		return finalCleanStrings, data


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

			outputWord.append([word])

		return outputWord, score


	def prepareSentences(self, wordList, alphasList, correctBadWords=False):
		print(CONST.LAPSED_TIME())
		def capitalizeFirstLetter(word):
			return word[0].upper() + word[1:]
		def joinWordParts(wordPrefix, wordSuffix):
			word = wordPrefix + wordSuffix[len(CONST.WORD_STEM_TRAIL_IDENTIFIER):]

			prefixExtra = 0
			while self.spellChecker.unknown([word]) and prefixExtra < 4:
				prefixExtra += 1
				word = wordPrefix[:-prefixExtra] + wordSuffix[len(CONST.WORD_STEM_TRAIL_IDENTIFIER):]
			if self.spellChecker.unknown([word]):
				word = wordPrefix + wordSuffix[len(CONST.WORD_STEM_TRAIL_IDENTIFIER):]

			return word
		

		noSpaceBefore = list("!*+,-./:;?)]^_}")
		noSpaceAfter = list("#(*+-/[^_{")

		outputString = ""
		addSpace = False
		wordPrev = capitalizeFirstLetter(wordList[0])
		for wordPart in wordList[1:]:
			if wordPart.startswith(CONST.WORD_STEM_TRAIL_IDENTIFIER):
				wordPrev = joinWordParts(wordPrev, wordPart)
				continue
			
			# capitalization conditions
			if wordPrev in ["."]:
				wordPart = capitalizeFirstLetter(wordPart)

			# if (previous word allows space after) and (current allows space before):
			if addSpace and (wordPrev not in noSpaceBefore):
				outputString = outputString + " "
			
			addSpace = True
			if wordPrev in noSpaceAfter:
				addSpace = False

			outputString = outputString + wordPrev
			wordPrev = wordPart
		if wordPrev not in noSpaceBefore:
			outputString = outputString + " "
		outputString = outputString + self.spellChecker.correction(wordPrev) if correctBadWords else wordPrev

		return outputString
	

	def sampleFirstWord(self, encoderInput):
		decoderInput = self.decoderEncodeData([[""]], initial=True)
		
		# sample first word
		outputs = self.samplingModelInit.predict_on_batch(encoderInput + decoderInput)
		wordOut = outputs[0]
		alphas = outputs[1]
		# prepare preproessed encoder for future sampling
		preprocessed = outputs[2:]
		preprocessed = [np.repeat(x, CONST.BEAM_SIZE, 0) for x in preprocessed]

		return (wordOut[0,0], alphas[0,0]), preprocessed[0], preprocessed[1:]									#negated scores for sorting later				


	def __call__(self, inputString):
		# prepare input sentence
		cleanString, encoderInput = self.encoderEncodeData(inputString)
		(wordSoftmax, firstAlpha), preprocessedEncoder, prevStates = self.sampleFirstWord(encoderInput)
		startingWord, initialScore = self.decodeWord(wordSoftmax, firstAlpha, cleanString)
		cumulativeScore = -initialScore								#negated scores for sorting later	
		
		predictedWords = startingWord
		nextDecoderInputWord = startingWord
		alphasList = [[firstAlpha]] * CONST.BEAM_SIZE
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

				allScores.append(cumulativeScore[i] * score)						# cummulate scores with existing sequences
				allPredictions.append(predictedWord)

			# if any prediction contained END OF SEQUENCE, translation finished
			if endOfSequence:
				bestStringIndexFromEOS = int(np.where(cumulativeScore == np.min(cumulativeScore[endOfSequence]))[0][0])
				outputString = self.prepareSentences(predictedWords[bestStringIndexFromEOS], alphasList[bestStringIndexFromEOS])
				
				print("End of sequence >>> " + outputString)
				return outputString


			# get highest scoring sequences
			bestPredictions = np.argsort(np.concatenate(allScores))[:CONST.BEAM_SIZE]
			cumulativeScore = np.concatenate(allScores)[bestPredictions]

			# get best sequences and highest scoring words for next prediction 
			nextDecoderInputWord = []
			predictedWordsNext = []
			alphasListNext = []
			for b in bestPredictions:
				i, j = b // CONST.BEAM_SIZE, b % CONST.BEAM_SIZE
				predictedWordsNext.append(predictedWords[i] + allPredictions[i][j])
				nextDecoderInputWord.append(allPredictions[i][j])
				alphasListNext.append(alphasList[i] + [alphas[i, 0]])
			# update states for next sampling according to selected best words
			stateIndices = bestPredictions // CONST.BEAM_SIZE
			prevStates = [p[stateIndices] for p in prevStates]

			predictedWords = predictedWordsNext
			alphasList = alphasListNext

		bestSentenceIndex = np.argmin(cumulativeScore)
		outputString = self.prepareSentences(predictedWords[bestSentenceIndex], alphasList[bestSentenceIndex])
		return outputString


