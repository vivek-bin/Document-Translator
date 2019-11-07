import json
import numpy as np
from spellchecker import SpellChecker

from . import constants as CONST
from . import preparedata as PD
from .trainmodel import loadModel
from .processing import fileaccess as FA


class Translator:
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
			data.append(charForwardData)
			data.append(charBackwardData)

		return data


	def encoderEncodeData(self, inputString, mergeUnk=False):
		cleanString = PD.cleanText(inputString, self.startLang)
		data = self.encodeData(cleanString, self.startLang)

		return cleanString, data


	def decoderEncodeData(self, data):
		# data = [x for xl in data for x in xl]
		dataEncoded = self.encodeData(data, self.endLang)
		if len(data) == 1 and data[0] == "":			# translation started, getting START OF SEQUENCE 
			dataEncoded = [x[:,0:1] for x in dataEncoded]
		elif data[0].count(CONST.UNIT_SEP) == 0:		# recurrent network being used, encoding one word at a time; remove START OF SEQUENCE and END OF SEQUENCE		
			dataEncoded = [x[:,1:2] for x in dataEncoded]
		else:
			dataEncoded = [x[:,:-1] for x in dataEncoded]

		return dataEncoded


	def decodeWord(self, wordSoftmax, alpha, cleanString):
		#get output word encodings
		topPredictions = np.argsort(-wordSoftmax)[:CONST.BEAM_SIZE]
		score = wordSoftmax[topPredictions]
		originalWordParts = [CONST.START_OF_SEQUENCE_TOKEN] + cleanString.split(CONST.UNIT_SEP) + [CONST.END_OF_SEQUENCE_TOKEN]

		outputWord = []
		for wordIndex in topPredictions:
			word = self.endLangDecoding[wordIndex]
			if word in (CONST.UNKNOWN_TOKEN, CONST.ALPHANUM_UNKNOWN_TOKEN):
				encoderPosition = np.argmax(alpha)
				originalWord = originalWordParts[encoderPosition]
				try:
					word = self.wordDictionary[originalWord]
				except KeyError:
					word = originalWord

			outputWord.append(word)

		return outputWord, score


	def prepareSentences(self, wordList, originalText, alphasList, correctBadWords=False):
		print(CONST.LAPSED_TIME())
		def capitalizeFirstLetter(word):
			return word[0].upper() + word[1:]
		def capitalizeLikeOriginal(originalWord, word):
			if word.isupper():
				return word.upper()
			if word[0].isupper():
				return capitalizeFirstLetter(word)
			return word
		def joinWordParts(wordPrefix, wordSuffix):
			word = wordPrefix + wordSuffix[len(CONST.WORD_STEM_TRAIL_IDENTIFIER):]

			prefixExtra = 0
			while self.spellChecker.unknown([word]) and prefixExtra < 4:
				prefixExtra += 1
				word = wordPrefix[:-prefixExtra] + wordSuffix[len(CONST.WORD_STEM_TRAIL_IDENTIFIER):]
			if self.spellChecker.unknown([word]):
				word = wordPrefix + wordSuffix[len(CONST.WORD_STEM_TRAIL_IDENTIFIER):]

			return word
		

		noSpaceBefore = list("!*+,-./:;?)]^_}'")
		noSpaceAfter = list("#(*+-/[^_{'")
		originalWords = [CONST.START_OF_SEQUENCE_TOKEN] + originalText.split(CONST.UNIT_SEP) + [CONST.END_OF_SEQUENCE_TOKEN]


		outputString = ""
		addSpace = False
		wordPrev = capitalizeFirstLetter(wordList[0])
		for i, wordPart in enumerate(wordList[1:], 1):
			if wordPart.startswith(CONST.WORD_STEM_TRAIL_IDENTIFIER):
				wordPrev = joinWordParts(wordPrev, wordPart)
				continue
			
			# capitalization conditions
			originalWord = originalWords[np.argmax(alphasList[i])]
			wordPart = capitalizeLikeOriginal(originalWord, wordPart)
			if wordPrev in ["."]:
				wordPart = capitalizeFirstLetter(wordPart)

			# if (previous word allows space after) and (current allows space before):
			if addSpace and (wordPrev not in noSpaceBefore):
				outputString = outputString + " "
			
			addSpace = True
			if wordPrev in noSpaceAfter:
				addSpace = False

			outputString = outputString + (self.spellChecker.correction(wordPrev) if correctBadWords else wordPrev)
			wordPrev = wordPart
			
		if wordPrev not in noSpaceBefore:
			outputString = outputString + " "
		outputString = outputString + (self.spellChecker.correction(wordPrev) if correctBadWords else wordPrev)

		return outputString
	

	def sampleFirstWord(self, encoderInput):
		decoderInput = self.decoderEncodeData([""])
		
		# sample first word
		outputs = self.samplingModelInit.predict_on_batch(encoderInput + decoderInput)
		wordOut = outputs[0]
		alphas = outputs[1]
		# prepare preproessed encoder for future sampling
		preprocessed = outputs[2:]
		preprocessed = [np.repeat(x, CONST.BEAM_SIZE, 0) for x in preprocessed]

		return (wordOut[0,-1], alphas[0,-1]), preprocessed[0], preprocessed[1:]


	def __call__(self, inputString):
		# prepare input sentence
		cleanString, encoderInput = self.encoderEncodeData(inputString)
		(wordSoftmax, firstAlpha), preprocessedEncoder, prevStates = self.sampleFirstWord(encoderInput)
		startingWord, initialScore = self.decodeWord(wordSoftmax, firstAlpha, cleanString[0])
		cumulativeScore = -initialScore												# negated scores for sorting later	
		
		predictedWords = startingWord
		nextDecoderInput = startingWord
		alphasList = [[firstAlpha]] * CONST.BEAM_SIZE
		numWords = 1
		while numWords < CONST.MAX_TRANSLATION_LENGTH:
			# decode rest of the sentences
			decoderInput = self.decoderEncodeData(nextDecoderInput)
			outputs = self.samplingModelNext.predict_on_batch([preprocessedEncoder] + prevStates + decoderInput)
			wordOut = outputs[0]
			alphas = outputs[1]
			prevStates = outputs[2:]
			numWords += 1
			
			# decode all sampled words
			allScores = []
			allPredictions = []
			endOfSequence = []
			for i in range(CONST.BEAM_SIZE):
				wordSoftmax = wordOut[i,-1]
				alpha = alphas[i,-1]
				predictedWord, score = self.decodeWord(wordSoftmax, alpha, cleanString[0])
				if CONST.END_OF_SEQUENCE_TOKEN in predictedWord:
					endOfSequence.append(i)

				allScores.append(cumulativeScore[i] * score)						# cummulate scores with existing sequences
				allPredictions.append(predictedWord)

			# if any prediction contained END OF SEQUENCE, translation finished
			if endOfSequence:
				bestStringIndexFromEOS = int(np.where(cumulativeScore == np.min(cumulativeScore[endOfSequence]))[0][0])
				outputString = self.prepareSentences(predictedWords[bestStringIndexFromEOS], originalText=cleanString[0], alphasList=alphasList[bestStringIndexFromEOS])
				
				print("End of sequence >>> " + outputString)
				return outputString


			# get highest scoring sequences
			bestPredictions = np.argsort(np.concatenate(allScores))[:CONST.BEAM_SIZE]
			cumulativeScore = np.concatenate(allScores)[bestPredictions]

			# get best sequences and highest scoring words for next prediction 
			predictedWordsNext = []
			alphasListNext = []
			for b in bestPredictions:
				i, j = b // CONST.BEAM_SIZE, b % CONST.BEAM_SIZE
				predictedWordsNext.append(predictedWords[i] + CONST.UNIT_SEP + allPredictions[i][j])
				alphasListNext.append(alphasList[i] + [alphas[i, -1]])
			predictedWords = predictedWordsNext
			alphasList = alphasListNext

			if self.modelNum == 1:
				# LSTM model : get last word, to feed to decoder for next prediction
				nextDecoderInput = [seq.split(CONST.UNIT_SEP)[-1] for seq in predictedWords]
			else:
				# TRANSFORMER model : feed current generated sequence
				nextDecoderInput = predictedWords

			# update states for next sampling according to selected best words
			stateIndices = bestPredictions // CONST.BEAM_SIZE
			prevStates = [p[stateIndices] for p in prevStates]

		print("sentence truncated!")
		bestSentenceIndex = np.argmin(cumulativeScore)
		outputString = self.prepareSentences(wordList=predictedWords[bestSentenceIndex].split(CONST.UNIT_SEP), originalText=cleanString[0], alphasList=alphasList[bestSentenceIndex])
		return outputString


