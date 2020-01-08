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

		self.wordGlossary = FA.readDictionaryGlossary()[self.startLang]
		self.wordDictionary = FA.readDictionaryPDF()[self.startLang]
		self.spellChecker = SpellChecker(language=endLang, distance=1)


	def encodeData(self, data, language):
		wordData = PD.encodeWords(data, language)
		encodedData = [wordData]
		if CONST.INCLUDE_CHAR_EMBEDDING:
			charForwardData = PD.encodeCharsForward(data, language)
			charBackwardData = PD.encodeCharsBackward(data, language)
			encodedData.append(charForwardData)
			encodedData.append(charBackwardData)

		return encodedData


	def encoderEncodeData(self, inputString, mergeUnk=False):
		if CONST.GLOSSARY_UNK_REPLACE:
			replWords = sorted(self.wordGlossary.keys(), key=lambda x: -len(x))
			replacedString = []
			replacedSubString = []
			for s in inputString:
				replacedWord = []
				for replWord in replWords:
					flag = True
					while flag:
						flag = False
						i = s.lower().find(replWord)
						if i >= 0:
							if i == 0 or not s[i-1].isalnum():
								if s[i+len(replWord)+1:].isalnum():
									temp = s[i:i+len(replWord)]
									replacedWord.append((i, temp))
									s = s.replace(temp, CONST.SUBSTITUTION_CHAR, 1)
									flag = True
				replacedString.append(s)

				replacedWord2 = []
				for i, substr in replacedWord[::-1]:
					replacedWord2 = [((ip if ip < i else ip + len(substr) - 1), sp) for ip, sp in replacedWord2]
					replacedWord2.append((i,substr))
				replacedWord = [x[1] for x in sorted(replacedWord2, key=lambda x:x[0])]
				replacedSubString.append(replacedWord)
			inputString = replacedString

		cleanString = PD.cleanText(inputString, self.startLang)

		if CONST.GLOSSARY_UNK_REPLACE:
			for i in range(len(cleanString)):
				s = cleanString[i]
				for w in replacedSubString[i]:
					s = s.replace(CONST.SUBSTITUTION_CHAR, w+CONST.SUBSTITUTION_CHAR, 1)
				cleanString[i] = s

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
				if CONST.GLOSSARY_UNK_REPLACE:
					try:
						word = self.wordGlossary[originalWord.replace(CONST.SUBSTITUTION_CHAR, "")]
					except KeyError:
						try:
							word = self.wordDictionary[originalWord]
						except KeyError:
							word = originalWord
				else:
					try:
						word = self.wordDictionary[originalWord]
					except KeyError:
						word = originalWord

			outputWord.append(word)

		return outputWord, score


	def normalizeScore(self, score, currentPrediction, prevAlphas):
		decodedWordCount = len(prevAlphas)
		originalWordCount = len(prevAlphas[0]) - 2

		lengthPenalty = ((5 + decodedWordCount)/(5+1))**CONST.LENGTH_PENALTY_COEFF
		
		coveragePenalty = 0
		for i in range(originalWordCount):
			alphaSum = sum([prevAlphas[j][i] for j in range(decodedWordCount)])
			alphaSum = min(alphaSum, 1.)
			coveragePenalty = coveragePenalty + np.log(alphaSum)
		coveragePenalty = CONST.COVERAGE_PENALTY_COEFF * coveragePenalty

		calculatedScore = (np.log(score) / lengthPenalty) + coveragePenalty
		return calculatedScore


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
		predictionList = wordList.split(CONST.UNIT_SEP)
		wordPrev = capitalizeFirstLetter(predictionList[0])

		for i, wordPart in enumerate(predictionList[1:], 1):
			if wordPart == CONST.END_OF_SEQUENCE_TOKEN:
				break

			if wordPart.startswith(CONST.WORD_STEM_TRAIL_IDENTIFIER):
				wordPrev = joinWordParts(wordPrev, wordPart)
				continue
			
			# capitalization conditions
			maxLikelihoodIndex = np.argmax(alphasList[i])
			originalWord = originalWords[maxLikelihoodIndex]
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


	def translate(self, inputString):
		# prepare input sentence
		cleanString, encoderInput = self.encoderEncodeData([inputString])
		cleanString = cleanString[0]		# we only decode one sentence at a time
		(wordSoftmax, firstAlpha), preprocessedEncoder, prevStates = self.sampleFirstWord(encoderInput)
		startingWord, initialScore = self.decodeWord(wordSoftmax, firstAlpha, cleanString)
		cumulativeScore = -np.log(initialScore)												# negated scores for sorting later	
		
		predictedWords = startingWord
		nextDecoderInput = startingWord
		alphasList = [[firstAlpha] for _ in range(CONST.BEAM_SIZE)]
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
			endOfSequence = 0
			for i in range(CONST.BEAM_SIZE):
				if CONST.END_OF_SEQUENCE_TOKEN in predictedWords[i].split(CONST.UNIT_SEP):
					endOfSequence = endOfSequence + 1
					alphas[i,-1] = np.zeros_like(alphas[i,-1])
					
					allScores.append(cumulativeScore[i] * np.ones((CONST.BEAM_SIZE,)))		# cumulate scores with existing sequences
					allPredictions.append([CONST.MASK_TOKEN]*CONST.BEAM_SIZE)
				else:
					predictedWord, score = self.decodeWord(wordOut[i,-1], alphas[i,-1], cleanString)
					score = self.normalizeScore(score, predictedWord, alphasList[i])

					allScores.append(cumulativeScore[i] * score)							# cumulate scores with existing sequences
					allPredictions.append(predictedWord)

			# if any prediction contained END OF SEQUENCE, translation finished
			if endOfSequence == CONST.BEAM_SIZE:
				bestSentenceIndex = np.argmin(cumulativeScore)
				outputString = self.prepareSentences(predictedWords[bestSentenceIndex], originalText=cleanString, alphasList=alphasList[bestSentenceIndex])
				
				print("End of sequence >>> " + outputString)
				return outputString


			# get highest scoring sequences
			bestPredictions = np.argsort(np.concatenate(allScores))[:CONST.BEAM_SIZE]
			cumulativeScore = np.concatenate(allScores)[bestPredictions]

			# get best sequences and highest scoring words for next prediction 
			predictedWords = [predictedWords[b//CONST.BEAM_SIZE] + CONST.UNIT_SEP + allPredictions[b//CONST.BEAM_SIZE][b%CONST.BEAM_SIZE] for b in bestPredictions]	
			alphasList = [alphasList[b//CONST.BEAM_SIZE] + [alphas[b//CONST.BEAM_SIZE, -1]] for b in bestPredictions]

			if self.modelNum == 1:
				# LSTM model : get last word, to feed to decoder for next prediction
				nextDecoderInput = [seq.split(CONST.UNIT_SEP)[-1] for seq in predictedWords]
			else:
				# TRANSFORMER model : feed current generated sequence
				nextDecoderInput = predictedWords

			# update states for next sampling according to selected best words
			stateIndices = bestPredictions // CONST.BEAM_SIZE
			prevStates = [p[stateIndices] for p in prevStates]

		completedSentences = [i for i, s in enumerate(predictedWords) if CONST.END_OF_SEQUENCE_TOKEN in s.split(CONST.UNIT_SEP)]
		if not completedSentences:
			print("sentence truncated!")
			completedSentences = list(range(len(predictedWords)))

		bestSentenceIndex = int(np.where(cumulativeScore == np.min(cumulativeScore[completedSentences]))[0][0])
		outputString = self.prepareSentences(wordList=predictedWords[bestSentenceIndex].split(CONST.UNIT_SEP), originalText=cleanString, alphasList=alphasList[bestSentenceIndex])
		return outputString

	def translateDocument(self, path):
		ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
		
		root = FA.readXMLFromDoc(path)
		paragraphs = root.findall('.//w:p', ns)

		for paragraph in paragraphs:
			rows = paragraph.findall('w:r', ns)
			for row in rows:
				rowText = row.find('w:t', ns)
				if rowText != None:
					rowText.text = "\n".join([self.translate(t) for t in rowText.text.split("\n")])
		
		FA.writeUpdatedDoc(root, path, path.split(".")[0] + "_" + self.endLang + "." + path.split(".")[1])
		return False

