import json
import numpy as np
import re
from spellchecker import SpellChecker

from . import constants as CONST
from . import preparedata as PD
from .processing import projextract as PE
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
								if not s[i+len(replWord):][0:1].isalnum():
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
					s = s.replace(CONST.SUBSTITUTION_CHAR, w+CONST.SUBSTITUTION_CHAR_2, 1)
				cleanString[i] = s

		encodedData = []
		encodedData.append(PD.encodeWords(cleanString, self.startLang))
		if CONST.INCLUDE_CHAR_EMBEDDING:
			encodedData.append(PD.encodeCharsForward(cleanString, self.startLang))
			encodedData.append(PD.encodeCharsBackward(cleanString, self.startLang))

		return cleanString, encodedData

	def decoderStartOfSequence(self):
		encodedData = []
		encodedData.append(PD.encodeWords([""], self.endLang)[:,:1])
		if CONST.INCLUDE_CHAR_EMBEDDING:
			encodedData.append(PD.encodeCharsForward([""], self.endLang)[:,:1])
			encodedData.append(PD.encodeCharsBackward([""], self.endLang)[:,:1])

		return encodedData

	def decoderEncodeData(self, data, prevInput, prevOutput):
		encodedData = []
		latestWords = np.expand_dims(np.array(prevOutput), 1)
		encodedData.append(np.concatenate((prevInput[0], latestWords), axis=1))
		if CONST.INCLUDE_CHAR_EMBEDDING:
			latestFChars = PD.encodeCharsForward(data, self.endLang)[:,-2:-1]
			encodedData.append(np.concatenate((prevInput[1], latestFChars), axis=1))
			latestBChars = PD.encodeCharsBackward(data, self.endLang)[:,-2:-1]
			encodedData.append(np.concatenate((prevInput[2], latestBChars), axis=1))

		print(encodedData[0].shape, encodedData[1].shape, encodedData[2].shape)

		if self.modelNum == 1:								# recurrent network being used, only fetch latest decoded word, or SOS at start
			encodedData = [x[:,-1:] for x in encodedData]
		
		return encodedData


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
						word = self.wordGlossary[originalWord.replace(CONST.SUBSTITUTION_CHAR_2, "").lower()]
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

		calculatedScore = -(np.log(score) / lengthPenalty) + coveragePenalty
		return calculatedScore


	def prepareSentences(self, wordList, originalText, alphasList, correctBadWords=False):
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
	

	def sampleFirstWord(self, encoderInput, decoderInput):
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
		decoderInput = self.decoderStartOfSequence()
		(wordOut, firstAlpha), preprocessedEncoder, prevStates = self.sampleFirstWord(encoderInput, decoderInput)
		startingWord, initialScore = self.decodeWord(wordOut, firstAlpha, cleanString)
		cumulativeScore = -np.log(initialScore)												# negated scores for sorting later	
		
		predictedWords = startingWord
		alphasList = [[firstAlpha] for _ in range(CONST.BEAM_SIZE)]
		prevInput = [np.repeat(x, CONST.BEAM_SIZE, 0) for x in decoderInput]
		prevOutput = np.argsort(-wordOut)[:CONST.BEAM_SIZE]
		for _ in range(CONST.MAX_TRANSLATION_LENGTH - 1):
			# decode rest of the sentences
			decoderInput = self.decoderEncodeData(predictedWords, prevInput, prevOutput)
			outputs = self.samplingModelNext.predict_on_batch([preprocessedEncoder] + prevStates + decoderInput)
			wordOut = outputs[0]
			alphas = outputs[1]
			prevStates = outputs[2:]
			
			# decode all sampled words
			allScores = []
			allPredictions = []
			allBeamsEOS = True
			for i in range(CONST.BEAM_SIZE):
				if CONST.END_OF_SEQUENCE_TOKEN in predictedWords[i].split(CONST.UNIT_SEP):
					alphas[i,-1] = np.zeros_like(alphas[i,-1])
					
					allScores.append(cumulativeScore[i] * np.ones((CONST.BEAM_SIZE,)))		# cumulate scores with existing sequences
					allPredictions.append([CONST.MASK_TOKEN]*CONST.BEAM_SIZE)
				else:
					allBeamsEOS = False
					predictedWord, score = self.decodeWord(wordOut[i,-1], alphas[i,-1], cleanString)
					score = self.normalizeScore(score, predictedWord, alphasList[i])

					allScores.append(cumulativeScore[i] + score)							# cumulate scores with existing sequences
					allPredictions.append(predictedWord)

			# if all predictions contained END OF SEQUENCE, translation finished
			if allBeamsEOS:
				break

			# get highest scoring sequences
			bestPredictions = np.argsort(np.concatenate(allScores))[:CONST.BEAM_SIZE]
			cumulativeScore = np.concatenate(allScores)[bestPredictions]
			bestBeams = bestPredictions // CONST.BEAM_SIZE
			bestWordOfBeam = bestPredictions % CONST.BEAM_SIZE

			# get best sequences and highest scoring words for next prediction 
			predictedWords = [predictedWords[b] + CONST.UNIT_SEP + allPredictions[b][w] for b, w in zip(bestBeams, bestWordOfBeam)]	
			alphasList = [alphasList[b] + [alphas[b, -1]] for b in bestBeams]
			prevStates = [prevState[bestBeams] for prevState in prevStates]
			prevInput = [x[bestBeams] for x in decoderInput]
			prevOutput = [np.argsort(-wordOut[b][-1])[w] for b, w in zip(bestBeams, bestWordOfBeam)]

		completedSentences = [i for i, s in enumerate(predictedWords) if CONST.END_OF_SEQUENCE_TOKEN in s.split(CONST.UNIT_SEP)]
		if not completedSentences:
			print("sentence truncated!")
			completedSentences = list(range(CONST.BEAM_SIZE))

		bestSentenceIndex = int(np.where(cumulativeScore == np.min(cumulativeScore[completedSentences]))[0][0])
		outputString = self.prepareSentences(wordList=predictedWords[bestSentenceIndex], originalText=cleanString, alphasList=alphasList[bestSentenceIndex])
		print("End of sequence >>> " + outputString)
		return outputString

	def updateTextTags(self, textTags, text):
		originalText = PE.joinXMLTextTags(textTags)
		relativePosFactor = len(text) / len(originalText)
		for tag in textTags:
			if not tag.text:
				continue
			if not tag.text.strip():
				if not text[:len(tag.text)].strip():
					text = text[len(tag.text):]
				continue
			if len(tag.text.strip()) == 1:
				for i, c in enumerate(tag.text):
					if c.strip():
						break

				tag.text = text[:i+1]
				text = text[i+1:]
				continue
			if not text:
				tag.text = ""
			l = int(round(len(tag.text) * relativePosFactor)) or 1
			tag.text = text[:l]
			text = text[l:]

		if text:
			textTags[-1].text = textTags[-1].text + text
	
	def translateDocument(self, path):
		print(CONST.LAPSED_TIME())
		tree = FA.readXMLFromDoc(path)
		root = tree.getroot()

		for textTags in PE.getXMLTextBlocks(root):
			original = PE.joinXMLTextTags(textTags)
			if re.search("[a-zA-Z]", original):
				translation = []
				for text in original.split("\n"):
					if re.search("[a-zA-Z]", text):
						translation.append(self.translate(text))
					else:
						translation.append(text)
				self.updateTextTags(textTags, "\n".join(translation))
		
		FA.writeUpdatedDoc(tree, path, path.split(".")[0] + "_" + self.endLang + "." + path.split(".")[1])
		return False

