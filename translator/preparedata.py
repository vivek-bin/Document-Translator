# coding: utf-8

from . import constants as CONST
from .processing import fileaccess as FA
import string
import re
import json
import numpy as np
from random import shuffle
from nltk.stem import SnowballStemmer
from os.path import commonprefix
import gc

STEMMER = {}
STEMMER["en"] = SnowballStemmer("english").stem
STEMMER["fr"] = SnowballStemmer("french").stem

#reading and cleaning the data
def readData():
	epFr, epEn = FA.loadStandard(CONST.EUROPARL, ("fr", "en"))
	ccFr, ccEn = FA.loadStandard(CONST.COMMON_CRAWL, ("fr", "en"))
	pcFr, pcEn = FA.loadStandard(CONST.PARA_CRAWL, ("fr", "en"))
	haFr, haEn = FA.loadHansards()
	feFr, feEn = FA.loadFraEng()
	fr = haFr + epFr + feFr + ccFr + pcFr
	en = haEn + epEn + feEn + ccEn + pcEn

	print("text read from disk")
	print(CONST.LAPSED_TIME())

	cleanText(fr, "fr")
	cleanText(en, "en")

	print("text clean finished")
	print(CONST.LAPSED_TIME())

	return fr, en

def readLangData(lang):
	data = []
	data += FA.loadStandard(CONST.EUROPARL, (lang,))[0]
	data += FA.loadStandard(CONST.COMMON_CRAWL, (lang,))[0]
	data += FA.loadStandard(CONST.PARA_CRAWL, (lang,))[0]

	if lang == "fr":
		data += FA.loadHansards()[0]
		data += FA.loadFraEng()[0]
	elif lang == "en":
		data += FA.loadHansards()[1]
		data += FA.loadFraEng()[1]

	print(lang, "text read from disk")
	print(CONST.LAPSED_TIME())

	cleanText(data, lang)
	print(lang, "text clean finished")
	print(CONST.LAPSED_TIME())

	return data

def cleanText(lines, language):
	assert type(lines) is list

	for i, line in enumerate(lines):
		lines[i] = cleanLine(line, language)
		if not i%100000:
			print(language, ":", i, CONST.LAPSED_TIME())
	
def cleanLine(line, language):
	line = line.replace("’","'")
	words = re.findall(r"\w+|-+|\.+|\W", line)
	
	words = [wordPart for word in words if word.strip() for wordPart in splitWordStem(word, language)]

	return CONST.UNIT_SEP.join(words)

def splitWordStem(word, language):
	global STEMMER
	if re.search(r"\d", word):
		return [word]
	wordStem = STEMMER[language](word)
	if word.lower() == wordStem:
		return [word]
	else:
		commonPrefixLen = len(commonprefix([wordStem, word.lower()]))
		wordStemOrig = word[:commonPrefixLen] + wordStem[commonPrefixLen:]
		wordTrail = CONST.WORD_STEM_TRAIL_IDENTIFIER + word[commonPrefixLen:]
		return [wordStemOrig, wordTrail]

def unusualSentenceIndices(data):
	dataCharFreq = getCharacterFrequencies(data)
	rareChars = set([ch for ch, count in dataCharFreq.items() if count < CONST.RARE_CHAR_COUNT])
	rareCharLines = [i for i, line in enumerate(data) if set(line) & rareChars]

	longLines = [i for i,line in enumerate(data) if line.count(CONST.UNIT_SEP) >= CONST.MAX_WORDS]

	return set(rareCharLines) | set(longLines)
	

def getCharacterFrequencies(file):
	uniqueChars = {}

	for line in file:
		for ch in line.lower():
			try:
				uniqueChars[ch] += 1
			except KeyError:
				uniqueChars[ch] = 1

	return uniqueChars

def getWordFrequencies(file):
	wordDict = {}
	
	for line in file:
		words = line.lower().split(CONST.UNIT_SEP)
		for word in words:
			try:
				wordDict[word] += 1
			except KeyError:
				wordDict[word] = 1
	
	print("vocabulary size : "+str(len(wordDict.keys())))

	return wordDict

def preProcessWriteData(lang):
	data = readLangData(lang)
	removeLinesIndices = unusualSentenceIndices(data)
	FA.writeProcessedData(data, lang+"-temp")

	return set(range(len(data))) - removeLinesIndices

# write encodings and encoded data to disk
def writeAllData():
	writeLinesIndicesFr = preProcessWriteData("fr")
	gc.collect()
	writeLinesIndicesEn = preProcessWriteData("en")
	gc.collect()
	
	order = list(writeLinesIndicesFr & writeLinesIndicesEn)
	shuffle(order)

	FA.writeProcessedData(FA.readProcessedData("fr-temp"), "fr", order)
	gc.collect()
	FA.writeProcessedData(FA.readProcessedData("en-temp"), "en", order)
	gc.collect()
	print("cleaned text saved to disk, not encoded")
	print(CONST.LAPSED_TIME())

def writeEncodingFromProcessed(lang):
	data = FA.readProcessedData(lang)

	writeWordEncoding(data, lang)
	if CONST.INCLUDE_CHAR_EMBEDDING:
		writeCharEncoding(data, lang)
	
	print(lang, "encodings written to disk")

def writeWordEncoding(data, language):
	words = getWordFrequencies(data)
	encoding = [word for word,count in words.items() if count >= CONST.MIN_WORD_COUNT and not re.search(r"\d",word)]
	writeEncoding(encoding, language + "_word")

def writeCharEncoding(data, language):
	chars = getCharacterFrequencies(data)
	encoding = [ch for ch,count in chars.items() if count >= CONST.MIN_CHAR_COUNT]
	writeEncoding(encoding, language + "_char")

def writeEncoding(encoding, name):
	encoding.sort()
	encoding.insert(0,CONST.MASK_TOKEN)
	encoding.append(CONST.UNKNOWN_TOKEN)			# unknown 
	encoding.append(CONST.ALPHANUM_UNKNOWN_TOKEN)	# alphanum unknown 
	encoding.append(CONST.START_OF_SEQUENCE_TOKEN)	# start of sequence
	encoding.append(CONST.END_OF_SEQUENCE_TOKEN)	# end of sequence

	with open(CONST.ENCODINGS + name + ".json", "w") as f:
		f.write(json.dumps(encoding))

	print(name + " encoding size : "+str(len(encoding)))
	return encoding

def writeEncodedData(data, language):
	#encoded data
	wordEncoded = encodeWords(data, language)
	if CONST.INCLUDE_CHAR_EMBEDDING:
		charForwardEncoded = encodeCharsForward(data, language)
		charBackwardEncoded = encodeCharsBackward(data, language)

	print("input text encoded words   : "+str(wordEncoded.shape))
	if CONST.INCLUDE_CHAR_EMBEDDING:
		print("input text encoded char(f) : "+str(charForwardEncoded.shape))
		print("input text encoded char(b) : "+str(charBackwardEncoded.shape))
	
	if CONST.INCLUDE_CHAR_EMBEDDING:
		np.savez_compressed(CONST.PROCESSED_DATA + language + "EncodedData", encoded=wordEncoded, charForwardEncoded=charForwardEncoded, charBackwardEncoded=charBackwardEncoded)
	else:
		np.savez_compressed(CONST.PROCESSED_DATA + language + "EncodedData", encoded=wordEncoded)

	if CONST.INCLUDE_CHAR_EMBEDDING:
		return wordEncoded, charForwardEncoded, charBackwardEncoded
	else:
		return (wordEncoded,)


# encode the data
def encodeWords(data, language):
	maxSequenceLenth = max([line.count(CONST.UNIT_SEP)+1 for line in data]) + 2		#start and end of sequence
	encodedData = np.zeros((len(data), maxSequenceLenth),dtype="uint16")			#initialize zero array

	with open(CONST.ENCODINGS+language+"_word.json", "r") as f:
		encoding = {word:i for i,word in enumerate(json.load(f))}

	for i,line in enumerate(data):
		encodedData[i][0] = encoding[CONST.START_OF_SEQUENCE_TOKEN]
		for j,word in enumerate(line.lower().split(CONST.UNIT_SEP)):
			try:
				encodedData[i,j+1] = encoding[word]
			except KeyError:
				encodedData[i,j+1] = encoding[CONST.ALPHANUM_UNKNOWN_TOKEN] if re.search(r"\d", word) else encoding[CONST.UNKNOWN_TOKEN]
				
		encodedData[i][j+2] = encoding[CONST.END_OF_SEQUENCE_TOKEN]

	return encodedData

def encodeCharsForward(data, language):
	data = [CONST.UNIT_SEP.join([word[:CONST.CHAR_INPUT_SIZE] for word in line.lower().split(CONST.UNIT_SEP)]) for line in data]
	encodedData = encodeChars(data, language)
	
	return encodedData

def encodeCharsBackward(data, language):
	data = [CONST.UNIT_SEP.join([word[:-CONST.CHAR_INPUT_SIZE-1:-1] for word in line.lower().split(CONST.UNIT_SEP)]) for line in data]
	encodedData = encodeChars(data, language)
	
	return encodedData

def encodeChars(data, language):
	maxSequenceLenth = max([line.count(CONST.UNIT_SEP)+1 for line in data]) + 2					#start and end of sequence
	encodedData = np.zeros((len(data), maxSequenceLenth, CONST.CHAR_INPUT_SIZE),dtype="uint8")		#initialize zero array

	with open(CONST.ENCODINGS+language+"_char.json", "r") as f:
		encoding = {ch:i for i,ch in enumerate(json.load(f))}

	for i,line in enumerate(data):
		for j,word in enumerate(line.split(CONST.UNIT_SEP)):
			for k,ch in enumerate(word):
				try:
					encodedData[i,j+1,k] = encoding[ch]
				except KeyError:
					encodedData[i,j+1,k] = encoding[CONST.UNKNOWN_TOKEN]
				except IndexError:
					print(line, word, encoding[ch], ch, i, j, k)
					raise IndexError

	return encodedData


# functions to extract information about the data
def charExamples(en, fr):
	frChars = getCharacterFrequencies(fr)
	enChars = getCharacterFrequencies(en)

	chExamples = {}
	for ch in frChars:
		lines = []
		for i, line in enumerate(fr):
			if ch in line.lower():
				lines.append(fr[i] + en[i])
			if len(lines) > 3:
				break
		chExamples[ch] = lines

	onlyEngChars = [ch for ch in enChars if ch not in frChars]
	for ch in onlyEngChars:
		lines = []
		for i, line in enumerate(en):
			if ch in line.lower():
				lines.append(fr[i] + en[i])
			if len(lines) > 3:
				break
		chExamples[ch] = lines

	exampleList = [ch+"\n"+"\n".join(lines)+"\n\n\n" for ch, lines in chExamples.items()]
	FA.writeFile("char examples.txt",exampleList)

def writeDataDetailsToLog(fr, en):
	frChars = getCharacterFrequencies(fr)
	enChars = getCharacterFrequencies(en)
	FA.writeFile("fr char counts.txt",[ch+" : "+str(count) for ch, count in frChars.items()])
	FA.writeFile("en char counts.txt",[ch+" : "+str(count) for ch, count in enChars.items()])


	frFreq = getWordFrequencies(fr)
	enFreq = getWordFrequencies(en)
	
	frFreqList = sorted([(k,v) for k,v in frFreq.items()],key=lambda x:x[1])
	enFreqList = sorted([(k,v) for k,v in enFreq.items()],key=lambda x:x[1])
	FA.writeFile("frFreq.txt",[w+" : "+str(f) for w,f in frFreqList])
	FA.writeFile("enFreq.txt",[w+" : "+str(f) for w,f in enFreqList])

	charExamples(en, fr)
	
def main():
	writeAllData()
	writeEncodingFromProcessed("fr")
	writeEncodingFromProcessed("en")



if __name__ == "__main__":
	main()

