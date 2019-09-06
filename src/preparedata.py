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

STEMMER = {}
STEMMER["en"] = SnowballStemmer("english").stem
STEMMER["fr"] = SnowballStemmer("french").stem

#reading and cleaning the data
def readData():
	epFr, epEn = FA.loadStandard(CONST.EUROPARL)
	ccFr, ccEn = FA.loadStandard(CONST.COMMON_CRAWL)
	haFr, haEn = FA.loadHansards()
	feFr, feEn = FA.loadFraEng()
	fr = haFr + epFr + feFr + ccFr
	en = haEn + epEn + feEn + ccEn

	frEn = list(zip(fr,en))
	shuffle(frEn)
	fr = [frLine for frLine, enLine in frEn]
	en = [enLine for frLine, enLine in frEn]
	
	print("text read from disk")
	print(CONST.LAPSED_TIME())
	return fr, en

def cleanText(lines, language):
	assert type(lines) is list

	for i, line in enumerate(lines):
		lines[i] = cleanLine(line, language)
		
	return lines

def cleanLine(line, language):
	line = line.replace("’","'")
	words = [w for s in re.findall(r'\b(?=\w*?\d)\w+(?:[\W_](?=\w*?\d)\w+)*|[^\W\d_]+|[\W_]', line) for w in re.findall(r'\s+|\S+',s)]
	
	words = [wordPart for word in words if word.strip() for wordPart in splitWordStem(word, language)]

	return CONST.UNIT_SEP.join(words)

def splitWordStem(word, language):
	global STEMMER
	wordStem = STEMMER[language](word)
	if word.lower() == wordStem:
		return [word]
	else:
		commonPrefixLen = len(commonprefix([wordStem, word.lower()]))
		wordStemOrig = word[:commonPrefixLen] + wordStem[commonPrefixLen:]
		wordTrail = CONST.WORD_STEM_TRAIL_IDENTIFIER + word[commonPrefixLen:]
		return [wordStemOrig, wordTrail]

def limitSentenceSize(fileFr,fileEn):
	maxWordsFr = [i for i,line in enumerate(fileFr) if line.count(CONST.UNIT_SEP) >= CONST.MAX_WORDS]
	maxWordsEn = [i for i,line in enumerate(fileEn) if line.count(CONST.UNIT_SEP) >= CONST.MAX_WORDS]
	maxWordLines = set(maxWordsFr) | set(maxWordsEn)

	fileFr2 = [line for i, line in enumerate(fileFr) if i not in maxWordLines]
	fileEn2 = [line for i, line in enumerate(fileEn) if i not in maxWordLines]
	
	return fileFr2, fileEn2

def cleanRareChars(fr, en):
	frChars = getCharacterFrequencies(fr)
	enChars = getCharacterFrequencies(en)

	frCharsRare = [ch for ch, count in frChars.items() if count < CONST.RARE_CHAR_COUNT]
	enCharsRare = [ch for ch, count in enChars.items() if count < CONST.RARE_CHAR_COUNT]
	rareChars = set(frCharsRare) & set(enCharsRare)
	removeLinesNumbers = set([i for i, line in enumerate(fr) if set(line) & rareChars] + [i for i, line in enumerate(en) if set(line) & rareChars])
	fr = [l for i,l in enumerate(fr) if i not in removeLinesNumbers]
	en = [l for i,l in enumerate(en) if i not in removeLinesNumbers]

	print("size of data : " +str(len(fr)))
	return fr, en
	

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


# write encodings and encoded data to disk
def writeEncodingsData(encodeDataToNumpy=False):
	fr, en = readData()

	fr = cleanText(fr, "fr")
	en = cleanText(en, "en")

	print("text clean finished")
	print(CONST.LAPSED_TIME())

	fr, en = limitSentenceSize(fr, en)
	fr, en = cleanRareChars(fr, en)

	print("text ready for encoding")
	print(CONST.LAPSED_TIME())

	# write word encoding to file
	writeWordEncoding(fr, "fr")
	writeWordEncoding(en, "en")
	if CONST.INCLUDE_CHAR_EMBEDDING:
		# write character encoding to file
		writeCharEncoding(fr, "fr")
		writeCharEncoding(en, "en")
	
	if encodeDataToNumpy:
		writeEncodedData(fr, "fr")
		writeEncodedData(en, "en")
		print("encoding and encoded text saved to disk")
	else:
		FA.writeProcessedData(fr, "fr")
		FA.writeProcessedData(en, "en")
		print("cleaned text saved to disk, not encoded")
	print(CONST.LAPSED_TIME())

def writeWordEncoding(data, language):
	words = getWordFrequencies(data)
	encoding = [word for word,count in words.items() if count >= CONST.MIN_WORD_COUNT and not re.match(r".*?\d",word)]
	writeEncoding(encoding, language + "_word")

def writeCharEncoding(data, language):
	chars = getCharacterFrequencies(data)
	encoding = [ch for ch,count in chars.items() if count >= CONST.MIN_CHAR_COUNT]
	writeEncoding(encoding, language + "_char")

def writeEncoding(encoding, name):
	encoding.sort()
	encoding.insert(0,CONST.MASK_TOKEN)
	encoding.append(CONST.UNKNOWN_TOKEN)			# unknown 
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
				encodedData[i][j+1] = encoding[word]
			except KeyError:
				encodedData[i][j+1] = encoding[CONST.UNKNOWN_TOKEN]
		encodedData[i][j+2] = encoding[CONST.END_OF_SEQUENCE_TOKEN]

	return encodedData

def encodeCharsForward(data, language):
	data = [CONST.UNIT_SEP.join([word[:CONST.CHAR_INPUT_SIZE] for word in line.split(CONST.UNIT_SEP)]) for line in data]
	encodedData = encodeChars(data, language)
	
	return encodedData

def encodeCharsBackward(data, language):
	data = [CONST.UNIT_SEP.join([word[:-CONST.CHAR_INPUT_SIZE-1:-1] for word in line.split(CONST.UNIT_SEP)]) for line in data]
	encodedData = encodeChars(data, language)
	
	return encodedData

def encodeChars(data, language):
	maxSequenceLenth = max([line.count(CONST.UNIT_SEP)+1 for line in data]) + 2					#start and end of sequence
	encodedData = np.zeros((len(data), maxSequenceLenth, CONST.CHAR_INPUT_SIZE),dtype="uint8")		#initialize zero array

	with open(CONST.ENCODINGS+language+"_char.json", "r") as f:
		encoding = {ch:i for i,ch in enumerate(json.load(f))}

	for i,line in enumerate(data):
		for j,word in enumerate(line.lower().split(CONST.UNIT_SEP)):
			for k,ch in enumerate(word):
				try:
					encodedData[i][j+1][k] = encoding[ch]
				except KeyError:
					encodedData[i][j+1][k] = encoding[CONST.UNKNOWN_TOKEN]

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
	writeEncodingsData()



if __name__ == "__main__":
	main()

