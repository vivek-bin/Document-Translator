# coding: utf-8

import numpy
import constants as CONST
import fileaccess as FA
import string
import re
import json
import numpy as np

def readEuroParlAndHansards():
	en = []
	fr = []

	epFr, epEn = readData(FA.loadEuroParl)
	fr = fr + epFr
	en = en + epEn

	hFr, hEn = readData(FA.loadHansards)
	fr = fr + hFr
	en = en + hEn



	fr, en = cleanRareChars(fr, en)
	return fr, en


def readData(loadFunc):
	fr, en = loadFunc()

	fr = cleanText(fr)
	en = cleanText(en)

	fr, en = limitSentenceSize(fr, en)
	return fr, en
	

def cleanText(lines):
	linesOut = []
	for line in lines:
		line = line.lower().replace("’","'")
		lineOut = cleanLine(line)
		linesOut.append(lineOut)
		
	return linesOut
	

def cleanLine(line):
	words = [w for s in re.findall(r'\b(?=\w*?\d)\w+(?:[\W_](?=\w*?\d)\w+)*|[^\W\d_]+|[\W_]', line) for w in re.findall(r'\s+|\S+',s)]
	words = [word for word in words if word.strip()]

	return CONST.UNIT_SEP.join(words)
	

def limitSentenceSize(fileFr,fileEn):
	maxWordsFr = [i for i,line in enumerate(fileFr) if line.count(CONST.UNIT_SEP) >= CONST.MAX_WORDS]
	maxWordsEn = [i for i,line in enumerate(fileEn) if line.count(CONST.UNIT_SEP) >= CONST.MAX_WORDS]
	maxWordLines = set(maxWordsFr) | set(maxWordsEn)

	fileFr2 = [line for i, line in enumerate(fileFr) if i not in maxWordLines]
	fileEn2 = [line for i, line in enumerate(fileEn) if i not in maxWordLines]
	
	return fileFr2, fileEn2

	
def getCharecters(file):
	uniqueChars = {}
	for line in file:
		for ch in line:
			if ch in uniqueChars:
				uniqueChars[ch] += 1
			else:
				uniqueChars[ch] = 1


	#uniqueChars = [str(ord(c)).zfill(6) for c in uniqueChars]
	return uniqueChars
	

def getWordFrequencies(file):
	wordDict = {}
	
	for line in file:
		words = line.split(CONST.UNIT_SEP)
		for word in words:
			if word in wordDict:
				wordDict[word] += 1
			else:
				wordDict[word] = 1
	
	print("vocabulary size : "+str(len(wordDict.keys())))

	return wordDict
	

def cleanRareChars(fr, en):
	frChars = getCharecters(fr)
	enChars = getCharecters(en)

	frCharsRare = [ch for ch, count in frChars.items() if count < CONST.MIN_CHAR_COUNT]
	enCharsRare = [ch for ch, count in enChars.items() if count < CONST.MIN_CHAR_COUNT]
	rareChars = set(frCharsRare) & set(enCharsRare)
	removeLinesNumbers = set([i for i, line in enumerate(fr) if set(line) & rareChars] + [i for i, line in enumerate(en) if set(line) & rareChars])
	fr = [l for i,l in enumerate(fr) if i not in removeLinesNumbers]
	en = [l for i,l in enumerate(en) if i not in removeLinesNumbers]

	print("size of data : " +str(len(fr)))
	return fr, en
	

def charExamples(en, fr):
	frChars = getCharecters(fr)
	enChars = getCharecters(en)

	chExamples = {}
	for ch in frChars:
		lines = []
		for i, line in enumerate(fr):
			if ch in line:
				lines.append(fr[i] + en[i])
			if len(lines) > 3:
				break
		chExamples[ch] = lines

	onlyEngChars = [ch for ch in enChars if ch not in frChars]
	for ch in onlyEngChars:
		lines = []
		for i, line in enumerate(en):
			if ch in line:
				lines.append(fr[i] + en[i])
			if len(lines) > 3:
				break
		chExamples[ch] = lines

	exampleList = [ch+"\n"+"\n".join(lines)+"\n\n\n" for ch, lines in chExamples.items()]
	FA.writeFile("char examples.txt",exampleList)


def logDataDetails(fr, en):
	frChars = getCharecters(fr)
	enChars = getCharecters(en)
	FA.writeFile("fr char counts.txt",[ch+" : "+str(count) for ch, count in frChars.items()])
	FA.writeFile("en char counts.txt",[ch+" : "+str(count) for ch, count in enChars.items()])


	frFreq = getWordFrequencies(fr)
	enFreq = getWordFrequencies(en)
	
	frFreqList = sorted([(k,v) for k,v in frFreq.items()],key=lambda x:x[1])
	enFreqList = sorted([(k,v) for k,v in enFreq.items()],key=lambda x:x[1])
	FA.writeFile("frFreq.txt",[w+" : "+str(f) for w,f in frFreqList])
	FA.writeFile("enFreq.txt",[w+" : "+str(f) for w,f in enFreqList])

	charExamples(en, fr)
	


def writeEncodings():
	fr, en = readEuroParlAndHansards()


	encoding = wordEncoding(fr)
	with open(CONST.ENCODING_PATH+"fr_word.json", "w") as f:
		f.write(json.dumps(encoding))
	encoding = charEncoding(fr)
	with open(CONST.ENCODING_PATH+"fr_char.json", "w") as f:
		f.write(json.dumps(encoding))


	encoding = wordEncoding(en)
	with open(CONST.ENCODING_PATH+"en_word.json", "w") as f:
		f.write(json.dumps(encoding))
	encoding = charEncoding(en)
	with open(CONST.ENCODING_PATH+"en_char.json", "w") as f:
		f.write(json.dumps(encoding))



def wordEncoding(data):
	words = getWordFrequencies(data)
	encoding = [word for word,count in words.items() if count >= CONST.MIN_WORD_COUNT and not re.match(r".*?\d",word)]

	encoding.insert(0,"MASK")
	encoding.append("UNK")
	
	print("word encoding size : "+str(len(encoding)))
	return encoding


def charEncoding(data):
	chars = getCharecters(data)
	encoding = [ch for ch,count in chars.items() if count >= CONST.MIN_WORD_COUNT]

	encoding.insert(0,"MASK")
	encoding.append("UNK")

	print("char encoding size : "+str(len(encoding)))
	return encoding


def createCharInput(input):
	forwardCharInput = [CONST.UNIT_SEP.join([word[:CONST.CHAR_INPUT_SIZE] for word in line.split(CONST.UNIT_SEP)]) for line in input]
	backwardCharInput = [CONST.UNIT_SEP.join([word[:-CONST.CHAR_INPUT_SIZE-1:-1] for word in line.split(CONST.UNIT_SEP)]) for line in input]

	return forwardCharInput, backwardCharInput


def loadDataFrToEng():
	#load main data
	fr, en = readEuroParlAndHansards()


	fr = [line.split(CONST.UNIT_SEP) for line in fr]
	en = [line.split(CONST.UNIT_SEP) for line in en]
	
	#create char level data for each word
	frCharForward, frCharBackward = createCharInput(fr)


	#encoded data
	frEncoded = encodeWords(fr, "fr")
	enEncoded = encodeWords(en, "en")
	frCharForwardEncoded = encodeChars(frCharForward,"fr")
	frCharBackwardEncoded = encodeChars(frCharBackward,"fr")

	print("input text encoded words : "+str(frEncoded.shape))
	print("input text encoded char(f) : "+str(frCharForwardEncoded.shape))
	print("input text encoded char(b) : "+str(frCharBackwardEncoded.shape))
	print("output text encoded words : "+str(enEncoded.shape))
	
	return (frEncoded, frCharForwardEncoded, frCharBackwardEncoded), enEncoded



def encodeWords(data,encodingName):
	encodedData = np.zeros((len(data), CONST.MAX_WORDS))			#initialize zero array


	with open(CONST.ENCODING_PATH+encodingName+"_word.json", "r") as f:
		encoding = json.load(f)
	UNKEncoding = len(encoding) - 1
	encoding = {word:i for i,word in enumerate(encoding)}


	for i,line in enumerate(data):
		for j,word in enumerate(line.split(CONST.UNIT_SEP)):
			try:
				encodedData[i][j] = encoding[word]
			except KeyError:
				encodedData[i][j] = UNKEncoding

	return encodedData



def encodeChars(data,encodingName):
	encodedData = np.zeros((len(data), CONST.MAX_WORDS,CONST.CHAR_INPUT_SIZE))			#initialize zero array


	with open(CONST.ENCODING_PATH+encodingName+"_char.json", "r") as f:
		encoding = json.load(f)
	UNKEncoding = len(encoding) - 1
	encoding = {ch:i for i,ch in enumerate(encoding)}


	for i,line in enumerate(data):
		for j,word in enumerate(line.split(CONST.UNIT_SEP)):
			for k,ch in enumerate(word):
				try:
					encodedData[i][j][k] = encoding[ch]
				except KeyError:
					encodedData[i][j][k] = UNKEncoding

	return encodedData




def main():
	writeEncodings()
	loadDataFrToEng()







if __name__ == "__main__":
	main()

