# coding: utf-8

import numpy
import constants as CONST
import fileaccess as FA
import string
import re

def readEuroParlAndHansards():
	epFr,epEn = readData(FA.loadEuroParl)
	hFr,hEn = readData(FA.loadHansards)

	return epFr+hFr, epEn+hEn


def readData(loadFunc):
	fr, en = loadFunc()

	fr = cleanTextFr(fr)
	en = cleanTextEn(en)

	fr, en = limitSentenceSize(fr, en)
	return fr, en
	

def cleanTextEn(lines):
	linesOut = []
	for line in lines:
		line = line.lower().replace("’","'")
		lineOut = cleanLineEn(line)
		linesOut.append(lineOut)
		
	return linesOut
	

def cleanLineEn(line):
	words = re.split(r"((?<=\D)[ \W]|[ \W](?=\D)| )",line)
	words = [word for word in words if word]


	# quotePos = [i for i,word in enumerate(words) if word == "'"]
	# apostopheS = [i for i in quotePos if words[i+1] == "s"]
	# try:
	# 	apostopheBlankS = [i for i in quotePos if words[i+1] == " " and words[i+2] == "s"]
	# except IndexError:
	# 	print("1Index error while cleaning apostophes : " + line)
	# try:
	# 	SApostophe = [i for i in quotePos if words[i-1][-1] == "s" and not(words[i+1] == "s") and not(words[i+1] == " " and words[i+2] == "s")]
	# except IndexError:
	# 	print(words)
	# 	print(quotePos)
	# 	print("2Index error while cleaning apostophes : " + line)

	# for i in apostopheS:
	# 	words[i:i+1+1] = [words[i]+words[i+1]]
	# for i in apostopheBlankS:
	# 	words[i:i+2+1] = [words[i]+words[i+2]]
	# for i in SApostophe:
	# 	words[i:i+1+1] = [words[i]+words[i+2]]

	words = [w for w in words if w.strip()]
	return CONST.UNIT_SEP.join(words)
	

def cleanTextFr(lines):
	linesOut = []
	for line in lines:
		line = line.lower().replace("«",'"').replace("»",'"').replace("’","'")
		lineOut = cleanLineFr(line)
		linesOut.append(lineOut)
		
	return linesOut
	

def cleanLineFr(line):
	words = re.split(r"((?<=\D)[ \W]|[ \W](?=\D)| )",line)
	words = [word for word in words if word]
	
	words = [w for w in words if w.strip()]
	return CONST.UNIT_SEP.join(words)


def limitSentenceSize(fileFr,fileEn):
	if len(fileFr) != len(fileEn):
		print("files not of same number of sentences!")
		return fileFr, fileEn
	
	fileFr2 = []
	fileEn2 = []
	numSamples = len(fileFr)
	for i in range(numSamples):
		if len(fileFr[i].split(CONST.UNIT_SEP)) > CONST.MAX_WORDS:
			continue
		elif len(fileEn[i].split(CONST.UNIT_SEP)) > CONST.MAX_WORDS:
			continue
		else:
			fileFr2.append(fileFr[i])
			fileEn2.append(fileEn[i])
	
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
	minCharCount = 15

	frChars = getCharecters(fr)
	enChars = getCharecters(en)

	frCharsRare = [ch for ch, count in frChars.items() if count < minCharCount]
	enCharsRare = [ch for ch, count in enChars.items() if count < minCharCount]
	rareChars = set(frCharsRare + enCharsRare)
	removeLinesNumbers = set([i for i, line in enumerate(fr) if set(line) & rareChars] + [i for i, line in enumerate(en) if set(line) & rareChars])
	fr = [l for i,l in enumerate(fr) if i not in removeLinesNumbers]
	en = [l for i,l in enumerate(en) if i not in removeLinesNumbers]

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



def main():
	fr, en = readEuroParlAndHansards()
	fr, en = cleanRareChars(fr, en)

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

	print("size of data : " +str(len(fr)))



if __name__ == "__main__":
	main()

