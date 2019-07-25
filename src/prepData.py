import numpy
import constants as CONST
import fileaccess as FA
import string


def readEuroParlAndHansards():
	epFr,epEn = readData(FA.loadEuroParl)
	hFr,hEn = readData(FA.loadHansards)

	return epFr+hFr, epEn+hEn

def readData(loadFunc):
	fr, en = loadFunc()
	fr, en = limitSentenceSize(fr, en)
	
	# frChars = getCharecters(fr)
	# enChars = getCharecters(en)
	# print("chars fr : " + str(len(frChars)) + "\n" + "".join(frChars))
	# print("chars en : " + str(len(enChars)) + "\n" + "".join(enChars))

	return fr, en
	

def cleanText(lines):
	linesOut = []
	for line in lines:
		lineOut = cleanLine(line.lower())
		linesOut.append(lineOut)
		
	return linesOut
	

def cleanLine(line):
	sentenceBreaks = set(".?!")
	wordBreaks = sentenceBreaks | set(" ,:;\"")
	
	allPunctuations = set(string.punctuation)



	line = line.lower()
	words = []
	word = ""
	for ch in line:
		if ch in wordBreaks:
			if word:
				words.append(word)
			words.append(ch)
			word = ""
		else:
			word = word + ch
	return words


def limitSentenceSize(fileFr,fileEn):
	if len(fileFr) != len(fileEn):
		print("files not of same number of sentences!")
		return fileFr, fileEn
	
	fileFr2 = []
	fileEn2 = []
	numSamples = len(fileFr)
	for i in range(numSamples):
		if len(fileFr[i].split()) > CONST.MAX_WORDS:
			continue
		elif len(fileEn[i].split()) > CONST.MAX_WORDS:
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
		words = line.split()
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

