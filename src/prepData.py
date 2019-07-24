import numpy
import constants as CONST
import fileaccess as FA


def readEuroParlAndHansards():
	epFr,epEn = readData(FA.loadEuroParl)
	hFr,hEn = readData(FA.loadHansards)

	return epFr+hFr, epEn+hEn

def readData(loadFunc):
	fr, en = loadFunc()
	
	#fr = cleanText(fr)
	#en = cleanText(en)
	
	fr, en = limitSentenceSize(fr, en)
	
	# frChars = getCharecters(fr)
	# enChars = getCharecters(en)
	# print("chars fr : " + str(len(frChars)) + "\n" + "".join(frChars))
	# print("chars en : " + str(len(enChars)) + "\n" + "".join(enChars))

	return fr, en
	

def cleanText(lines):
	padSpacesBoth = " .,'\"!?@#[]():;/\\«»-’" 
	# wordBreakCondLeft = ""
	# wordBreakCondRight = "’"
	linesOut = []
	for line in lines:
		line = line.lower()

		for j,ch in enumerate(line):
			if ch in padSpacesBoth:
				line = line[:j] + " " + ch + " " + line[j+1:]
		
		linesOut.append(line.split())
		
	return linesOut
	

def limitSentenceSize(fileFr,fileEn):	
	fileEn2 = []
	fileFr2 = []
	if len(fileFr) != len(fileEn):
		print("files not of same number of sentences!")
		return fileFr2, fileEn2
	
	for i in range(len(fileFr)):
		if fileFr[i].count(" ") > CONST.MAX_WORDS or fileEn[i].count(" ") > CONST.MAX_WORDS:
			fileFr.pop(i)
			fileEn.pop(i)
			
	return fileFr, fileEn

	
def getCharecters(file):
	uniqueChars = set()
	
	for line in file:
		for word in line:
			uniqueChars.update(word)
	
	#uniqueChars = [str(ord(c)).zfill(6) for c in uniqueChars]
	return sorted(uniqueChars)
	

def getWordFrequencies(file):
	wordDict = {}
	
	for line in file:
		for word in line:
			if word in wordDict:
				wordDict[word] += 1
			else:
				wordDict[word] = 1
	
	return wordDict
	
	
	
if __name__ == "__main__":
	fr, en = readEuroParlAndHansards()
	frFreq = getWordFrequencies(fr)
	enFreq = getWordFrequencies(en)
	
	frFreqList = [(k,v) for k,v in frFreq.items()]
	enFreqList = [(k,v) for k,v in enFreq.items()]

	frFreqList.sort(key=lambda x:x[1])
	enFreqList.sort(key=lambda x:x[1])

	FA.writeFile("frFreq.txt",[w+" : "+str(f) for w,f in frFreqList])
	FA.writeFile("enFreq.txt",[w+" : "+str(f) for w,f in enFreqList])

	print(frFreqList[0])
	print(enFreqList[0])

	print(frFreqList[-5])
	print(enFreqList[-5])

	# frChars = getCharecters(fr)
	# enChars = getCharecters(en)
	# frCharDict = {c:ord(c) for c in frChars}
	# enCharDict = {c:ord(c) for c in enChars}
	
	# print("".join(sorted(frCharDict.keys())))
	# print("".join(sorted(enCharDict.keys())))
	
	




