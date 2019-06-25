import numpy
import constants as CONST
import fileaccess as FA

MAX_WORDS = 60

def readData():
	epFr,epEn = FA.loadEuroParl()
	hFr,hEn = FA.loadHansards()
	
	epFr = cleanText(epFr)
	epEn = cleanText(epEn)
	hFr = cleanText(hFr)
	hEn = cleanText(hEn)
	
	epFr,epEn = limitSentenceSize(epFr, epEn)
	hFr,hEn = limitSentenceSize(hFr, hEn)
	
	return epFr+hFr, epEn+hEn
	

def cleanText(lines):
	linesOut = []
	for line in lines:
		line = line.replace("."," . ").replace(","," , ").replace("'"," ' ").replace("!"," ! ").replace("?"," ? ")
		linesOut.append(line)
		
	return linesOut
	

def limitSentenceSize(fileFr,fileEn):
	global MAX_WORDS
	
	fileEn2 = []
	fileFr2 = []
	if len(fileFr) != len(fileEn):
		print("files not of same number of sentences!")
		return fileFr2, fileEn2
	
	for i in range(len(fileFr)):
		if len(fileFr[i].split()) < MAX_WORDS and len(fileEn[i].split()) < MAX_WORDS:
			fileFr2.append(fileFr[i])
			fileEn2.append(fileEn[i])
			
	return fileFr2, fileEn2

	
def getCharecters(file):
	uniqueChars = set()
	
	for line in file:
		uniqueChars.update(list(line))
	
	#return sorted([str(ord(c)).zfill(6) for c in uniqueChars])
	return uniqueChars
	

def getWordIndex(file):
	uniqueWords = set()
	
	for line in file:
		uniqueWords.update(line.split())
	
	return uniqueWords
	
	

	
if __name__ == "__main__":
	fr, en = readData()
	frIndex = getWordIndex(fr)
	enIndex = getWordIndex(en)
	
	frChars = getCharecters(fr)
	enChars = getCharecters(en)
	frCharDict = {c:ord(c) for c in frChars}
	enCharDict = {c:ord(c) for c in enChars}
	
	print("".join(sorted(frCharDict.keys())))
	print("".join(sorted(enCharDict.keys())))
	
	print(len(frIndex))
	print(len(enIndex))
	




