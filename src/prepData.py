import numpy
import constants as CONST
import fileaccess as FA

MAX_WORDS = 60

def readData():
	epFr,epEn = FA.loadEuroParl()
	hFr,hEn = FA.loadHansards()
	
	fr, en = limitSentenceSize(epFr+hFr, epEn+hEn)
	
	fr = [l.split() for l in fr]
	en = [l.split() for l in en]
	
	uniqueFr = set()
	uniqueEn = set()
	#for words in fr:
	#	uniqueFr.update(words)
	#for words in en:
	#	uniqueEn.update(words)
	
	print(len(uniqueFr))
	#print(len(uniqueEn))
	
	

def limitSentenceSize(fileFr,fileEn):
	global MAX_WORDS
	fileEn2 = []
	fileFr2 = []
	for i in range(len(fileFr)):
		if len(fileFr[i].split()) < MAX_WORDS and len(fileEn[i].split()) < MAX_WORDS:
			fileFr2.append(fileFr[i])
			fileEn2.append(fileEn[i])
			
	return fileFr2, fileEn2


def cleanData():
	return






def vectorizeData():
	return






def loadCleanData():
	return

if __name__ == "__main__":
	readData()




