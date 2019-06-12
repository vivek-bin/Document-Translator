import numpy
import constants as CONST
import fileaccess as FA

MAX_WORDS = 60

def readData():
	epFr,epEn = FA.loadEuroParl()
	hFr,hEn = FA.loadHansards()
	
	fr, en = limitSentenceSize(epFr+hFr, epEn+hEn)
	
	
	

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







def vectorizeData():







def loadCleanData():







