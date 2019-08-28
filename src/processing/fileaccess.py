from .. import constants as CONST
import gzip
import os

MAX_WORDS = 60

def readFile(fileName):
	file = []
	with open(fileName, encoding="utf8") as f:
		file = list(f)
	file = [l.lower() for l in file]
	return file

def readArchiveFile(fileName):
	with gzip.open(fileName,"rb") as f:
		file = f.read().decode("ISO-8859-1").lower()
	
	file = file.strip().split("\n")
	return file
	
	
def loadEuroParl():
	fileFr = readFile(CONST.EUROPARL_FR)
	fileEn = readFile(CONST.EUROPARL_EN)

	if len(fileEn) != len(fileFr):
		raise Exception("EuroParl corpus lengths mismatch")

	print("europarl length = "+str(len(fileEn)))
	return fileFr,fileEn
	
	
def loadHansards():
	fileEn = []
	fileFr = []
	
	for fileDir in [CONST.HANSARDS_SENATE_TRAIN,CONST.HANSARDS_HOUSE_TRAIN]:
		fileList = os.listdir(fileDir)
		fileList = list(set([f[:-4] for f in fileList]))
		for fileName in fileList:
			fileEn.extend(readArchiveFile(fileDir+fileName+"e.gz"))
			fileFr.extend(readArchiveFile(fileDir+fileName+"f.gz"))
	
	if len(fileEn) != len(fileFr):
		raise Exception("Hansards corpus lengths mismatch")

	print("hansards length = "+str(len(fileEn)))
	return fileFr,fileEn
	



def writeFile(fileName, fileData):
	file = open(CONST.LOGS+fileName,mode="w",encoding="utf-8")
	file.writelines("\n".join(fileData))
	file.close()
