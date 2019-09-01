from .. import constants as CONST
import gzip
import os

MAX_WORDS = 60

def readFile(fileName):
	file = []
	with open(fileName, encoding="utf8") as f:
		file = list(f)

	return file

def readArchiveFile(fileName):
	with gzip.open(fileName,"rb") as f:
		file = f.read().decode("ISO-8859-1").lower()
	
	file = file.strip().split("\n")
	return file
	

def loadStandard(name):
	fileFr = readFile(CONST.DATA + name + ".fr")
	fileEn = readFile(CONST.DATA + name + ".en")

	if len(fileEn) != len(fileFr):
		raise Exception("{0} corpus lengths mismatch! en:{1} vs fr:{2}".format(name, len(fileEn), len(fileFr)))

	print(name + " length = "+str(len(fileEn)))
	return fileFr, fileEn
	
	
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

	print("Hansards length = "+str(len(fileEn)))
	return fileFr, fileEn
	

def loadFraEng():
	fileBoth = readFile(CONST.FRA_EN_DATA)
	fileBoth = [line.split("\t") for line in fileBoth]
	fileEn = [x[0] for x in fileBoth]
	fileFr = [x[1] for x in fileBoth]

	lineSplitCheck = [len(x) for x in fileBoth if len(x)!= 2]
	if lineSplitCheck:
		raise Exception("Fra-eng corpus erroneous tabs")

	print("fra-eng length = "+str(len(fileEn)))
	return fileFr, fileEn
	



def writeFile(fileName, fileData):
	file = open(CONST.LOGS+fileName,mode="w",encoding="utf-8")
	file.writelines("\n".join(fileData))
	file.close()
