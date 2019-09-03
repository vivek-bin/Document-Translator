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
	
def readDictionaryPDF(bestWordOnly=True):
	from src import constants as CONST
	from PyPDF2 import PdfFileReader

	with open(CONST.DICTIONARY_PATH, "rb") as pdfFileBinary:
		pdfFile = PdfFileReader(pdfFileBinary)
		pageTexts = []
		for i in range(3, pdfFile.numPages-1):
			footerLen = len("English-french (dictionnaire)English-french Dictionary\n" + str(i))
			pageTexts.append(pdfFile.getPage(i).extractText()[:-footerLen].split("\n"))

	dictList = [x.lower().split(":") for page in pageTexts for x in page if x]
	engToFrDict = {x[0].strip():[v.strip() for v in x[1].split(", ")] for x in dictList}
	
	frToEngDict = {}
	for key, valueList in engToFrDict.items():
		for value in valueList:
			try:
				frToEngDict[value].append(key)
			except KeyError:
				frToEngDict[value] = [key]
		
	if bestWordOnly:
		# already sorted as such in dictionary for eng->fr
		engToFrDict = {key:valueList[0] for key,valueList in engToFrDict.items()}
		# select shortest word as best
		frToEngDict = {key:[v for v in valueList if len(v) == min([len(v) for v in valueList])][0] for key, valueList in frToEngDict.items()}
	
	wordDict = {}
	wordDict["en"] = engToFrDict
	wordDict["fr"] = frToEngDict

	return wordDict


def writeFile(fileName, fileData):
	file = open(CONST.LOGS+fileName,mode="w",encoding="utf-8")
	file.writelines("\n".join(fileData))
	file.close()
