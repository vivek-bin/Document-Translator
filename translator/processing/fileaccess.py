from .. import constants as CONST
import gzip
import os
import xlrd
import csv
from PyPDF2 import PdfFileReader
from docx_utils.flatten import opc_to_flat_opc
import xml.etree.ElementTree as ET 


def getAllFilePaths(path):
	paths = []
	for root, _, files in os.walk(path):
		for f in files:
			paths.append(os.path.join(root,f))
	return paths

def readFile(fileName):
	file = []
	with open(fileName, encoding="utf8") as f:
		file = list(f)

	return file

def writeFile(fileName, fileData):
	file = open(CONST.LOGS+fileName,mode="w",encoding="utf-8")
	file.writelines("\n".join(fileData))
	file.close()

def readCSV(fileName):
	with open(fileName, 'rb') as f:
		reader = csv.reader(f)
		data = list(reader)
	return data
	
def writeCSV(fileName, rows, mode="a"):
	with open(fileName, mode, encoding="utf-8-sig") as f:
		writer = csv.writer(f)
		for row in rows:
			writer.writerow(row)

def writeProcessedData(data, fileName):
	with open(CONST.PROCESSED_DATA + fileName + ".txt", "w", encoding="utf-8") as f:
		for line in data:
			f.write(line + "\n")

def readProcessedData(fileName, startPos=0, endPos=CONST.DATA_COUNT):
	assert startPos < CONST.DATA_COUNT
	assert endPos <= CONST.DATA_COUNT
	with open(CONST.PROCESSED_DATA + fileName + ".txt", encoding="utf8") as f:
		lines = [line.strip() for i, line in enumerate(f) if i >= startPos and i < endPos]

	return lines

def lenProcessedData(fileName):
	with open(CONST.PROCESSED_DATA + fileName + ".txt", encoding="utf8") as f:
		for i, _ in enumerate(f, 1):
			pass
	return min(i, CONST.DATA_COUNT)

def readArchiveFile(fileName):
	with gzip.open(fileName,"rb") as f:
		file = f.read().decode("ISO-8859-1").lower()
	
	file = file.strip().split("\n")
	return file
	
def writeXMLToDoc(tree, filePath):
	ET.register_namespace('w',"http://schemas.openxmlformats.org/wordprocessingml/2006/main")
	ET.register_namespace('pkg',"http://schemas.microsoft.com/office/2006/xmlPackage")

    # convert and write to disk -> load to mem -> delete from disk
	tree.write(filePath.split(".")[0]+".xml")
	raise Exception
	gfhfhf

	return tree

def readXMLFromDoc(fileName):
	ET.register_namespace('w',"http://schemas.openxmlformats.org/wordprocessingml/2006/main")
	ET.register_namespace('pkg',"http://schemas.microsoft.com/office/2006/xmlPackage")

    # convert and write to disk -> load to mem -> delete from disk
	xmlPath = fileName.split('.')[0]+'.xml'
	opc_to_flat_opc(fileName ,xmlPath)
	tree = ET.parse(xmlPath)
	os.remove(xmlPath)

	return tree

def extractTextFromXML(path):
	ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
	outText = []
	
	tree = readXMLFromDoc(path)
	root = tree.getroot()
	paragraphs = root.findall('.//w:p', ns)
	for paragraph in paragraphs:
		rows = paragraph.findall('w:r', ns)
		paraText = ""
		for row in rows:
			rowText = row.find('w:t', ns)
			if rowText != None:
				paraText = paraText + rowText.text
		
		if paraText.strip():
			outText.append(paraText)
	
	return outText

def writeAllSFDData():
	englishPaths = getAllFilePaths(CONST.PROJECT_TRANSLATIONS_EN_PATH)
	frenchPaths = getAllFilePaths(CONST.PROJECT_TRANSLATIONS_FR_PATH)

	for englishPath, frenchPath in zip(englishPaths, frenchPaths):
		englishText = extractTextFromXML(englishPath)
		frenchText = extractTextFromXML(frenchPath)
		
		writeCSV(CONST.PROJECT_TRANSLATIONS_EXTRACT_CSV_PATH, zip(englishText, frenchText))

	
def loadSFDData():
	return readCSV(CONST.PROJECT_TRANSLATIONS_EXTRACT_CSV_PATH)

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

def readDictionaryGlossary(bestWordOnly=True):
	wb = xlrd.open_workbook(CONST.GLOSSARY_PATH) 
	sheet = wb.sheet_by_index(0) 
	i = 0
	frToEngDict = {}
	engToFrDict = {}
	while sheet.cell_value(i, 0):
		i = i + 1
		a = str(sheet.cell_value(i, 0)).strip()
		b = str(sheet.cell_value(i, 1)).strip()
		frToEngDict[a.lower()] = b
		engToFrDict[b.lower()] = a
	
	wordDict = {}
	wordDict["en"] = engToFrDict
	wordDict["fr"] = frToEngDict

	return wordDict