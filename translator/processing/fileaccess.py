from .. import constants as CONST
import gzip
import os
import xlrd
import csv
from PyPDF2 import PdfFileReader
import xml.etree.ElementTree as ET
import zipfile
import math
import io
import tempfile
import re

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
	with open(fileName, mode, encoding="utf-8-sig", newline="") as f:
		writer = csv.writer(f)
		for row in rows:
			writer.writerow(row)

def writeProcessedData(data, fileName, order=None):
	with open(CONST.PROCESSED_DATA + fileName + ".txt", "w", encoding="utf-8") as f:
		if order is None:
			for line in data:
				f.write(line + "\n")
		else:
			assert len(data) >= len(order)
			for i in order:
				f.write(data[i] + "\n")

def readProcessedData(fileName, startPos=0, endPos=CONST.DATA_COUNT):
	assert startPos < CONST.DATA_COUNT
	assert endPos <= CONST.DATA_COUNT
	with open(CONST.PROCESSED_DATA + fileName + ".txt", encoding="utf8") as f:
		lines = [line.rstrip() for i, line in enumerate(f) if i >= startPos and i < endPos]

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
	
def writeUpdatedDoc(tree, oldFilePath, newFilePath):
	root = tree.getroot()
	for k, v in CONST.DOCX_NAMESPACES.items():
		ET.register_namespace(k, v)
		if k not in ["mc", "w"]:
			root.attrib["xmlns:"+k] = v
	extension = oldFilePath.split(".")[-1].lower()

	zin = zipfile.ZipFile (oldFilePath, 'r')
	zout = zipfile.ZipFile (newFilePath, 'w')
	for fileInfo in zin.infolist():
		fileData = zin.read(fileInfo.filename)
		if fileInfo.filename in ["word/document.xml", "xl/sharedStrings.xml"]:
			with tempfile.TemporaryFile() as temp:
				temp.write(b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n')
				tree.write(temp)
				temp.seek(0)
				text = temp.read().decode()
			if extension == "docx":
				nameSpaces = re.findall("<w:document .*?>", fileData.decode())
				text = re.sub("<w:document .*?>", nameSpaces[0], text, 1)
			elif extension == "xlsx":
				nameSpaces = re.findall("<sst .*?>", fileData.decode())
				text = re.sub("<ns0:sst .*?>", nameSpaces[0], text, 1)
				text = text.replace("<ns0:", "<").replace("</ns0:", "</")
			else:
				raise Exception("unknown document type, only docx and xlsx allowed")
				
			
			zout.writestr(fileInfo, text)
		else:
			zout.writestr(fileInfo, fileData)
	zout.close()
	zin.close()

def readXMLFromDoc(fileName):
	for k, v in CONST.DOCX_NAMESPACES.items():
		ET.register_namespace(k, v)
	extension = fileName.split(".")[-1].lower()
	docxFile = zipfile.ZipFile(fileName)
	if extension == "docx":
		xml = docxFile.read("word/document.xml")
	elif extension == "xlsx":
		xml = docxFile.read("xl/sharedStrings.xml")
	else:
		raise Exception("unknown document type, only docx and xlsx allowed")
	docxFile.close()
	tree = ET.parse(io.StringIO(xml.decode()))

	return tree
	
def loadSFDData():
	return readCSV(CONST.PROJECT_TRANSLATIONS_EXTRACT_CSV_PATH)

def loadStandard(name, langs):
	data = (readFile(CONST.DATA + name + "." + lang) for lang in langs)

	lens = (len(d) for d in data)

	if min(lens) != max(lens):
		raise Exception(name + " corpus lengths mismatch! " + str(lens))

	print(name, "length =", lens[0])
	return data
	
	
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
	
	for i in range(sheet.nrows):
		fr = str(sheet.cell_value(i, 0)).strip()
		en = str(sheet.cell_value(i, 1)).strip()
		if fr and en:
			fr = [x for x in fr.split("|") if x.strip()]
			en = [x for x in en.split("|") if x.strip()]
			for f in fr:
				frToEngDict[f.lower()] = en[0]
			for e in en:
				engToFrDict[e.lower()] = fr[0]
	
	wordDict = {}
	wordDict["en"] = engToFrDict
	wordDict["fr"] = frToEngDict

	return wordDict