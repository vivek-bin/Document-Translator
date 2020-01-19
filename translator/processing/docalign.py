from .. import constants as CONST
from .. import preparedata as PD
from . import fileaccess as FA
import os
import re
import shutil

def groupPaths(inputDir):
	paths = {}
	for path in FA.getAllFilePaths(inputDir):
		fileName = ".".join(path.split("/")[-1].split(".")[:-1])
		extension = path.split("/")[-1].split(".")[-1].lower()

		if "$" not in fileName and extension[:3] in ("xls", "doc"):
			pieces = re.split("[^a-zA-Z0-9]", fileName)
			paths[path] = [x for x in pieces if x and not(x.isnumeric() and len(x)==8)]

	groups = {}
	for f in paths.keys():
		groupList = []
		for f2 in paths.keys():
			for i in range(min(len(paths[f]), len(paths[f2]), 4)):
				if paths[f][i].lower() != paths[f2][i].lower():
					break
			else:
				groupList.append(f2)
	
		l = min([len(paths[p]) for p in groupList]+[4])
		group = "_".join(paths[f][:l])
		try:
			groups[group].update(groupList)
		except KeyError:
			groups[group] = set(groupList)

	return groups

def copyGroup(filePaths, outFileDir):
	beforeTranslation = False
	afterTranslation = False
	for filePath in filePaths:
		fileName = ".".join(filePath.split("/")[-1].split(".")[:-1])
		if "correct" in fileName.lower() or "translat" in fileName.lower():
			afterTranslation = True
		else:
			beforeTranslation = True
	
	for filePath in filePaths:
		fileName = ".".join(filePath.split("/")[-1].split(".")[:-1])
		fileExtension = filePath.split("/")[-1].split(".")[-1].lower()

		if (afterTranslation and beforeTranslation) or (afterTranslation and fileExtension.startswith("xls")):
			if not os.path.isdir(outFileDir):
				os.mkdir(outFileDir)

			copyDestName = "{}.{}".format(fileName, fileExtension)
			i = 2
			while os.path.isfile(outFileDir + copyDestName):
				if os.path.getsize(outFileDir + copyDestName)==os.path.getsize(filePath):
					break
				copyDestName = "{}({}).{}".format(fileName, i, fileExtension)
				i = i + 1
			else:
				shutil.copyfile(filePath, outFileDir + copyDestName)
	
def matchAllDirectories():
	for d in sorted(os.listdir(CONST.PROJECT_TRANSLATIONS_PATH)):
		inputDir = CONST.PROJECT_TRANSLATIONS_PATH + d + "/"
		groups = groupPaths(inputDir)

		outputDir = CONST.PROJECT_TRANSLATIONS_MATCHED_PATH + d + "/"
		if not os.path.isdir(outputDir):
			os.mkdir(outputDir)
		
		for group, filePaths in groups.items():
			outFileDir = outputDir + group + "/"
			copyGroup(filePaths, outFileDir)

def extractPairedData():
	for d in sorted(os.listdir(CONST.PROJECT_TRANSLATIONS_MATCHED_PATH))[:1]:
		inputDir = CONST.PROJECT_TRANSLATIONS_MATCHED_PATH + d + "/"
		outputDir = inputDir + "csv/"
		for d2 in sorted(os.listdir(inputDir)):
			inputDir2 = inputDir + d2 + "/"
			docFiles = []
			if not os.path.isdir(outputDir):
				os.mkdir(outputDir)
			for fileName in sorted(os.listdir(inputDir2), key=lambda x:len(x)):
				if fileName.split(".")[-1].startswith("xls"):
					pass # excel files processed individually
				else:
					docFiles.append(inputDir2+fileName)
			if docFiles:
				assert len(docFiles) == 2
				frenchXML = FA.readXMLFromDoc(docFiles[0])
				frenchTextGen = PD.getXMLTextBlocks(frenchXML, skipContents=True)

				englishXML = FA.readXMLFromDoc(docFiles[1])
				englishTextGen = PD.getXMLTextBlocks(englishXML, skipContents=True)

				FA.writeCSV(outputDir+d2+"_dataset.csv", zip(PD.joinXMLTextGen(englishTextGen), PD.joinXMLTextGen(frenchTextGen)))
			
