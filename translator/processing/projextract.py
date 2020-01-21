from .. import constants as CONST
from . import fileaccess as FA
import os
import re
import shutil
import xlrd

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
	
def groupFilesAllDirectories():
	for d in sorted(os.listdir(CONST.PROJECT_TRANSLATIONS_PATH)):
		inputDir = CONST.PROJECT_TRANSLATIONS_PATH + d + "/"
		groups = groupPaths(inputDir)

		outputDir = CONST.PROJECT_TRANSLATIONS_MATCHED_PATH + d + "/"
		if not os.path.isdir(outputDir):
			os.mkdir(outputDir)
		
		for group, filePaths in groups.items():
			outFileDir = outputDir + group + "/"
			copyGroup(filePaths, outFileDir)




def extractFromExcel(filePath):
	xls = xlrd.open_workbook(filePath)
	sheetNames = xls.sheet_names()
	if len(sheetNames) == 1:
		sheet = xls.sheet_by_index(0)
	else:
		if "PLAN DE TEST" in sheetNames:
			sheet = xls.sheet_by_name("PLAN DE TEST")
		elif "TEST PLAN" in sheetNames:
			sheet = xls.sheet_by_name("TEST PLAN")
		else:
			raise Exception("Dont know which sheet to extract from")

	cols = {}
	colNums = {}
	for i in range(sheet.ncols):
		val = sheet.cell_value(0,i).strip()
		if val:
			cols[val] = i
			colNums[i] = val

	data = []
	frCols = ["COMMENTAIRE PROCESSUS", "CONSIGNES D EXECUTION"]
	enColLists = [["PROCESS COMMENT", "PROCESS COMMENTS", "Process comment"], ["INSTRUCTION FOR PERFORMING", "RUNNING  INSTRUCTIONS", "EXECUTION ADVICE", "Execution advice", "INSTRUCTIONS FOR PERFORMING", "EXECUTION INSTRUCTIONS", "COMMENTS FOR PROCESSING"]]
	for i in range(len(frCols)):
		if frCols[i] not in cols.keys():
			continue
		frPos = cols[frCols[i]]
		enPos = frPos + 1
		if enPos not in colNums.keys() or colNums[enPos] not in enColLists[i]:
			continue
		
		for j in range(1, sheet.nrows):
			fr = [x.strip() for x in str(sheet.cell_value(j, frPos)).split("\n") if x.strip()]
			en = [x.strip() for x in str(sheet.cell_value(j, enPos)).split("\n") if x.strip()]

			if fr and en:
				en.extend([""]*(len(fr)-len(en)))		#pad
				fr.extend([""]*(len(en)-len(fr)))		#pad
				data.extend(list(zip(en, fr)))

	return data

def getXMLTextBlocks(x, skipContents=False):
	def isTag(element, tag):
		return ((element.tag.split("}")[-1]) == tag)

	if isTag(x, "p"):
		textTags = [rt for pt in x for rt in pt if isTag(pt, "r") and (isTag(rt, "t") or isTag(rt, "tab"))]
		tabPos = [i for i, t in enumerate(textTags) if isTag(t, "tab")]
		prevPos = 0
		for pos in tabPos:
			temp = textTags[prevPos:pos]
			prevPos = pos + 1
			yield temp
		yield textTags[prevPos:]
	else:
		for xt in x:
			if (not skipContents) or (not isTag(xt, "sdtContent")):
				yield from getXMLTextBlocks(xt, skipContents=skipContents)

def joinXMLTextGen(textTagsGen):
	for textTags in textTagsGen:
		x = joinXMLTextTags(textTags)
		if x:
			yield x
	
def joinXMLTextTags(textTags):
	joinedText = ""
	for tt in textTags:
		attributes = [a.split("}")[-1] for a in tt.attrib.keys()]
		if "space" in attributes:
			text = tt.text
		else:
			text = tt.text.strip()
		joinedText = joinedText + text
	
	return joinedText.strip()


def extractFilesAllDirectories():
	for d in sorted(os.listdir(CONST.PROJECT_TRANSLATIONS_MATCHED_PATH)):
		print(d)
		inputDir = CONST.PROJECT_TRANSLATIONS_MATCHED_PATH + d + "/"
		outputDir = inputDir + "csv/"

		for d2 in sorted(os.listdir(inputDir)):
			inputDir2 = inputDir + d2 + "/"
			docFiles = []
			if not os.path.isdir(outputDir):
				os.mkdir(outputDir)
			
			for fileName in sorted(os.listdir(inputDir2), key=lambda x:len(x)):
				if fileName.split(".")[-1].startswith("xls"):
					xlsData = extractFromExcel(inputDir2+fileName)	 # excel files processed individually
					FA.writeCSV(outputDir+fileName+"_dataset.csv", xlsData)
				else:
					docFiles.append(inputDir2+fileName)
			
			if docFiles:
				assert len(docFiles) == 2, str(docFiles)
				frenchXML = FA.readXMLFromDoc(docFiles[0])
				frenchTextGen = getXMLTextBlocks(frenchXML.getroot(), skipContents=True)
				frenchTextGen = joinXMLTextGen(frenchTextGen)

				englishXML = FA.readXMLFromDoc(docFiles[1])
				englishTextGen = getXMLTextBlocks(englishXML.getroot(), skipContents=True)
				englishTextGen = joinXMLTextGen(englishTextGen)

				z = []
				for en, fr in zip(englishTextGen, frenchTextGen):
					if "[" in en and "]" in en:
						if en.split("[")[0].strip().lower() == fr.strip().lower():
							en = "[".join(en.split("[")[1:])
							en = "]".join(en.split("]")[:-1])
					z.append((en, fr))

				FA.writeCSV(outputDir + d2 + "_dataset.csv", z)
			
