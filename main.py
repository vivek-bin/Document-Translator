import sys
from translator import constants as CONST

VALID_FLAGS = ["--prep", "--train", "--translate"]

def main():
	flags = sys.argv[1:]
	prepareDataFlag = False
	trainModelFlag = False
	translateFlag = False
	modelNum = 0
	text = []
	documents = []


	try:
		if flags[0] not in VALID_FLAGS:
			print("Invalid flag. Valid flags are "+", ".join(VALID_FLAGS))
			return False
		if "--prep" == flags[0]:
			prepareDataFlag = True
		if "--train" == flags[0]:
			trainModelFlag = True
		if "--translate" == flags[0]:
			translateFlag = True
		flags.pop(0)
	except IndexError:
		translateFlag = True
	
	try:
		i = flags.index("-m")
		modelNum = int(flags.pop(i+1))
		flags.pop(i)
	except ValueError:
		modelNum = 1

	try:
		i = flags.index("-s")
		flags.pop(i)
		text = flags
	except ValueError:
		pass

	try:
		i = flags.index("-d")
		flags.pop(i)
		documents = flags
	except ValueError:
		pass

	if prepareDataFlag:
		from translator.preparedata import writeAllData, writeEncodingFromProcessed
		writeAllData()
		writeEncodingFromProcessed("fr")
		writeEncodingFromProcessed("en")

	if trainModelFlag:
		assert modelNum in [1, 2]

		from translator.trainmodel import trainModel
		trainModel(startLang="fr", endLang="en", modelNum=modelNum)

	if translateFlag:
		text = ["Ma question porte sur un sujet qui est à l'ordre du jour du jeudi.", "valorisation versement libre valorisation.", "ljh kblhblb gbk."] + text		#testing
		assert modelNum in [1, 2]
		assert bool(text) ^ bool(documents)

		from translator.translate import Translator
		frToEngTranslater = Translator(startLang="fr", endLang="en", modelNum=modelNum)
		print(CONST.LAPSED_TIME())
		if documents:
			for doc in documents:
				frToEngTranslater.translateDocument(doc)
		else:
			for line in text:
				_ = frToEngTranslater.translate(line)
		print(CONST.LAPSED_TIME())


	return True


if __name__ == "__main__":
	main()