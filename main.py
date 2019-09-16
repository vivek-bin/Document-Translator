import sys
from translator import constants as CONST

VALID_FLAGS = ["--prep", "--train", "--translate"]

def main():
	flags = sys.argv[1:]
	prepareDataFlag = False
	trainModelFlag = False
	translateFlag = False

	if not flags:
		translateFlag = True
		modelNum = 1
	else:
		for flag in flags:
			if flag not in VALID_FLAGS and flag not in ["1","2"]:
				print("Invalid flag. Valid flags are "+", ".join(VALID_FLAGS))
				return False
		if "--prep" == flags[0]:
			prepareDataFlag = True
		if "--train" == flags[0]:
			trainModelFlag = True
			try:
				modelNum = int(flags[1])
			except IndexError:
				modelNum = 1
			
		if "--translate" == flags[0]:
			translateFlag = True
			try:
				modelNum = int(flags[1])
			except IndexError:
				modelNum = 1

	if prepareDataFlag:
		from translator.preparedata import writeEncodingsData
		writeEncodingsData()

	if trainModelFlag:
		from translator.trainmodel import trainModel
		trainModel(startLang="fr", endLang="en", modelNum=modelNum)

	if translateFlag:
		from translator.translate import Translator
		frToEngTranslater = Translator(startLang="fr", endLang="en", modelNum=modelNum)
		print(CONST.LAPSED_TIME())
		text = ["Ma question porte sur un sujet qui est à l'ordre du jour du jeudi et que je soulèverai donc une nouvelle fois."]
		print(frToEngTranslater(text))
		print(CONST.LAPSED_TIME())


	return True


if __name__ == "__main__":
	main()