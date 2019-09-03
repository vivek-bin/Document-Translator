import sys
from src import constants as CONST

VALID_FLAGS = ["--prep", "--train", "--translate"]

def main():
	flags = sys.argv[1:]
	prepareDataFlag = False
	trainModelFlag = False
	translateFlag = False

	if not flags:
		trainModelFlag = True
		modelNum = 1
	else:
		for flag in flags:
			if flag not in VALID_FLAGS and flag not in ["1","2"]:
				print("Invalid flag. Valid flags are "+", ".join(VALID_FLAGS))
				return False
		if "--prep" in flags:
			prepareDataFlag = True
		if "--train" in flags:
			trainModelFlag = True
			try:
				modelNum = int(flags[flags.index("--train")+1])
			except (ValueError, IndexError):
				modelNum = 1
			
		if "--translate" in flags:
			translateFlag = True

	if prepareDataFlag:
		from src.preparedata import writeEncodingsData
		writeEncodingsData()

	if trainModelFlag:
		from src.trainmodel import trainModel
		trainModel(modelNum)

	if translateFlag:
		from src.translate import Translater
		frToEngTranslater = Translater(startLang="fr",endLang="en")
		print(CONST.LAPSED_TIME())
		text = ["Ma question porte sur un sujet qui est à l'ordre du jour du jeudi et que je soulèverai donc une nouvelle fois."]
		print(frToEngTranslater(text))
		print(CONST.LAPSED_TIME())


	return True


if __name__ == "__main__":
	main()