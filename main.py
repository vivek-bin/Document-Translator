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
		from src.translate import translate
		text = [""]
		_ = translate(text)


	return True


if __name__ == "__main__":
	main()