import sys

VALID_FLAGS = ["--prep", "--train", "--translate"]

def main():
	flags = sys.argv[1:]
	prepareDataFlag = False
	trainModelFlag = False
	translateFlag = False

	if not flags:
		trainModelFlag = True
	else:
		for flag in flags:
			if flag not in VALID_FLAGS:
				print("Invalid flag. Valid flags are "+", ".join(VALID_FLAGS))
				return False
		if "--prep" in flags:
			prepareDataFlag = True
		if "--train" in flags:
			trainModelFlag = True
		if "--translate" in flags:
			translateFlag = True

	if prepareDataFlag:
		from src.processing.preparedata import writeEncodingsData
		writeEncodingsData()

	if trainModelFlag:
		from src.trainmodel import trainModel
		trainModel()

	if translateFlag:
		from src.translate import translate
		text = [""]
		_ = translate(text)


	return True


if __name__ == "__main__":
	main()