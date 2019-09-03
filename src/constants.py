from inspect import getsourcefile
from os.path import abspath
from os.path import dirname
from os.path import isdir
from time import time
import numpy as np

START_TIME = time()
def LAPSED_TIME():
	return "{:10.2f} seconds".format((time() - START_TIME)).rjust(60,"-")

###paths
GOOGLE_DRIVE_PATH = "/content/drive/My Drive/"
if isdir(GOOGLE_DRIVE_PATH):
	PATH = GOOGLE_DRIVE_PATH
	PROJECT = GOOGLE_DRIVE_PATH
else:
	PATH = dirname(dirname(dirname(abspath(getsourcefile(lambda:0))))) + "/"
	PROJECT = dirname(dirname(abspath(getsourcefile(lambda:0)))) + "/"


###data paths
ENCODINGS = PROJECT + "encodings/"
MODELS = PATH + "models/"
DATA = PATH + "data/"
LOGS = PATH + "logs/"

EUROPARL = "EuroParl/europarl-v7.fr-en"
COMMON_CRAWL = "training-parallel-commoncrawl/commoncrawl.fr-en"
PARA_CRAWL ="paracrawl-release1.en-fr.zipporah0-dedup-clean/paracrawl-release1.en-fr.zipporah0-dedup-clean"
GIGA_FREN = "training-giga-fren/giga-fren.release2.fixed"

HANSARDS = DATA + "hansard/"
HANSARDS_HOUSE = HANSARDS + "sentence-pairs/house/debates/"
HANSARDS_SENATE = HANSARDS + "sentence-pairs/senate/debates/"
HANSARDS_HOUSE_TRAIN = HANSARDS_HOUSE + "training/"
HANSARDS_HOUSE_TEST = HANSARDS_HOUSE + "testing/"
HANSARDS_SENATE_TRAIN = HANSARDS_SENATE + "training/"
HANSARDS_SENATE_TEST = HANSARDS_SENATE + "testing/"

FRA_EN_DATA = DATA + "fra-eng/fra.txt"

DICTIONARY_PATH = DATA + "English-French_Dictionary.pdf"


PROCESSED_DATA = DATA + "processed input/"


###data parameters
DATA_COUNT = int(10 * 1000 * 1000)
DATA_PARTITIONS = 14

UNIT_SEP = "\x1f"
MASK_TOKEN = "MASK"
UNKNOWN_TOKEN = "UNK"
START_OF_SEQUENCE_TOKEN = "SOS"
END_OF_SEQUENCE_TOKEN = "EOS"
WORD_STEM_TRAIL_IDENTIFIER = "##"

RARE_CHAR_COUNT = 80
MIN_CHAR_COUNT = 150
MIN_WORD_COUNT = 100
CHAR_INPUT_SIZE = 4


###model parameters
INCLUDE_CHAR_EMBEDDING = False

DECODER_ENCODER_DEPTH = 2
ENCODER_ATTENTION_STAGES = 4
DECODER_ATTENTION_STAGES = 4
NUM_ATTENTION_HEADS = 8
MAX_WORDS = 120
EMBEDDING_SIZE = 256
WORD_EMBEDDING_SIZE = (EMBEDDING_SIZE * 3) // 4
CHAR_EMBEDDING_SIZE = ((EMBEDDING_SIZE - WORD_EMBEDDING_SIZE) // CHAR_INPUT_SIZE) // 2
NUM_LSTM_UNITS = 256
ATTENTION_UNITS = 256
FEED_FORWARD_UNITS = 1024

MAX_POSITIONAL_EMBEDDING = np.array([[pos/np.power(10, 8. * i / EMBEDDING_SIZE) for i in range(EMBEDDING_SIZE)] for pos in range(MAX_WORDS + 50)])
MAX_POSITIONAL_EMBEDDING[:, 0::2] = np.sin(MAX_POSITIONAL_EMBEDDING[:, 0::2])
MAX_POSITIONAL_EMBEDDING[:, 1::2] = np.cos(MAX_POSITIONAL_EMBEDDING[:, 1::2])

LSTM_ACTIVATION = "tanh"
DENSE_ACTIVATION = "relu"


###training parameters
TRAIN_SPLIT_PCT = 0.90
TRAIN_SPLIT = int(TRAIN_SPLIT_PCT * DATA_COUNT)
BATCH_SIZE = 128
NUM_EPOCHS = 40
VALIDATION_SPLIT_PCT = 0.1
VALIDATION_SPLIT = int(VALIDATION_SPLIT_PCT * TRAIN_SPLIT)
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.05
LOSS_FUNCTION = "sparse_categorical_crossentropy"
EVALUATION_METRIC = "sparse_categorical_accuracy"




MODEL_NAME_SUFFIX = ".hdf5"
MODEL_CHECKPOINT_NAME_SUFFIX = "-{epoch:04d}-{val_" + EVALUATION_METRIC + ":.4f}" + MODEL_NAME_SUFFIX
MODEL_TRAINED_NAME_SUFFIX = "-Trained" + MODEL_NAME_SUFFIX


###translation parameters
MAX_TRANSLATION_LENGTH = 200
BEAM_SIZE = 16