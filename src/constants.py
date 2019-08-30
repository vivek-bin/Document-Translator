from inspect import getsourcefile
from os.path import abspath
from os.path import dirname
from os.path import isdir
from time import time
import numpy as np

START_TIME = time()
def LAPSED_TIME():
	return "{:10.2f} seconds".format((time() - START_TIME)).rjust(60,"-")

#paths
GOOGLE_DRIVE_PATH = "/content/drive/My Drive/"
if isdir(GOOGLE_DRIVE_PATH):
	PATH = GOOGLE_DRIVE_PATH
	PROJECT = GOOGLE_DRIVE_PATH
else:
	PATH = dirname(dirname(dirname(abspath(getsourcefile(lambda:0))))) + "/"
	PROJECT = dirname(dirname(abspath(getsourcefile(lambda:0)))) + "/"


MODEL_PATH = PATH + "models/"
ENCODING_PATH = PROJECT + "encodings/"


DATA = PATH + "data/"
LOGS = PATH + "logs/"

EUROPARL = DATA + "EuroParl/"
HANSARDS = DATA + "hansard/"

EUROPARL_EN = EUROPARL + "europarl-v7.fr-en.en"
EUROPARL_FR = EUROPARL + "europarl-v7.fr-en.fr"

HANSARDS_HOUSE = HANSARDS + "sentence-pairs/house/debates/"
HANSARDS_SENATE = HANSARDS + "sentence-pairs/senate/debates/"


HANSARDS_HOUSE_TRAIN = HANSARDS_HOUSE + "training/"
HANSARDS_HOUSE_TEST = HANSARDS_HOUSE + "testing/"
HANSARDS_SENATE_TRAIN = HANSARDS_SENATE + "training/"
HANSARDS_SENATE_TEST = HANSARDS_SENATE + "testing/"


PROCESSED_DATA = DATA + "processed input/"


DATA_COUNT = int(5 * 1000 * 1000)

UNIT_SEP = "\x1f"
MASK_TOKEN = "MASK"
UNKNOWN_TOKEN = "UNK"
START_OF_SEQUENCE_TOKEN = "SOS"
END_OF_SEQUENCE_TOKEN = "EOS"
WORD_STEM_TRAIL_IDENTIFIER = "##"

RARE_CHAR_COUNT = 30


MIN_CHAR_COUNT = 50
MIN_WORD_COUNT = 20
CHAR_INPUT_SIZE = 4

LSTM_ACTIVATION = "tanh"
DENSE_ACTIVATION = "relu"


ENCODER_ATTENTION_STAGES = 6
DECODER_ATTENTION_STAGES = 4
NUM_ATTENTION_HEADS = 4
MAX_WORDS = 120
EMBEDDING_SIZE = 256
WORD_EMBEDDING_SIZE = (EMBEDDING_SIZE * 3) // 4
CHAR_EMBEDDING_SIZE = ((EMBEDDING_SIZE - WORD_EMBEDDING_SIZE) // CHAR_INPUT_SIZE) // 2
NUM_LSTM_UNITS = 256
ATTENTION_UNITS = 512

MAX_POSITIONAL_EMBEDDING = np.array([[pos/np.power(10, 8. * i / EMBEDDING_SIZE) for i in range(EMBEDDING_SIZE)] for pos in range(MAX_WORDS + 50)])
MAX_POSITIONAL_EMBEDDING[:, 0::2] = np.sin(MAX_POSITIONAL_EMBEDDING[:, 0::2])
MAX_POSITIONAL_EMBEDDING[:, 1::2] = np.cos(MAX_POSITIONAL_EMBEDDING[:, 1::2])


BATCH_SIZE = 128
NUM_EPOCHS = 80
VALIDATION_SPLIT = 0.1
LEARNING_RATE = 0.01
LEARNING_RATE_DECAY = 0.00
LOSS_FUNCTION = "sparse_categorical_crossentropy"
EVALUATION_METRIC = "sparse_categorical_accuracy"


TRAIN_SPLIT_PCT = 0.80
TRAIN_SPLIT = int(TRAIN_SPLIT_PCT * DATA_COUNT)


MODEL_CHECKPOINT_NAME_START = "AttLSTMChP-"
MODEL_CHECKPOINT_NAME_END = ".hdf5"
MODEL_CHECKPOINT_NAME = MODEL_CHECKPOINT_NAME_START + "{epoch:04d}-{val_" + EVALUATION_METRIC + ":.4f}" + MODEL_CHECKPOINT_NAME_END



MAX_TRANSLATION_LENGTH = 100