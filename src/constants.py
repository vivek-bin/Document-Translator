from inspect import getsourcefile
from os.path import abspath
from os.path import dirname

#paths
PATH = dirname(dirname(dirname(abspath(getsourcefile(lambda:0))))) + "/"
PROJECT = dirname(dirname(abspath(getsourcefile(lambda:0)))) + "/"


MODEL_PATH = PROJECT + "models/"
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


DATA_COUNT = int(1.0 * 1000 * 1000)

UNIT_SEP = "\x1f"
MASK_TOKEN = "MASK"
UNKNOWN_TOKEN = "UNK"
START_OF_SEQUENCE_TOKEN = "SOS"
END_OF_SEQUENCE_TOKEN = "EOS"

MIN_CHAR_COUNT = 30
MIN_WORD_COUNT = 20
CHAR_INPUT_SIZE = 8


MAX_WORDS = 60
INPUT_SEQUENCE_LENGTH = MAX_WORDS + 2
WORD_EMBEDDING_SIZE = 128
CHAR_EMBEDDING_SIZE = 8
NUM_LSTM_UNITS = 128
ATTENTION_UNITS = 128



TRAIN_SPLIT_PCT = 0.80
TRAIN_SPLIT = int(TRAIN_SPLIT_PCT * DATA_COUNT)