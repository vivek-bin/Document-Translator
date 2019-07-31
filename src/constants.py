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


DATA_COUNT = 10 * 1000 * 1000

MAX_WORDS = 60
UNIT_SEP = "\x1f"

MIN_CHAR_COUNT = 30
MIN_WORD_COUNT = 20
CHAR_INPUT_SIZE = 8

INPUT_SENTENCE_LENGTH = 64
OUTPUT_SENTENCE_LENGTH = 1
INPUT_WORDS_COUNT = 10000
OUTPUT_WORDS_COUNT = 10000
EMBEDDING_SIZE = 128
NUM_LSTM_UNITS = 128

