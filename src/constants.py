from inspect import getsourcefile
from os.path import abspath
from os.path import dirname
from os.path import isdir

#paths
GOOGLE_DRIVE_PATH = "/content/drive/My Drive/"
if isdir(GOOGLE_DRIVE_PATH):
	PATH = GOOGLE_DRIVE_PATH
	PROJECT = GOOGLE_DRIVE_PATH
else:
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


DATA_COUNT = int(0.5 * 1000 * 1000)

UNIT_SEP = "\x1f"
MASK_TOKEN = "MASK"
UNKNOWN_TOKEN = "UNK"
START_OF_SEQUENCE_TOKEN = "SOS"
END_OF_SEQUENCE_TOKEN = "EOS"

MIN_CHAR_COUNT = 30
MIN_WORD_COUNT = 20
CHAR_INPUT_SIZE = 8


LSTM_ACTIVATION = "tanh"
DENSE_ACTIVATION = "relu"

NUM_ATTENTION_HEADS = 4
MAX_WORDS = 60
WORD_EMBEDDING_SIZE = 256
CHAR_EMBEDDING_SIZE = (WORD_EMBEDDING_SIZE // CHAR_INPUT_SIZE) // 2
NUM_LSTM_UNITS = 128
ATTENTION_UNITS = 128


BATCH_SIZE = 128
NUM_EPOCHS = 1000
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