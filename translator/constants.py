from inspect import getsourcefile
from os.path import abspath
from os.path import dirname
from os.path import isdir
from time import time, ctime
import numpy as np
import os
import sys

print(ctime().rjust(60,"-"))
START_TIME = time()
def LAPSED_TIME():
	return "{:10.2f} seconds".format((time() - START_TIME)).rjust(60,"-")

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


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
DATA_PARTITIONS = 50

UNIT_SEP = "\x1f"
MASK_TOKEN = "MASK"
UNKNOWN_TOKEN = "UNK"
ALPHANUM_UNKNOWN_TOKEN = "ALPHANUM_UNK"
START_OF_SEQUENCE_TOKEN = "SOS"
END_OF_SEQUENCE_TOKEN = "EOS"
WORD_STEM_TRAIL_IDENTIFIER = "##"

RARE_CHAR_COUNT = 80
MIN_CHAR_COUNT = 150
MIN_WORD_COUNT = 100
CHAR_INPUT_SIZE = 4


###model parameters
SCALE_DOWN_MODEL_BY = 1
INCLUDE_CHAR_EMBEDDING = True

#common model params
NUM_ATTENTION_HEADS = 8
MAX_WORDS = 120
EMBEDDING_SIZE = 256 // SCALE_DOWN_MODEL_BY
WORD_EMBEDDING_SIZE = (EMBEDDING_SIZE * 3) // 4
CHAR_EMBEDDING_SIZE = ((EMBEDDING_SIZE - WORD_EMBEDDING_SIZE) // CHAR_INPUT_SIZE) // 2
ATTENTION_UNITS = 512 // SCALE_DOWN_MODEL_BY
MODEL_BASE_UNITS = 512 // SCALE_DOWN_MODEL_BY
FEED_FORWARD_UNITS = 2048 // SCALE_DOWN_MODEL_BY

LSTM_ACTIVATION = "tanh"
LSTM_RECURRENT_ACTIVATION = "sigmoid"
DENSE_ACTIVATION = "relu"

#recurrent model specific params
DECODER_ENCODER_DEPTH = 2
RECURRENT_LAYER_RESIDUALS = True
#transformer model specific params
ENCODER_ATTENTION_STAGES = 6
DECODER_ATTENTION_STAGES = 6

#transformer preprocessed data
MAX_POSITIONAL_EMBEDDING = np.array([[pos/np.power(10, 8. * i / EMBEDDING_SIZE) for i in range(EMBEDDING_SIZE)] for pos in range(MAX_WORDS + 50)])
MAX_POSITIONAL_EMBEDDING[:, 0::2] = np.sin(MAX_POSITIONAL_EMBEDDING[:, 0::2])
MAX_POSITIONAL_EMBEDDING[:, 1::2] = np.cos(MAX_POSITIONAL_EMBEDDING[:, 1::2])

###training parameters
TRAIN_SPLIT_PCT = 0.90
TRAIN_SPLIT = int(TRAIN_SPLIT_PCT * DATA_COUNT)
BATCH_SIZE = 32
NUM_EPOCHS = 30
VALIDATION_SPLIT_PCT = 0.1
VALIDATION_SPLIT = int(VALIDATION_SPLIT_PCT * TRAIN_SPLIT)
LEARNING_RATE = 0.01 / (5**0)
LEARNING_RATE_DECAY = 0.1
LABEL_SMOOTHENING = 0.0
LOSS_FUNCTION = "sparse_categorical_crossentropy"
EVALUATION_METRIC = "sparse_categorical_accuracy"
USE_TENSORBOARD = False

MODEL_NAME_SUFFIX = ".hdf5"
MODEL_CHECKPOINT_NAME_SUFFIX = "-{epoch:04d}-{val_" + EVALUATION_METRIC + ":.4f}" + MODEL_NAME_SUFFIX
MODEL_TRAINED_NAME_SUFFIX = "-Trained" + MODEL_NAME_SUFFIX


###translation parameters
MAX_TRANSLATION_LENGTH = 200
BEAM_SIZE = 8