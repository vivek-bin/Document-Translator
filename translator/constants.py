from inspect import getsourcefile
from os.path import abspath
from os.path import dirname
from os.path import isdir
from time import time, ctime
import os
import sys
from tensorflow.keras import backend as K

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

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
ENCODINGS = PATH + "encodings/"
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

PROJECT_TRANSLATIONS_PATH = DATA + "project_translations/"
PROJECT_TRANSLATIONS_MATCHED_PATH = DATA + "project_translations_matched/"
PROJECT_TRANSLATIONS_EXTRACT_PATH = DATA + "project_translations_extracted/"
PROJECT_TRANSLATIONS_EXTRACT_CSV_PATH = PROJECT_TRANSLATIONS_PATH + "dataset.csv"


DICTIONARY_PATH = DATA + "English-French_Dictionary.pdf"
GLOSSARY_PATH = DATA + "glossary.xlsx"

PROCESSED_DATA = DATA + "processed input/"

DOCX_NAMESPACES = {"wpc":"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas",
    "mc":"http://schemas.openxmlformats.org/markup-compatibility/2006", 
    "o":"urn:schemas-microsoft-com:office:office", 
    "r":"http://schemas.openxmlformats.org/officeDocument/2006/relationships", 
    "m":"http://schemas.openxmlformats.org/officeDocument/2006/math", 
    "v":"urn:schemas-microsoft-com:vml", 
    "wp14":"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing", 
    "wp":"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing", 
    "w10":"urn:schemas-microsoft-com:office:word", 
    "w":"http://schemas.openxmlformats.org/wordprocessingml/2006/main", 
    "w14":"http://schemas.microsoft.com/office/word/2010/wordml", 
    "wpg":"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup", 
    "wpi":"http://schemas.microsoft.com/office/word/2010/wordprocessingInk", 
    "wne":"http://schemas.microsoft.com/office/word/2006/wordml", 
    "wps":"http://schemas.microsoft.com/office/word/2010/wordprocessingShape"}


###data parameters
DATA_COUNT = int(50 * 1000 * 1000)

SUBSTITUTION_CHAR = "\x1a"
SUBSTITUTION_CHAR_2 = "\x1d"
UNIT_SEP = "\x1f"
MASK_TOKEN = "MASK"
UNKNOWN_TOKEN = "UNK"
ALPHANUM_UNKNOWN_TOKEN = "ALPHANUM_UNK"
START_OF_SEQUENCE_TOKEN = "SOS"
END_OF_SEQUENCE_TOKEN = "EOS"
WORD_STEM_TRAIL_IDENTIFIER = "##"
NUM_WORDPIECES = lambda x:x.count(UNIT_SEP)

RARE_CHAR_COUNT = 80
MIN_CHAR_COUNT = 300
MIN_WORD_COUNT = 200
CHAR_INPUT_SIZE = 4


###model parameters
SCALE_DOWN_MODEL = 1
INCLUDE_CHAR_EMBEDDING = True
SHARED_INPUT_OUTPUT_EMBEDDINGS = True
LAYER_NORMALIZATION = True

#common model params
NUM_ATTENTION_HEADS = 8
MAX_WORDS = 100
EMBEDDING_SIZE = 256 // SCALE_DOWN_MODEL
WORD_EMBEDDING_SIZE = (EMBEDDING_SIZE * 3) // 4
CHAR_EMBEDDING_SIZE = ((EMBEDDING_SIZE - WORD_EMBEDDING_SIZE) // CHAR_INPUT_SIZE) // 2
ATTENTION_UNITS = 512 // SCALE_DOWN_MODEL
MODEL_BASE_UNITS = 512 // SCALE_DOWN_MODEL
FEED_FORWARD_UNITS = 2048 // SCALE_DOWN_MODEL
L1_REGULARISATION = 1e-4
L2_REGULARISATION = 1e-4
BIAS_INITIALIZER = "glorot_uniform"

LSTM_ACTIVATION = "tanh"
LSTM_RECURRENT_ACTIVATION = "sigmoid"
DENSE_ACTIVATION = lambda x: K.maximum(x, x * 0.1) # leaky relu

LAYER_NORMALIZATION_ARGUMENTS = dict(center=True, scale=True)

#recurrent model specific params
DECODER_ENCODER_DEPTH = 4
ENCODER_BIDIREC_DEPTH = 2
RECURRENT_LAYER_RESIDUALS = True
PEEPHOLE = True
#transformer model specific params
ENCODER_ATTENTION_STAGES = 6
DECODER_ATTENTION_STAGES = 6

###training parameters
DATA_PARTITIONS = 100
TRAIN_SPLIT_PCT = 0.90
TRAIN_SPLIT = int(TRAIN_SPLIT_PCT * DATA_COUNT)
BATCH_SIZE = 32
NUM_EPOCHS = 3
VALIDATION_SPLIT_PCT = 0.1
VALIDATION_SPLIT = int(VALIDATION_SPLIT_PCT * TRAIN_SPLIT)
LR_MODE = 1         			# 0="unchanged", 1="decay on plateau", 2="ramp up and ramp down"
REDUCE_LR_DECAY = 0.1
REDUCE_LR_PATIENCE = 3
LEARNING_RATE = 1e-4 * (REDUCE_LR_DECAY ** 0)
LEARNING_RATE_MIN = LEARNING_RATE * 1e-3
LEARNING_RATE_DECAY = 0.0
SCHEDULER_LEARNING_SCALE = 1.1
SCHEDULER_LEARNING_RATE = 1e-4
SCHEDULER_LEARNING_RAMPUP = 0.5
SCHEDULER_LEARNING_DECAY = 4
SCHEDULER_LEARNING_RATE_MIN = SCHEDULER_LEARNING_RATE * 1e-3
LABEL_SMOOTHENING = 0.0
LOSS_FUNCTION = "sparse_categorical_crossentropy"
EVALUATION_METRIC = "sparse_categorical_accuracy"
CHECKPOINT_PERIOD = 1
USE_TENSORBOARD = False

MODEL_NAME_SUFFIX = ".hdf5"
MODEL_CHECKPOINT_NAME_SUFFIX = "-{epoch:04d}-{val_" + EVALUATION_METRIC + ":.4f}" + MODEL_NAME_SUFFIX
MODEL_TRAINED_NAME_SUFFIX = "-Trained" + MODEL_NAME_SUFFIX


###translation parameters
GLOSSARY_UNK_REPLACE = True
MAX_TRANSLATION_LENGTH = 200
BEAM_SIZE = 8
LENGTH_PENALTY_COEFF = 0.2
COVERAGE_PENALTY_COEFF = 0.2
