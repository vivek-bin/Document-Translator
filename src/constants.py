from inspect import getsourcefile
from os.path import abspath
from os.path import dirname

#paths
PATH = dirname(dirname(dirname(abspath(getsourcefile(lambda:0))))) + "\\"
PROJECT = dirname(dirname(abspath(getsourcefile(lambda:0)))) + "\\"

DATA = PATH + "data\\"

EUROPARL = DATA + "EuroParl\\"
HANSARDS = DATA + "hansard\\"

EUROPARL_EN = EUROPARL + "europarl-v7.fr-en.en"
EUROPARL_FR = EUROPARL + "europarl-v7.fr-en.fr"

HANSARDS_HOUSE = HANSARDS + "sentence-pairs\\house\\debates\\"
HANSARDS_SENATE = HANSARDS + "sentence-pairs\\senate\\debates\\"


HANSARDS_HOUSE_TRAIN = HANSARDS_HOUSE + "training\\"
HANSARDS_HOUSE_TEST = HANSARDS_HOUSE + "testing\\"
HANSARDS_SENATE_TRAIN = HANSARDS_SENATE + "training\\"
HANSARDS_SENATE_TEST = HANSARDS_SENATE + "testing\\"



