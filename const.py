PAD_TAG = "<PAD>"
START_TAG = "<START>"
STOP_TAG = "<STOP>"

import os
BASEPATH = os.getcwd()
DATAPATH = os.path.join(BASEPATH, 'Data/')
MODELPATH = os.path.join(BASEPATH, 'Model/')


TAG_TO_IX = {"a": 0, "b": 1, "c": 2, "o": 3,
             START_TAG: 4, STOP_TAG: 5, PAD_TAG: 6}
