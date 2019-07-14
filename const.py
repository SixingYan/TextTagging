PAD_TAG = "<PAD>"
START_TAG = "<START>"
STOP_TAG = "<STOP>"

import os
BASEPATH = os.getcwd()
DATAPATH = os.path.join(BASEPATH, 'Data/')
MODELPATH = os.path.join(BASEPATH, 'Model/')


tag_to_ix = {"a": 0, "b": 1, "c": 2, "o": 3,
             START_TAG: 4, STOP_TAG: 5, PAD_TAG: 6}
ix_to_tag = {tp[1]: tp[0] for tp in tag_to_ix.items()}
ix_to_tag[6] = "o"
ix_to_tag[5] = "o"
ix_to_tag[4] = "o"