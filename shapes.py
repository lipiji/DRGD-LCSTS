# -*- coding: utf-8 -*-
#pylint: skip-file

class LcstsCharOneSentShapes(object):
    DIM_X = 350
    DIM_Y = DIM_X
    MIN_LEN_X = 5
    MIN_LEN_Y = 5
    MAX_LEN_X = 120 + 1
    MAX_LEN_Y = 25 + 1
    MIN_LEN_PREDICT = 10
    MAX_LEN_PREDICT = 20
    MIN_NUM_X = 1
    MAX_NUM_X = 1
    NUM_Y = 1
    BATCH_SIZE = 300
    TESTING_BATCH_SIZE = 1 #241 # 71 * 10 == 710, 241 * 3 = 723, TESTING_BATCH_SIZE <= BATCH_SIZE for LVT

    #LVT_DICT_SIZE = 200
    LVT_DICT_SIZE = 3000

    UNI_LOW_FREQ_THRESHOLD = 0
    BI_LOW_FREQ_THRESHOLD = 15
