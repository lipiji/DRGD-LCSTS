# -*- coding: utf-8 -*-

from os import makedirs
from os.path import exists
from commons import *
from transform_raw_lcsts import transform_lcsts_datasets
from extract_lcsts_news_info import write_w2i, write_lfw_embedding, write_eos_embedding, write_frequent_words
from write_lcsts_dataset_info import write_news
from shapes import *

shape_cls = LcstsCharOneSentShapes
LCSTS_TRAINING_PATH = TRAINING_DATA_PATH + "lcsts/"
LCSTS_TESTING_PATH = TESTING_DATA_PATH + "lcsts/"
LCSTS_VALIDATION_PATH = VALIDATION_DATA_PATH + "lcsts/"
LCSTS_CHAR_RESULT_PATH = RESULT_PATH + "lcsts_training/char/lcsts_testing/"

if not exists(LCSTS_TRAINING_PATH):
    makedirs(LCSTS_TRAINING_PATH)
if not exists(LCSTS_TESTING_PATH):
    makedirs(LCSTS_TESTING_PATH)
if not exists(LCSTS_VALIDATION_PATH):
    makedirs(LCSTS_VALIDATION_PATH)
if not exists(LCSTS_CHAR_RESULT_PATH):
    makedirs(LCSTS_CHAR_RESULT_PATH)
if not exists(ROOT_PATH + "model/"):
    makedirs(ROOT_PATH + "model/")
if not exists(ROOT_PATH + "tmp/"):
    makedirs(ROOT_PATH + "tmp/")
print "transform raw LCSTS dataset:"
transform_lcsts_datasets(LCSTS_PATH, shape_cls)
print "write w2i:"
uni_occur_times, uni_w2i = write_w2i(LCSTS_TRAINING_PATH, shape_cls, True, True)
print "write frequent chars:"
write_frequent_words(LCSTS_TRAINING_PATH, uni_occur_times, uni_w2i, 3000)
print "write LCSTS training data:"
write_news(LCSTS_TRAINING_PATH + "char_trimmed/", LCSTS_TRAINING_PATH + "info/", True)
print "write LCSTS testing data:"
write_news(LCSTS_TESTING_PATH + "char_trimmed/", LCSTS_TESTING_PATH + "info/", True)
print "write LCSTS validation data:"
write_news(LCSTS_VALIDATION_PATH + "char_trimmed/", LCSTS_VALIDATION_PATH + "info/", True)
print "write lfw embedding:"
write_lfw_embedding(shape_cls.DIM_X)
print "write eos embedding:"
write_eos_embedding(shape_cls.DIM_X)
