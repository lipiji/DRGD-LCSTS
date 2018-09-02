# -*- coding: utf-8 -*-
import operator
import numpy as np
import cPickle as pickle
import os
from utils_preprocess import read_news
from utils_pg import rebuild_dir
from commons import *

def write_lfw_embedding(dim):
    print "write lfw embedding"
    lfw_emb = np.random.uniform(-0.25, 0.25, dim)
    pickle.dump(lfw_emb, open(ROOT_PATH + "tmp/lfw_emb_dim" + str(dim), "wb"))

def write_eos_embedding(dim):
    print "write eos embedding"
    eos_emb = np.random.uniform(-0.25, 0.25, dim)
    pickle.dump(eos_emb, open(ROOT_PATH + "tmp/eos_emb_dim" + str(dim), "wb"))

def count_gram(sentence, uni_occur_times, bi_occur_times, bigram_delimiter = None):
    words = sentence.split()
    for e in words:
        if e in uni_occur_times:
            uni_occur_times[e] += 1
        else:
            uni_occur_times[e] = 1
    if bi_occur_times != None:
        for i in xrange(len(words) - 1):
            if bigram_delimiter == None:
                e = words[i] + words[i + 1]
            else:
                e = words[i] + bigram_delimiter + words[i + 1]
            if e in bi_occur_times:
                bi_occur_times[e] += 1
            else:
                bi_occur_times[e] = 1

def write_w2i(dataset_root, shape_cls, is_unicode, is_char):
    dst_names = ("uni_char_occur_time", "uni_c2i", "uni_i2c") if is_unicode and is_char else ("uni_word_occur_time", "uni_w2i", "uni_i2w")
    src_root = dataset_root + "char_trimmed/" if is_unicode and is_char else dataset_root + "word_trimmed/"
    print "write w2i:", src_root
    fnames = os.listdir(src_root)
    print "total number:", len(fnames)
    uni_occur_times = {}

    i = 0    
    for fname in fnames:
        abstract, contents = read_news(src_root + fname, True, is_unicode)
        count_gram(abstract, uni_occur_times, None)
        for e in contents:
            count_gram(e, uni_occur_times, None)
        i += 1
        if i % 50000 == 0:
            print i, "files have been counted"

    uni_w2i = {}
    uni_i2w = {}
    dic_size = 0
    for e in uni_occur_times:
        if uni_occur_times[e] <= shape_cls.UNI_LOW_FREQ_THRESHOLD:
            continue
        uni_w2i[e] = dic_size
        dic_size += 1
    uni_w2i[LOW_FREQ_WORD] = dic_size
    uni_w2i[END_OF_SENT] = dic_size + 1

    for k in uni_w2i:
        uni_i2w[uni_w2i[k]] = k

    print "uni_w2i size:", len(uni_w2i)
    pickle.dump(uni_occur_times, open(dataset_root + dst_names[0], "wb"))
    pickle.dump(uni_w2i, open(dataset_root + dst_names[1], "wb"))
    pickle.dump(uni_i2w, open(dataset_root + dst_names[2], "wb"))

    return uni_occur_times, uni_w2i

def write_frequent_words(dataset_root, uni_occur_times, uni_w2i, top_k):
    print "write frequent words:", dataset_root
    words = sorted(uni_occur_times.items(), key = operator.itemgetter(1), reverse = True)
    results = []
    i = 0
    j = 0
    while i < top_k:
        tmp = words[j][0]
        if tmp != None and tmp in uni_w2i:
            results.append(tmp)
            i += 1
        j += 1

    pickle.dump(results, open(dataset_root + "frequent_chars", "wb"))
