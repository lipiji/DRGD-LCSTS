# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import os.path
import time
from operator import itemgetter
import theano
import numpy as np
import cPickle as pickle
from random import shuffle
from utils_preprocess import *

class BatchData:
    def __init__(self, flist, raw_data, modules, consts, options, lvt_i2i = None):
        self.batch_size = consts["testing_batch_size"] if options["is_predicting"] else consts["batch_size"] 
        if options["has_learnable_w2v"]:
            self.x = np.zeros((consts["len_x"], self.batch_size, 1), dtype = np.int64)
            self.y = np.zeros((consts["len_y"], self.batch_size, 1), dtype = np.int64)
            if lvt_i2i != None:
                self.y_lvt = np.zeros((consts["len_y"], self.batch_size, 1), dtype = np.int64)
        else:
            assert lvt_i2i == None
            self.x = np.zeros((consts["len_x"], self.batch_size, consts["dim_x"]), dtype = theano.config.floatX)
            self.y = np.zeros((consts["len_y"], self.batch_size, consts["dim_y"]), dtype = theano.config.floatX)
        self.x_mask = np.zeros((consts["len_x"], self.batch_size, 1), dtype = theano.config.floatX)
        if options["is_predicting"]:
            self.y_mask = None
        else:
            self.y_mask = np.zeros((consts["len_y"], self.batch_size, 1), dtype = theano.config.floatX)
        self.len_y = []

        for idx_doc in xrange(len(flist)):
            try:
                rd = raw_data[flist[idx_doc]]
            except (Exception) as e:
                print idx_doc
                print e
                raise

            for idx_sent in xrange(len(rd.x)):
                sent = rd.x[idx_sent]

                for idx_word in xrange(len(sent)):
                    w = sent[idx_word]
                    if options["has_learnable_w2v"]:
                            self.x[idx_word, idx_doc, 0] = w
                    else:
                        embedding = modules["w2v"][w] if w in modules["w2v"].vocab else modules["lfw_emb"]
                        self.x[idx_word, idx_doc, :] = embedding
                    self.x_mask[idx_word, idx_doc, 0] = 1

            if options["has_y"]:
                for idx_word in xrange(len(rd.y)):
                    w = rd.y[idx_word]
                    if options["has_learnable_w2v"]:
                        self.y[idx_word, idx_doc, 0] = w
                        if lvt_i2i != None:
                            self.y_lvt[idx_word, idx_doc, 0] = lvt_i2i[w]
                    else:
                        self.y[idx_word, idx_doc, w] = 1
                    if not options["is_predicting"]:
                        self.y_mask[idx_word, idx_doc, 0] = 1
                self.len_y.append(len(rd.y))
            else:
                self.y = self.y_mask = None

class RawData(object):
    def __init__(self, fdata, modules, options):
        x, y = fdata
        self.file_related_words = set() if options["has_lvt_trick"] else None

        if options["has_learnable_w2v"]:
            self.y = []
            for w in y:
                if w not in modules["w2i"]:
                    continue
                i = modules["w2i"][w]
                self.y.append(i)
                if options["has_lvt_trick"]:
                    self.file_related_words.add(i)
            self.y.append(modules["eos_emb"])
        else:
            self.y = y
            self.y.append(modules["eos_emb"])
    
        if options["has_learnable_w2v"]:
            self.x = []
            for con in x:
                tmp = []
                for w in con:
                    i = modules["w2i"][w] if w in modules["w2i"] else modules["lfw_emb"]
                    tmp.append(i)
                    if options["has_lvt_trick"]:
                        self.file_related_words.add(i)
                tmp.append(modules["eos_emb"])
                self.x.append(tmp)
            if options["has_lvt_trick"]:
                self.file_related_words.add(modules["eos_emb"])
        else:
            self.x = x
            self.x.append(modules["eos_emb"])

class DataGen(object):
    def _read_file(self, src_path, options):
        try:
            y, x = read_info(src_path, options["has_y"], options["is_unicode"])
        except Exception, e:
            print e
            print "EXCEPTION: wrong data files:", src_path
            raise
        if y == None:
            raise IOError("No abstract")
        elif len(x) == 0:
            raise IOError("No content")

        return (x, y)

    def __init__(self, root_path, src_path, modules, consts, options):
        self.batch_size = consts["testing_batch_size"] if options["is_predicting"] else consts["batch_size"] 
        if options["has_lvt_trick"]:
            self.freq_words = [modules["w2i"][e] for e in modules["freq_words"]]
            self.lvt_dict_size = consts["lvt_dict_size"]
        else:
            self.lvt_dict_size = 0
        tmp = src_path.split("/")
        if  options["is_predicting"]:
            mode = "predicting" if options["use_testing_dataset"]  else "validating"
        else:
            mode = "training"
        pickle_name = ",".join(("DataGen::raw_data", mode, tmp[-3], tmp[-2], str(options["is_debugging"]),
            str(options["has_y"]), str(options["is_unicode"]), str(options["has_learnable_w2v"]),
            str(options["has_lvt_trick"]), str(self.batch_size), str(self.lvt_dict_size)))
        pickle_name = root_path + "tmp/" + pickle_name
        print pickle_name

        running_start = time.time()
        if os.path.exists(pickle_name):
            print "loading existing DataGen...",
            self.fnames, self.raw_data = pickle.load(open(pickle_name, "r"))
            print "finished, time:", time.time() - running_start
        else:
            print "initialize new DataGen"
            print "start loading files... ",
            self.all_files = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"] if options["is_debugging"] else os.listdir(src_path)
            if options["is_predicting"] and not options["use_testing_dataset"]:
                self.all_files = [str(x) for x in xrange(500)] # just for validating

            self.fnames = []
            self.raw_data = [None] * len(self.all_files)
            batch = []
            data = []
            i = 0
            j = 0
            print "finished,", len(self.all_files), "files in all, time:", time.time() - running_start
            print "start building data structure..."
            for fname in self.all_files:
                if fname[0] == ".":
                    continue

                full_path = src_path + fname
                try:
                    tmp = self._read_file(full_path, options)
                except (Exception) as ex:
                    print "WARNING: bad file", full_path
                    print "Exception:", ex
                    print "Skip this file"
                    continue
                batch.append(fname)
                data.append(tmp)
                i += 1

                if i >= self.batch_size:
                    for fname, fdata in zip(batch, data):
                        fname = int(fname)
                        self.raw_data[fname] = RawData(fdata, modules, options)
                        self.fnames.append(fname)
                    batch = []
                    data = []
                    i = 0
                j += 1
                if j % 50000 == 0:
                    print j, "files have been processed, time:", time.time() - running_start

            print "start dumping... ",
            pickle.dump((self.fnames, self.raw_data), open(pickle_name, "wb"), protocol = pickle.HIGHEST_PROTOCOL)
            print "finished, time:", time.time() - running_start

        self._reset_data()
        self.num_batch = len(self.fnames) / self.batch_size
        print "finish initializing DataGen,", self.num_batch, "batches are generated"

    def _reset_data(self):
        shuffle(self.fnames)
        self.idx_batch = 0

    def get_data(self, modules, consts, options):
        if self.idx_batch >= self.num_batch:
            self._reset_data()
        flist = self.fnames[self.idx_batch * self.batch_size : (self.idx_batch + 1) * self.batch_size]
        self.idx_batch += 1

        if options["has_lvt_trick"]:
            lvt_words = set()
            for e in flist:
                lvt_words |= self.raw_data[e].file_related_words

            if len(lvt_words) > self.lvt_dict_size:
                print "WARNING: LVT_DICT_SIZE is too small"
                print "len(lvt_words):", len(lvt_words)
                print "self.lvt_dict_size:", self.lvt_dict_size
                print "fname : len(file_related_words) in flist:"
                tmp = []
                for e in flist:
                    tmp.append((e, len(self.raw_data[e].file_related_words)))
                tmp.sort(key = itemgetter(1))
                for e in tmp:
                    print e[0], ":", e[1], ",\t", 
                print
                return flist, None, None

            len_diff = self.lvt_dict_size - len(lvt_words)
            if len_diff > len(self.freq_words):
                print "WARNING: LVT_DICT_SIZE is too large"
                print "lvt_dict_size:", self.lvt_dict_size
                print "len(lvt_words):", len(lvt_words)
                print "len(self.freq_words):", len(self.freq_words)
                return flist, None, None

            i = 0
            j = 0
            while j < len_diff:
                if self.freq_words[i] not in lvt_words:
                    lvt_words.add(self.freq_words[i])
                    j += 1
                i += 1

            lvt_i2i = {}
            lvt_dict = []
            i = 0
            for e in lvt_words:
                lvt_i2i[e] = i
                lvt_dict.append(e)
                i += 1
            lvt_dict = np.array(lvt_dict, dtype = np.int64)

            return flist, BatchData(flist, self.raw_data, modules, consts, options, lvt_i2i), lvt_dict
        else:
            return flist, BatchData(flist, self.raw_data, modules, consts, options)
