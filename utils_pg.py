# -*- coding: utf-8 -*-
#pylint: skip-file
import numpy as np
from numpy.random import random as rand
import theano
import theano.tensor as T
import cPickle as pickle
import sys
import os
import shutil
from copy import deepcopy
from utils_preprocess import washer
from commons import *


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_normal_weight(shape, scale=0.01):
    return np.random.normal(loc=0.0, scale=scale, size=shape)

def init_uniform_weight(shape):
    return np.random.uniform(-0.1, 0.1, shape)

def init_xavier_weight_uniform(shape):
    return np.random.uniform(-np.sqrt(6. / (shape[0] + shape[1])), np.sqrt(6. / (shape[0] + shape[1])), shape)

def init_xavier_weight(shape):
    fan_in, fan_out = shape
    s = np.sqrt(2. / (fan_in + fan_out))
    return init_normal_weight(shape, s)

def init_ortho_weight(shape):
    W = np.random.normal(0.0, 1.0, (shape[0], shape[0]))
    u, s, v = np.linalg.svd(W)
    return u

def init_weights(shape, name, sample = "xavier", num_concatenate = 1, axis_concatenate = -1):
    if sample == "uniform":
        if num_concatenate == 1:
            values = init_uniform_weight(shape)
        elif num_concatenate > 1:
            l = []
            for i in range(num_concatenate):
                l.append(init_uniform_weight(shape))
            values = np.concatenate(l, axis = axis_concatenate)
        else:
            raise RuntimeError("Wrong num_concatenate:" + str(num_concatenate))

    elif sample == "normal":
        if num_concatenate == 1:
            values = init_normal_weight(shape)
        elif num_concatenate > 1:
            l = []
            for i in range(num_concatenate):
                l.append(init_normal_weight(shape))
            values = np.concatenate(l, axis = axis_concatenate)
        else:
            raise RuntimeError("Wrong num_concatenate:" + str(num_concatenate))

    elif sample == "xavier":
        if num_concatenate == 1:
            values = init_xavier_weight(shape)
        elif num_concatenate > 1:
            l = []
            for i in range(num_concatenate):
                l.append(init_xavier_weight(shape))
            values = np.concatenate(l, axis = axis_concatenate)
        else:
            raise RuntimeError("Wrong num_concatenate:" + str(num_concatenate))

    elif sample == "ortho":
        if num_concatenate == 1:
            values = init_ortho_weight(shape)
        elif num_concatenate > 1:
            l = []
            for i in range(num_concatenate):
                l.append(init_ortho_weight(shape))
            values = np.concatenate(l, axis = axis_concatenate)
        else:
            raise RuntimeError("Wrong num_concatenate:" + str(num_concatenate))

    else:
        raise ValueError("Unsupported initialization scheme: %s" % sample)

    return theano.shared(floatX(values), name)

def init_gradws(shape, name):
    return theano.shared(floatX(np.zeros(shape)), name)

def init_bias(size, name, num_concatenate = 1):
    if num_concatenate >= 1:
        values = np.zeros((size * num_concatenate,))
    else:
        raise RuntimeError("Wrong num_concatenate:" + str(num_concatenate))
    return theano.shared(floatX(values), name)

def init_real_num(name):
    return theano.shared(rand(), name)

def rebuild_dir(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
        except OSError:
            pass
    os.mkdir(path)

def save_model(f, model):
    ps = {}
    for p in model.params:
        ps[p.name] = p.get_value()
    if model.sub_params != None:
        for p in model.sub_params:
            ps[p[0].name] = p[0].get_value()
    pickle.dump(ps, open(f, "wb"), protocol = pickle.HIGHEST_PROTOCOL)

def load_model(f, model):
    ps = pickle.load(open(f, "rb"))
    for p in model.params:
        p.set_value(ps[p.name])
    if model.sub_params != None:
        for p in model.sub_params:
            p[0].set_value(ps[p[0].name])
    return model

def check_nan(x):
    b = np.isnan(x).flatten().tolist()
    for e in b:
        if e:
            print "is nan"
            return True

    print "is not nan"
    return False

def write_tensor3(path, tensor):
    with file(path, "w") as f_dst:
        f_dst.write("# Array shape: {0}\n".format(tensor.shape))
        for i in tensor:
            np.savetxt(f_dst, i)
            f_dst.write("# New Slice\n")

def print_sent_dec(y_pred, y, y_mask, modules, consts, options, lvt_dict = None):
    print "golden truth and prediction samples:"
    max_y_words = np.sum(y_mask, axis = 0)
    max_y_words = max_y_words.reshape((consts["batch_size"]))
    max_num_docs = 16 if consts["batch_size"] > 16 else consts["batch_size"]
    is_unicode = options["is_unicode"]
    for idx_doc in range(max_num_docs):
        print idx_doc + 1, "----------------------------------------------------------------------------------------------------"
        sent_true= ""
        for idx_word in range(max_y_words[idx_doc]):
            i = y[idx_word, idx_doc, 0] if options["has_learnable_w2v"] else np.argmax(y[idx_word, idx_doc, :]) 
            sent_true += modules["i2w"][i]
        if is_unicode:
            print sent_true.encode("utf-8")
        else:
            print sent_true

        print
        sent_pred = ""
        for idx_word in range(max_y_words[idx_doc]):
            i = np.argmax(y_pred[idx_word, idx_doc, :])
            if options["has_lvt_trick"]:
                i = lvt_dict[i]
            sent_pred += modules["i2w"][i]
            if not is_unicode:
                sent_pred += " "
        if is_unicode:
            print sent_pred.encode("utf-8")
        else:
            print sent_pred
    print "----------------------------------------------------------------------------------------------------"
    print 

def write_summ(dst_path, summ_list, num_summ, i2w = None, score_list = None):
    assert num_summ > 0
    with open(dst_path, "w") as f_summ:
        if num_summ == 1:
            if score_list != None:
                f_summ.write(str(score_list[0]))
                f_summ.write("\t")
            if i2w != None:
                #for e in summ_list:
                #    print i2w[int(e)],
                #print "\n"
                s = u"".join([i2w[int(e)] for e in summ_list]).encode("utf-8")
            else:
                s = " ".join(summ_list)
            f_summ.write(s)
            f_summ.write("\n")
        else:
            assert num_summ == len(summ_list)
            if score_list != None:
                assert num_summ == len(score_list)

            for i in xrange(num_summ):
                if score_list != None:
                    f_summ.write(str(score_list[i]))
                    f_summ.write("\t")
                if i2w != None:
                    #for e in summ_list[i]:
                    #    print i2w[int(e)],
                    #print "\n"
                    s = u"".join([i2w[int(e)] for e in summ_list[i]]).encode("utf-8")
                else:
                    s = " ".join(summ_list[i])
                f_summ.write(s)
                f_summ.write("\n")

# p = ROOT_PATH + "training_data/agiga/agiga_small/"
# total_w2i = pickle.load(open(p + "uni_w2i", "r"))
# g = BatchDictGen(p, total_w2i, 3000)
# l = make_file_list(p + "info/", 8)
# a, b, c = g.get_dict(l[0])
# print len(a), len(b), c[0 : 50]
