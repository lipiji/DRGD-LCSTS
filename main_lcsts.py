# -*- coding: utf-8 -*-
#pylint: skip-file
import os
cudaid = 0
os.environ["THEANO_FLAGS"] = "device=cuda" + str(cudaid)

import sys
import time
import numpy as np
import cPickle as pickle
import copy
#from gensim.models import Word2Vec
from random import shuffle
import math

from data import *
from rnn import *
from utils_pg import *
from commons import *
from shapes import LcstsCharOneSentShapes

LCSTS_DICT_PATH = TRAINING_DATA_PATH + "lcsts/"
LCSTS_TRAINING_DATA_PATH = TRAINING_DATA_PATH + "lcsts/info/"
LCSTS_TESTING_DATA_PATH = TESTING_DATA_PATH + "lcsts/info/"
LCSTS_VALIDATION_DATA_PATH = VALIDATION_DATA_PATH + "lcsts/info/"

LCSTS_CHAR_RESULT_PATH = RESULT_PATH + "lcsts_training/char/lcsts_testing/"
LCSTS_CHAR_SUMM_PATH = LCSTS_CHAR_RESULT_PATH + "summ/"
LCSTS_CHAR_MODEL_PATH = LCSTS_CHAR_RESULT_PATH + "model/"
LCSTS_CHAR_CHINESE_SUMM_PATH = LCSTS_CHAR_RESULT_PATH + "chinese_summ/"
LCSTS_CHAR_CHINESE_MODEL_PATH = LCSTS_CHAR_RESULT_PATH + "chinese_model/"

def print_basic_info(modules, consts, options):
    if options["is_debugging"]:
        print "\nWARNING: IN DEBUGGING MODE\n"
    print "optimizer:", modules["optimizer"],
    if modules["optimizer"] != "adadelta":
        print ", lr:", consts["lr"]
    else:
        print
    
    if options["has_learnable_w2v"]:
        print "USE LEARNABLE W2V EMBEDDING"
    if options["is_bidirectional"]:
        print "USE BI-DIRECTIONAL RNN"
    if options["has_lvt_trick"]:
        print "USE LVT TRICK"

    for k in consts:
        print k + ":", consts[k]

def init_modules(shape_cls):
    options = {}

    options["is_debugging"] = False
    options["is_predicting"] = False
    options["use_testing_dataset"] = False # validating or testing

    options["is_unicode"] = True
    options["has_y"] = True

    options["has_lvt_trick"] = False
    options["has_learnable_w2v"] = True
    options["is_bidirectional"] = True
    options["beam_decoding"] = True # False for greedy decoding

    assert options["is_unicode"] == True

    consts = {}
    consts["idx_gpu"] = cudaid

    consts["dim_x"] = shape_cls.DIM_X
    consts["dim_y"] = shape_cls.DIM_Y
    consts["len_x"] = shape_cls.MAX_LEN_X + 1 # plus 1 for eos
    consts["len_y"] = shape_cls.MAX_LEN_Y + 1
    consts["min_len_predict"] = shape_cls.MIN_LEN_PREDICT
    consts["max_len_predict"] = shape_cls.MAX_LEN_PREDICT
    consts["num_x"] = shape_cls.MAX_NUM_X
    consts["num_y"] = shape_cls.NUM_Y
    consts["batch_size"] = 2 if options["is_debugging"] else shape_cls.BATCH_SIZE
    
    if options["is_debugging"]:
        consts["testing_batch_size"] = 1 if options["beam_decoding"] else 5
    else:
        consts["testing_batch_size"] = 1 if options["beam_decoding"] else shape_cls.TESTING_BATCH_SIZE 
  
    # for vae
    consts["updated_batch_size"] = consts["batch_size"]
    if options["is_predicting"]:
        consts["updated_batch_size"] = consts["testing_batch_size"]

    h_size = 500
    consts["hidden_size"] = [h_size]
    consts["latent_size"] = h_size
    consts["lvt_dict_size"] = shape_cls.LVT_DICT_SIZE

    consts["lr"] = 1.
    consts["beam_size"] = 10

    consts["max_epoch"] = 1000 if options["is_debugging"] else 30 
    consts["num_model"] = 1
    consts["print_time"] = 20
    consts["save_epoch"] = 1
    consts["testing_print_size"] = 200

    assert consts["dim_x"] == consts["dim_y"]
    assert consts["beam_size"] >= 1
    if options["has_lvt_trick"]:
        assert consts["lvt_dict_size"] != None
        assert consts["testing_batch_size"] <= consts["batch_size"]

    modules = {}

    modules["w2i"] = pickle.load(open(LCSTS_DICT_PATH + "uni_c2i", "r"))
    modules["i2w"] = pickle.load(open(LCSTS_DICT_PATH + "uni_i2c", "r"))
    if options["has_lvt_trick"]:
        modules["freq_words"] = pickle.load(open(LCSTS_DICT_PATH + "frequent_chars", "r"))

    if options["has_learnable_w2v"]:
        modules["w2v"] = None
        modules["lfw_emb"] = modules["w2i"][LOW_FREQ_WORD]
        modules["eos_emb"] = modules["w2i"][END_OF_SENT]
    else:
        #modules["w2v"] = Word2Vec.load_word2vec_format(ROOT_PATH + "raw_data/lcsts_char_w2v.bin", binary = True)
        modules["lfw_emb"] = pickle.load(open(ROOT_PATH + "tmp/lfw_emb_dim" + consts["dim_x"], "r"))
        modules["eos_emb"] = pickle.load(open(ROOT_PATH + "tmp/eos_emb_dim" + consts["dim_y"], "r"))

    # try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
    modules["optimizer"] = "adadelta" # adam的lr大了会出现nan

    return modules, consts, options

def greedy_decode(flist, batch, model, modules, consts, options, lvt_dict):
    dec_result = [[] for i in xrange(consts["testing_batch_size"])]
    num_left = consts["testing_batch_size"]

    word_emb, dec_state = model.encode(batch.x, batch.x_mask, consts["testing_batch_size"])
    next_y = -np.ones((1, consts["testing_batch_size"], 1), dtype="int64")

    for step in xrange(consts["max_len_predict"]):
        if num_left == 0:
            break
        if options["has_lvt_trick"]:
            y_pred, dec_state = model.decode_once(next_y, word_emb, dec_state, batch.x_mask, consts["testing_batch_size"], lvt_dict)
        else:
            y_pred, dec_state = model.decode_once(next_y, word_emb, dec_state, batch.x_mask, consts["testing_batch_size"])
        next_y = np.argmax(y_pred[:, :], axis = 1).reshape((1, consts["testing_batch_size"], 1))

        for idx_doc in xrange(consts["testing_batch_size"]):
            if dec_result[idx_doc] == None:
                continue

            idx_max = next_y[0][idx_doc][0]
            if options["has_lvt_trick"]:
                idx_max = lvt_dict[idx_max]
                next_y[0, idx_doc, 0] = idx_max
            if idx_max == modules["eos_emb"]:
                fname = str(flist[idx_doc])
                if len(dec_result[idx_doc]) >= consts["min_len_predict"]:
                    write_summ("".join((LCSTS_CHAR_SUMM_PATH, "summ.", fname)), dec_result[idx_doc], 1)
                    write_summ("".join((LCSTS_CHAR_CHINESE_SUMM_PATH, "summ.", fname)), dec_result[idx_doc], 1, modules["i2w"])

                    ly = batch.len_y[idx_doc]
                    y_true = batch.y[0 : ly, idx_doc, 0].tolist()
                    y_true = [str(i) for i in y_true[:-1]] # delete <eos>
                    write_summ("".join((LCSTS_CHAR_MODEL_PATH, "model.", fname)), y_true, 1)
                    write_summ("".join((LCSTS_CHAR_CHINESE_MODEL_PATH, "model.", fname)), y_true, 1, modules["i2w"])

                dec_result[idx_doc] = None
                num_left -= 1
            else:
                dec_result[idx_doc].append(str(idx_max))

    for idx_doc in xrange(consts["testing_batch_size"]):
        fname = str(flist[idx_doc])
        if dec_result[idx_doc] != None and len(dec_result[idx_doc]) >= consts["min_len_predict"]:
            write_summ("".join((LCSTS_CHAR_SUMM_PATH, "summ.", fname)), dec_result[idx_doc], 1)
            write_summ("".join((LCSTS_CHAR_CHINESE_SUMM_PATH, "summ.", fname)), dec_result[idx_doc], 1, modules["i2w"])

            ly = batch.len_y[idx_doc]
            y_true = batch.y[0 : ly, idx_doc, 0].tolist()
            y_true = [str(i) for i in y_true[:-1]] # delete <eos>
            write_summ("".join((LCSTS_CHAR_MODEL_PATH, "model.", fname)), y_true, 1)
            write_summ("".join((LCSTS_CHAR_CHINESE_MODEL_PATH, "model.", fname)), y_true, 1, modules["i2w"])

def beam_decode(fname, batch, model, modules, consts, options, lvt_dict):
    beam_size = consts["beam_size"]
    num_live = 1
    num_dead = 0
    samples = []
    sample_scores = np.zeros(beam_size, dtype = theano.config.floatX)

    last_traces = [[]]
    last_scores = np.zeros(1, dtype = theano.config.floatX)
    last_states = []
    last_states_z = []

    word_emb, dec_state = model.encode(batch.x,  batch.x_mask, consts["testing_batch_size"])
    next_y = -np.ones((1, num_live, 1), dtype="int64")
    dec_state_z = np.zeros((1, consts["latent_size"]), dtype = theano.config.floatX)

    for step in xrange(consts["max_len_predict"]):
        tile_word_emb = np.tile(word_emb, (num_live, 1))
        tile_x = np.tile(batch.x, (num_live, 1))

        if options["has_lvt_trick"]:
            y_pred, dec_state, dec_state_z = model.decode_once(next_y, tile_word_emb, dec_state, dec_state_z, tile_x, batch.x_mask, num_live, lvt_dict)
        else:
            y_pred, dec_state, dec_state_z = model.decode_once(next_y, tile_word_emb, dec_state, dec_state_z, tile_x, batch.x_mask, num_live)
        dict_size = y_pred.shape[-1]

        cand_scores = last_scores + np.log(y_pred) # 分数最大越好
        cand_scores = cand_scores.flatten()
        idx_top_joint_scores = np.argsort(cand_scores)[-(beam_size - num_dead):]

        idx_last_traces = idx_top_joint_scores / dict_size
        idx_word_now = idx_top_joint_scores % dict_size
        top_joint_scores = cand_scores[idx_top_joint_scores]

        traces_now = []
        scores_now = np.zeros((beam_size - num_dead), dtype = theano.config.floatX)
        states_now = []
        states_now_z = []

        for i, [j, k] in enumerate(zip(idx_last_traces, idx_word_now)):
            if options["has_lvt_trick"]:
                traces_now.append(last_traces[j] + [lvt_dict[k]])
            else:
                traces_now.append(last_traces[j] + [k])
            scores_now[i] = copy.copy(top_joint_scores[i])
            states_now.append(copy.copy(dec_state[j, :]))
            states_now_z.append(copy.copy(dec_state_z[j, :]))

        num_live = 0
        last_traces = []
        last_scores = []
        last_states = []
        last_states_z = []

        for i in xrange(len(traces_now)):
            if traces_now[i][-1] == modules["eos_emb"]:
                samples.append([str(e) for e in traces_now[i][:-1]])
                sample_scores[num_dead] = scores_now[i]
                num_dead += 1
            else:
                last_traces.append(traces_now[i])
                last_scores.append(scores_now[i])
                last_states.append(states_now[i])
                last_states_z.append(states_now_z[i])
                num_live += 1
        if num_live == 0 or num_dead >= beam_size:
            break

        last_scores = np.array(last_scores).reshape((num_live, 1))
        next_y = np.array([e[-1] for e in last_traces], dtype = "int64").reshape((1, num_live, 1))
        dec_state = np.array(last_states, dtype = theano.config.floatX).reshape((num_live, dec_state.shape[-1]))
        dec_state_z = np.array(last_states_z, dtype = theano.config.floatX).reshape((num_live, dec_state_z.shape[-1]))
        assert num_live + num_dead == beam_size

    if num_live > 0:
        for i in xrange(num_live):
            samples.append([str(e) for e in last_traces[i]])
            sample_scores[num_dead] = last_scores[i]
            num_dead += 1

    #weight by length
    for i in xrange(len(sample_scores)):
        sent_len = len(samples[i])
        # Google's Neural Machine Translation System
        lpy = math.pow((10 + sent_len), 0.6) / math.pow((10 + 1), 0.6)
        sample_scores[i] = sample_scores[i]  / lpy

    idx_sorted_scores = np.argsort(sample_scores) # 低分到高分
    ly = batch.len_y[0]
    y_true = batch.y[0 : ly, 0, 0].tolist()
    y_true = [str(i) for i in y_true[:-1]] # delete <eos>

    sorted_samples = []
    sorted_scores = []
    filter_idx = []
    for e in idx_sorted_scores:
        if len(samples[e]) >= consts["min_len_predict"]:
            filter_idx.append(e)
    if len(filter_idx) == 0:
        filter_idx = idx_sorted_scores
    for e in filter_idx:
        sorted_samples.append(samples[e])
        sorted_scores.append(sample_scores[e])

    num_samples = len(sorted_samples)
    if len(sorted_samples) == 1:
        sorted_samples = sorted_samples[0]
        num_samples = 1
    
    try:
	    write_summ("".join((LCSTS_CHAR_CHINESE_SUMM_PATH, "summ.", fname)), sorted_samples, num_samples, modules["i2w"], sorted_scores)
	    write_summ("".join((LCSTS_CHAR_CHINESE_MODEL_PATH, "model.", fname)), y_true, 1, modules["i2w"])
	    write_summ("".join((LCSTS_CHAR_SUMM_PATH, "summ.", fname)), sorted_samples[-1], 1)
	    write_summ("".join((LCSTS_CHAR_MODEL_PATH, "model.", fname)), y_true, 1)
    except Exception as e:
        print fname
        print sorted_samples
        print sorted_scores
        raise

def predict(model, modules, consts, options):
    if options["beam_decoding"]:
        print "using beam search"
    else:
        print "using greedy search"
    rebuild_dir(LCSTS_CHAR_SUMM_PATH)
    rebuild_dir(LCSTS_CHAR_MODEL_PATH)
    rebuild_dir(LCSTS_CHAR_CHINESE_SUMM_PATH)
    rebuild_dir(LCSTS_CHAR_CHINESE_MODEL_PATH)  
    
    if options["use_testing_dataset"]:
        print "start predicting..."
        dg = DataGen(ROOT_PATH, LCSTS_TESTING_DATA_PATH, modules, consts, options)
    else:
        print "start validating..."
        dg = DataGen(ROOT_PATH, LCSTS_VALIDATION_DATA_PATH, modules, consts, options)

    num_files = len(dg.raw_data)
    num_batches = dg.num_batch

    running_start = time.time()

    partial_num = 0
    total_num = 0
    for idx_batch in xrange(num_batches):
        try:
            if options["has_lvt_trick"]:
                flist, batch, lvt_dict = dg.get_data(modules, consts, options)
                if batch == None and lvt_dict == None:
                    print "SKIP THIS BATCH\n"
                    continue
            else:
                flist, batch = dg.get_data(modules, consts, options)
                lvt_dict = None
        except (Exception) as e:
            print e
            print "SKIP THIS BATCH\n"
            continue

        if options["beam_decoding"]:
            beam_decode(str(flist[0]), batch, model, modules, consts, options, lvt_dict)
        else:
            greedy_decode(flist, batch, model, modules, consts, options, lvt_dict)

        partial_num += consts["testing_batch_size"]
        total_num += consts["testing_batch_size"]
        if partial_num >= consts["testing_print_size"]:
            print total_num, "summs are generated"
            partial_num = 0

def run(existing_model_name = None):
    shape_cls = LcstsCharOneSentShapes
    modules, consts, options = init_modules(shape_cls)

    if options["is_predicting"]:
        need_load_model = True
        training_model = False
        predict_model = True
    else:
        need_load_model = False
        training_model = True
        predict_model = False

    print_basic_info(modules, consts, options)

    if training_model:
        dg = DataGen(ROOT_PATH, LCSTS_TRAINING_DATA_PATH, modules, consts, options)
        num_files = len(dg.raw_data)
        num_batches = dg.num_batch

    running_start = time.time()
    for idx_model in xrange(consts["num_model"]):
        
        print "compiling model " + str(idx_model + 1) 
        model = RNN(modules, consts, options)

        model_name = "w2v.lcsts.seq2seq"
        existing_epoch = 0
        if need_load_model:
            if existing_model_name == None:
                existing_model_name = "w2v.lcsts.seq2seq.gpu5.epoch5.20"
            print "loading existed model:", existing_model_name
            model = load_model(ROOT_PATH + "model/" + existing_model_name, model)

        if training_model:
            print "start training model " + str(idx_model + 1) 
            print_size = num_files / consts["print_time"] if num_files >= consts["print_time"] else num_files

            last_total_error = float("inf")
            print "max epoch:", consts["max_epoch"]
            for epoch in xrange(0, consts["max_epoch"]):
                print "epoch: ", epoch + existing_epoch
                num_partial = 1
                total_error = 0.0
                partial_num_files = 0
                epoch_start = time.time()
                partial_start = time.time()

                for idx_batch in xrange(num_batches):
                    try:
                        if options["has_lvt_trick"]:
                            _, batch, lvt_dict = dg.get_data(modules, consts, options)
                            if batch == None and lvt_dict == None:
                                print "SKIP THIS BATCH\n"
                                continue
                        else:
                            _, batch = dg.get_data(modules, consts, options)
                    except (Exception) as e:
                        print e
                        print "SKIP THIS BATCH\n"
                        continue

                    if options["has_lvt_trick"]:
                        cost, a, b, y_pred = model.train(batch.x, batch.y, batch.x_mask, batch.y_mask, consts["batch_size"], consts["lr"], lvt_dict, batch.y_lvt)
                    else:
                        cost, a, b, c, d, y_pred = model.train(batch.x, batch.y, batch.x_mask, batch.y_mask, consts["batch_size"], consts["lr"])
                    
                    #print cost, a, b, c, d
                    total_error += cost
                    partial_num_files += consts["batch_size"]
                    if partial_num_files / print_size == 1 and idx_batch < num_batches:
                        print idx_batch + 1, "batches have been processed,", 
                        print "average cost until now:", "cost =", total_error / (idx_batch + 1), ",", 
                        print "time:", time.time() - partial_start
                        partial_num_files = 0
                        if not options["is_debugging"]:
                            print "save model... ",
                            save_model(ROOT_PATH + "model/" + model_name + ".gpu" + str(consts["idx_gpu"]) + ".epoch" + str(epoch / consts["save_epoch"] + existing_epoch) + "." + str(num_partial), model)
                            print "finished"
                        num_partial += 1
                
                print "in this epoch, total average cost =", total_error / (idx_batch + 1), ",", 
                print "time:", time.time() - epoch_start
                if options["has_lvt_trick"]:
                    print_sent_dec(y_pred, batch.y, batch.y_mask, modules, consts, options, lvt_dict)
                else:
                    print_sent_dec(y_pred, batch.y, batch.y_mask, modules, consts, options)
                if last_total_error > total_error or options["is_debugging"]:
                    last_total_error = total_error
                    if not options["is_debugging"]:
                        print "save model... ",
                        save_model(ROOT_PATH + "model/" + model_name + ".gpu" + str(consts["idx_gpu"]) + ".epoch" + str(epoch / consts["save_epoch"] + existing_epoch) + "." + str(num_partial), model)
                        print "finished"
                else:
                    print "optimization finished"
                    break
            print "save final model... ",
            save_model(ROOT_PATH + "model/" + model_name + "final.gpu" + str(consts["idx_gpu"]) + ".epoch" + str(epoch / consts["save_epoch"] + existing_epoch) + "." + str(num_partial), model)
            print "finished"
        else:
            print "skip training model"

        if predict_model:
            predict(model, modules, consts, options)
    print "Finished, time:", time.time() - running_start

if __name__ == "__main__":
    np.set_printoptions(threshold = np.inf)
    existing_model_name = sys.argv[1] if len(sys.argv) > 1 else None
    run(existing_model_name)
