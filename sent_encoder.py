# -*- coding: utf-8 -*-
#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class SentEncoderLayer(object):
    def __init__(self, layer_input, sent_mask, shape):
        prefix = "SentEncoder_"
        self.sent_emb, self.features = layer_input
        self.in_size, self.num_features = shape

        self.b_features = init_real_num(prefix + "b_features")
        self.w_features = init_weights((self.num_features, 1), prefix + "w_features", sample = "uniform")
        #self.w_emb = init_weights((self.in_size, 1), prefix + "w_emb", sample = "uniform")

        self.sent_sim = (T.dot(self.features, self.w_features) + self.b_features) * sent_mask
        self.sent_sim = T.addbroadcast(self.sent_sim, self.sent_sim.ndim - 1)

        '''
        if is_pooling:
            threshold = T.sort(-self.sent_sim, axis = 0)[TOP_K - 1, :]
            self.sent_mask_top_k = T.ge(self.sent_sim, -threshold)
            self.sent_mask_top_k = T.addbroadcast(self.sent_mask_top_k, self.sent_mask_top_k.ndim - 1)
            self.doc_emb = T.sum(self.sent_sim * self.sent_emb * self.sent_mask_top_k, axis = 0, dtype = theano.config.floatX)
        else:
        '''
        self.doc_emb = T.sum(self.sent_sim * self.sent_emb, axis = 0, dtype = theano.config.floatX)
        
        self.params = [self.b_features, self.w_features]
        #self.params = [self.w_emb, self.b, self.w_features]
