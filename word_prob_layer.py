# -*- coding: utf-8 -*-
#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class WordProbLayer(object):
    def __init__(self, layer_input, shape, is_predicting, has_lvt_trick):
        prefix = "WordProbLayer_"
        if has_lvt_trick:
            self.ds, self.ac, self.y_emb, self.cp_idx, self.dec_z, self.lvt_dict = layer_input
            self.hidden_size, self.dim_ac, self.dim_y, self.dict_size, self.latent_size, self.lvt_dict_size = shape
        else:
            self.ds, self.ac, self.y_emb, self.cp_idx, self.dec_z = layer_input
            self.hidden_size, self.dim_ac, self.dim_y, self.dict_size, self.latent_size = shape

        self.w_ds = init_weights((self.hidden_size, self.dim_y), prefix + "W_ds", sample = "xavier")
        self.b_ds = init_bias(self.dim_y, prefix + "b_ds")
        self.w_ac = init_weights((self.dim_ac, self.dim_y), prefix + "W_ac", sample = "xavier")
        self.w_y = init_weights((self.dim_y, self.dim_y), prefix + "W_y", sample = "xavier")
        self.w_logit = init_weights((self.dim_y, self.dict_size), prefix + "W_logit", sample = "xavier")
        self.b_logit = init_bias(self.dict_size, prefix + "b_logit")
        
        self.W_zh = init_weights((self.latent_size, self.hidden_size), prefix + "W_zh")
        self.b_zh = init_bias(self.hidden_size, prefix + "b_zh")
        self.W_hu = init_weights((self.hidden_size, self.hidden_size), prefix + "W_hu")
        self.W_zu = init_weights((self.hidden_size, self.hidden_size), prefix + "W_zu")
        self.b_zu = init_bias(self.hidden_size, prefix + "b_zu")
       
        self.params = [self.w_ds, self.b_ds, self.w_ac, self.w_y,
                       self.W_zh, self.b_zh, self.W_hu, self.W_zu, self.b_zu]

        if has_lvt_trick:
            self.sub_w_logit = self.w_logit[:, self.lvt_dict]
            self.sub_b_logit = self.b_logit[self.lvt_dict]

            self.sub_params = [(self.w_logit, self.sub_w_logit, (self.dim_y, self.lvt_dict_size)),
                (self.b_logit, self.sub_b_logit, (self.lvt_dict_size, ))]
        else:
            self.params += [self.w_logit, self.b_logit]

        
        # vae decoder
        z_h = T.tanh(T.dot(self.dec_z, self.W_zh) + self.b_zh)
        u = T.nnet.sigmoid(T.dot(self.ds, self.W_hu) + T.dot(z_h, self.W_zu) + self.b_zu)
        self.ds = self.ds * u + (1 - u) * z_h

        logit = T.dot(self.ds, self.w_ds) + T.dot(self.y_emb, self.w_y) + T.dot(self.ac, self.w_ac) + self.b_ds
        logit = T.tanh(logit)
        logit = T.dot(logit, self.sub_w_logit) + self.sub_b_logit if has_lvt_trick else T.dot(logit, self.w_logit) + self.b_logit

        old_shape = logit.shape


        if is_predicting:
            self.y_pred = T.nnet.softmax(logit)
        else:
            self.y_pred = T.nnet.softmax(logit.reshape((old_shape[0] * old_shape[1], old_shape[2]))).reshape(old_shape)


