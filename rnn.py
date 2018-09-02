# -*- coding: utf-8 -*-
#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import sys

from utils_pg import *
from word_encoder import *
from word_decoder import *
from word_prob_layer import *
from updates import *

class RNN(object):
    def __init__(self, modules, consts, options):
        if options["has_learnable_w2v"]:
            self.x = T.tensor3("x", dtype = "int64")
            self.y = T.tensor3("y", dtype = "int64")
        else:
            self.x = T.tensor3("x")
            self.y = T.tensor3("y")
        self.x_mask = T.tensor3("x_mask")
        self.x_mask = T.addbroadcast(self.x_mask, self.x_mask.ndim - 1)
        self.y_mask = T.tensor3("y_mask")
        self.y_mask = T.addbroadcast(self.y_mask, self.y_mask.ndim - 1)
        self.lr = T.scalar("lr")
        self.batch_size = T.iscalar("batch_size")
        self.updated_batch_size = consts["updated_batch_size"]
        
        self.has_learnable_w2v = options["has_learnable_w2v"]
        self.is_predicting = options["is_predicting"]
        self.is_bidirectional = options["is_bidirectional"]
        self.has_lvt_trick = options["has_lvt_trick"]
        self.beam_decoding = options["beam_decoding"]

        self.dim_x = consts["dim_x"]
        self.dim_y = consts["dim_y"]
        self.len_x = consts["len_x"]
        self.len_y = consts["len_y"]
        self.hidden_size = consts["hidden_size"]
        self.latent_size = consts["latent_size"]
        self.ctx_size = self.hidden_size[-1] * 2 if self.is_bidirectional else self.hidden_size[-1]
        self.dict_size = len(modules["w2i"])
        self.lvt_dict_size = consts["lvt_dict_size"]
        self.optimizer = modules["optimizer"]

        self.params = []
        self.sub_params = [] if self.has_lvt_trick else None

        self.define_layers(modules, consts, options)
        if not self.is_predicting:
            self.define_train_funcs(modules, consts, options)

    def define_layers(self, modules, consts, options):
        if self.has_learnable_w2v:
            self.w_rawdata_emb = init_weights((self.dict_size, self.dim_x), "w_rawdata_emb", sample = "normal")
            x_flat = self.x.flatten()
            x_emb = self.w_rawdata_emb[x_flat, :]
            x_emb = T.reshape(x_emb, (self.len_x, self.batch_size, self.dim_x))
            if self.is_bidirectional:
                xb_flat = self.x[::-1].flatten()
                xb_mask = self.x_mask[::-1]
                xb_emb = self.w_rawdata_emb[xb_flat, :]
                xb_emb = T.reshape(xb_emb, (self.len_x, self.batch_size, self.dim_x))
            self.params.append(self.w_rawdata_emb)

        # LM layer
        layer_input = x_emb
        mask = self.x_mask
        shape = (self.dim_x, self.batch_size)
        word_encoder_layer = WordEncoderLayer(layer_input, mask, shape, self.hidden_size, "forward")
        self.params += word_encoder_layer.params
        if self.is_bidirectional:
            layer_input = xb_emb
            mask = xb_mask
            shape = (self.dim_x, self.batch_size)
            word_encoder_layer_b = WordEncoderLayer(layer_input, mask, shape, self.hidden_size, "backward")
            self.params += word_encoder_layer_b.params
        
            word_emb_f = word_encoder_layer.word_emb
            word_emb_b = word_encoder_layer_b.word_emb
            self.word_emb = self.concatenate((word_emb_f, word_emb_b[::-1]), word_emb_f.ndim - 1)
        else:
            self.word_emb = word_encoder_layer.word_emb

        self.dec_init_state = T.sum(self.word_emb * self.x_mask, axis = 0) / T.sum(self.x_mask, axis = 0)
        self.W_init_state = init_weights((self.ctx_size, self.hidden_size[-1]), "W_init_state")
        self.b_init_state = init_bias(self.hidden_size[-1], "b_init_state")
        self.dec_init_state = T.tanh(T.dot(self.dec_init_state, self.W_init_state) + self.b_init_state)
        self.params += [self.W_init_state, self.b_init_state]

        if self.is_predicting:
            self.encode = theano.function(inputs = [self.x, self.x_mask, self.batch_size],
                outputs = [self.word_emb, self.dec_init_state],
                on_unused_input = 'ignore')

        if self.has_learnable_w2v:
            y_flat = self.y.flatten()
            if self.is_predicting:
                y_emb = ifelse(T.lt(T.sum(y_flat), 0), 
                        T.zeros((self.batch_size, self.dim_y)), self.w_rawdata_emb[y_flat, :]) # call sum() for computing a scalar condition
                y_emb = T.reshape(y_emb, (self.batch_size, self.dim_y))
            else:
                y_emb = self.w_rawdata_emb[y_flat, :]
                y_emb = T.reshape(y_emb, (self.len_y, self.batch_size, self.dim_y))
                y_shifted = T.zeros_like(y_emb)
                y_shifted = T.set_subtensor(y_shifted[1:, :, :], y_emb[:-1, :, :])
                y_emb = y_shifted
        
        if self.is_predicting:
            self.dec_next_state = T.matrix("dec_next_state")
            self.dec_next_state_z = T.matrix("dec_next_state_z")
            layer_input = (y_emb, self.word_emb, self.dec_next_state, self.x, self.dec_next_state_z)
        else:
            layer_input = (y_emb, self.word_emb, self.dec_init_state, self.x, None)
        mask = (self.x_mask, None) if self.is_predicting else (self.x_mask, self.y_mask)
        shape = (self.dim_y, self.hidden_size[-1], self.ctx_size, self.batch_size, self.updated_batch_size, self.latent_size)
        word_decoder_layer = WordDecoderLayer(layer_input, mask, shape, self.is_predicting, self.beam_decoding)
        self.params += word_decoder_layer.params

        self.dec_status = word_decoder_layer.hidden_status
        self.atted_context = word_decoder_layer.atted_context
        self.z_params = word_decoder_layer.z_params

        self.cp_idx = word_decoder_layer.cp_idx
        self.dec_z = word_decoder_layer.dec_z
        self.dec_mu = word_decoder_layer.dec_mu
        self.dec_var = word_decoder_layer.dec_var
        
        if self.has_lvt_trick:
            self.lvt_dict = T.lvector("lvt_dict")
            if not self.is_predicting:
                self.y_lvt = T.tensor3("y_lvt", dtype = "int64")

        if self.has_lvt_trick:
            layer_input = (self.dec_status, self.atted_context, y_emb, self.cp_idx, self.dec_z, self.lvt_dict)
            shape = (self.hidden_size[-1], self.ctx_size, self.dim_y, self.dict_size, self.latent_size, self.lvt_dict_size)
        else:
            layer_input = (self.dec_status, self.atted_context, y_emb, self.cp_idx, self.dec_z)
            shape = (self.hidden_size[-1], self.ctx_size, self.dim_y, self.dict_size, self.latent_size)
        word_prob_layer = WordProbLayer(layer_input, shape, self.is_predicting, self.has_lvt_trick)
        self.params += word_prob_layer.params
        if self.has_lvt_trick:
            self.sub_params += word_prob_layer.sub_params

        self.y_pred = word_prob_layer.y_pred

        if self.is_predicting:
            inputs = [self.y, self.word_emb, self.dec_next_state, self.dec_next_state_z, self.x, self.x_mask, self.batch_size]
            if self.has_lvt_trick:
                inputs += [self.lvt_dict]
            self.decode_once = theano.function(inputs = inputs, outputs = [self.y_pred, self.dec_status, self.dec_z], on_unused_input = 'ignore')

    def categorical_crossentropy(self, modules):
        if self.has_lvt_trick:
            y_flat = self.y_lvt.flatten()
            y_flat_idx = T.arange(y_flat.shape[0]) * self.lvt_dict_size + y_flat
        else:
            y_flat = self.y.flatten()
            y_flat_idx = T.arange(y_flat.shape[0]) * self.dict_size + y_flat
        cost = -T.log(self.y_pred.flatten()[y_flat_idx] + 1e-15)
        cost = cost.reshape(self.y.shape)
        cost = T.sum(cost * self.y_mask, axis = 0)
        cost = cost.reshape((self.batch_size, ))
        return cost 
    
    def kld(self, mu, var):
        return 0.5 * T.sum(1 + T.log(var + 1e-15) - mu**2 - var, axis=1)

    def define_train_funcs(self, modules, consts, options):
        a = -T.mean(self.kld(self.dec_mu, self.dec_var))
        b = T.mean(self.categorical_crossentropy(modules))
        cost = a + b

        gparams = []
        sub_gparams = [] if self.has_lvt_trick else None
        for param in self.params:
            gparams.append(T.clip(T.grad(cost, param), -10, 10))
        if self.has_lvt_trick:
            for param in self.sub_params:
                sub_gparams.append(T.clip(T.grad(cost, param[1]), -10, 10))
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, self.sub_params, sub_gparams, self.lr, z_params=self.z_params)

        inputs = [self.x, self.y, self.x_mask, self.y_mask, self.batch_size, self.lr]
        if self.has_lvt_trick:
            inputs += [self.lvt_dict, self.y_lvt]

        self.train = theano.function(inputs = inputs,
                                     outputs = [cost, a, b, T.mean(self.dec_mu), T.mean(self.dec_var), self.y_pred],
            updates = updates,
            on_unused_input = 'ignore')
        '''
        self.validate = theano.function(inputs = [self.x, self.y, self.x_mask, self.y_mask, 
            self.batch_size],
            outputs = [cost, self.y_pred],
            on_unused_input = 'ignore')
        '''
        
    def concatenate(self, tensor_list, axis = 0):
        """
        Alternative implementation of `theano.tensor.concatenate`.
        This function does exactly the same thing, but contrary to Theano's own
        implementation, the gradient is implemented on the GPU.
        Backpropagating through `theano.tensor.concatenate` yields slowdowns
        because the inverse operation (splitting) needs to be done on the CPU.
        This implementation does not have that problem.
        :usage:
            >>> x, y = T.matrices('x', 'y')
            >>> c = concatenate([x, y], axis=1)
        :parameters:
            - tensor_list : list
                list of Theano tensor expressions that should be concatenated.
            - axis : int
                the tensors will be joined along this axis.
        :returns:
            - out : tensor
                the concatenated tensor expression.
        """
        concat_size = sum(tt.shape[axis] for tt in tensor_list)
    
        output_shape = ()
        for k in range(axis):
            output_shape += (tensor_list[0].shape[k],)
        output_shape += (concat_size,)
        for k in range(axis + 1, tensor_list[0].ndim):
            output_shape += (tensor_list[0].shape[k],)
    
        out = T.zeros(output_shape)
        offset = 0
        for tt in tensor_list:
            indices = ()
            for k in range(axis):
                indices += (slice(None),)
            indices += (slice(offset, offset + tt.shape[axis]),)
            for k in range(axis + 1, tensor_list[0].ndim):
                indices += (slice(None),)
    
            out = T.set_subtensor(out[indices], tt)
            offset += tt.shape[axis]
    
        return out
