#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class GRUDecLayer(object):
    def __init__(self, layer_input, mask, shape, is_predicting, beam_decoding):
        prefix = "WordDecoderLayer_"
        self.y_emb, self.context, self.init_state, self.xidx, self.state_z = layer_input
        self.x_mask, self.y_mask = mask
        self.dim_y, self.hidden_size, self.ctx_size, self.batch_size, self.updated_batch_size, self.latent_size = shape
        self.is_predicting = is_predicting

        self.W = init_weights((self.dim_y, self.hidden_size), prefix + "W", num_concatenate = 2, axis_concatenate = 1)
        self.U = init_weights((self.hidden_size, self.hidden_size), prefix + "U", "ortho", num_concatenate = 2, axis_concatenate = 1)
        self.b = init_bias(self.hidden_size, prefix + "b", num_concatenate = 2)
        
        self.Wx = init_weights((self.dim_y, self.hidden_size), prefix + "Wx")
        self.Wxz = init_weights((self.dim_y, self.latent_size), prefix + "Wxz")
        self.bxz = init_bias(self.latent_size, prefix + "bxz")

        self.Ux = init_weights((self.hidden_size, self.hidden_size), prefix + "Ux", "ortho")
        self.bx = init_bias(self.hidden_size, prefix + "bx")

        self.Wc_att = init_weights((self.ctx_size, self.ctx_size), prefix + "Wc_att", "ortho")
        self.b_att = init_bias(self.ctx_size, prefix + "b_att")

        self.W_comb_att = init_weights((self.hidden_size, self.ctx_size), prefix + "W_comb_att")
        self.U_att = init_weights((self.ctx_size, 1), prefix + "U_att")

        self.U_nl = init_weights((self.hidden_size, self.hidden_size), prefix + "U_nl", "ortho", num_concatenate = 2, axis_concatenate = 1)
        self.b_nl = init_bias(self.hidden_size, prefix + "b_nl", num_concatenate = 2)
        self.Ux_nl = init_weights((self.hidden_size, self.hidden_size), prefix + "Ux_nl", "ortho")
        self.bx_nl = init_bias(self.hidden_size, prefix + "bx_nl")

        self.Wc = init_weights((self.ctx_size, self.hidden_size), prefix + "Wc", num_concatenate = 2, axis_concatenate = 1)
        self.Wcx = init_weights((self.ctx_size, self.hidden_size), prefix + "Wcx")

        self.W_hz = init_weights((self.hidden_size, self.hidden_size), prefix + "W_hz")
        self.W_zz = init_weights((self.latent_size, self.hidden_size), prefix + "W_zz")

        self.W_hu = init_weights((self.hidden_size, self.latent_size), prefix + "W_hu")
        self.b_hu = init_bias(self.latent_size, prefix + "b_hu")
        self.W_hsigma = init_weights((self.hidden_size, self.latent_size), prefix + "W_hsigma")
        self.b_hsigma = init_bias(self.latent_size, prefix + "b_hsigma")
        
        z_params = [self.W_hu, self.b_hu, self.W_hsigma, self.b_hsigma]
        self.params = [self.W, self.U, self.b,
                self.Wx, self.Ux, self.bx,
                self.U_nl, self.b_nl, self.Ux_nl, self.bx_nl,
                self.Wc, self.Wcx,
                self.Wc_att, self.b_att, 
                self.W_comb_att, self.U_att,
                self.W_hz, self.W_zz,
                self.W_hu, self.b_hu, self.W_hsigma, self.b_hsigma,
                self.Wxz, self.bxz] 

        if is_predicting:
            if beam_decoding:
                self.x_mask = T.tile(self.x_mask, (1, self.batch_size, 1))
            self.y_mask = T.ones((self.batch_size, 1))
    
        self.pctx = T.dot(self.context, self.Wc_att) + self.b_att
        self.x = T.dot(self.y_emb, self.W) + self.b
        self.xx = T.dot(self.y_emb, self.Wx) + self.bx
        self.xxz = T.dot(self.y_emb, self.Wxz) + self.bxz

        def _slice(x, n):
            if x.ndim == 3:
                return x[:, :, n * self.hidden_size : (n + 1) * self.hidden_size]
            return x[:, n * self.hidden_size : (n + 1) * self.hidden_size]

        def _get_word_atten(pctx, h1, W_comb_att, U_att, x_mask):
            unreg_att = T.tanh(pctx + T.dot(h1, W_comb_att)) * x_mask
            unreg_att = T.dot(unreg_att, U_att)

            word_atten = T.exp(unreg_att - T.max(unreg_att, axis = 0, keepdims = True)) * x_mask
            sum_word_atten = T.sum(word_atten, axis = 0, keepdims = True)
            word_atten = T.switch(T.eq(word_atten, 0.0), 0.0, word_atten / sum_word_atten)
            word_atten = T.addbroadcast(word_atten, word_atten.ndim - 1)

            return word_atten


        def _active(x, xx, xxz, y_mask,
                pre_h, pre_z,
                pctx, context, x_mask,
                U, Ux,
                U_nl, Ux_nl, b_nl, bx_nl,
                Wc, Wcx,
                W_comb_att, U_att,
                W_hz, W_zz,
                W_hu, b_hu, W_hsigma, b_hsigma,
                xidx):

            tmp1 = T.nnet.sigmoid(T.dot(pre_h, U) + x)
            r1 = _slice(tmp1, 0)
            u1 = _slice(tmp1, 1)
            h1 = T.tanh(T.dot(pre_h * r1, Ux) +  xx)
            h1 = u1 * pre_h + (1.0 - u1) * h1
            h1 = y_mask * h1 + (1.0 - y_mask) * pre_h
 
            # recurrent-vae encoder
            xh_z = T.nnet.sigmoid(T.dot(pre_z, W_zz) + T.dot(h1, W_hz) + xxz)
            mu = T.dot(xh_z, W_hu) + b_hu
            log_var = T.dot(xh_z, W_hsigma) + b_hsigma
            var = T.exp(log_var)
            sigma = T.sqrt(var)
            eps = 0.0
            if not self.is_predicting:
                eps = floatX(np.random.normal(0, 1, (self.updated_batch_size, self.latent_size)))
                eps = T.reshape(eps, mu.shape)
                eps = T.clip(eps, -5, 5)
            z = mu + sigma * eps

                       
            # len(x) * batch_size * 1
            word_atten = _get_word_atten(pctx, h1, W_comb_att, U_att, x_mask)
            atted_ctx = T.sum(word_atten * context, axis = 0)

            tmp2 = T.nnet.sigmoid(T.dot(atted_ctx, Wc) + T.dot(h1, U_nl) + b_nl)
            r2 = _slice(tmp2, 0)
            u2 = _slice(tmp2, 1)
            h2 = T.tanh(T.dot(atted_ctx, Wcx) + T.dot(h1 * r2, Ux_nl) + bx_nl)
            h2 = u2 * h1 + (1.0 - u2) * h2
            h2 = y_mask * h2 + (1.0 - y_mask) * h1
           
           
            cp_idx = T.argmax(word_atten, axis=0).reshape((self.batch_size, 1))
            cp_idx = xidx[cp_idx[:,0], T.arange(self.batch_size)]

            return h2, z, atted_ctx, cp_idx, mu, var

        sequences = [self.x, self.xx, self.xxz, self.y_mask]
        non_sequences = [self.pctx, self.context, self.x_mask,
                self.U, self.Ux,
                self.U_nl, self.Ux_nl, self.b_nl, self.bx_nl,
                self.Wc, self.Wcx,
                self.W_comb_att, self.U_att,
                self.W_hz, self.W_zz,
                self.W_hu, self.b_hu, self.W_hsigma, self.b_hsigma,
                self.xidx.reshape((self.xidx.shape[0], self.batch_size))]

        if self.is_predicting:
            print "use one-step decoder"
            hs, zs, ac, cp_idx, mu, var = _active(*(sequences + [self.init_state, self.state_z] + non_sequences))
        else:
            init_z = T.zeros((self.batch_size, self.latent_size), dtype = theano.config.floatX)
            [hs, zs, ac, cp_idx, mu, var], _ = theano.scan(_active, 
                    sequences = sequences,
                    outputs_info = [self.init_state, init_z, None, None, None, None], 
                    non_sequences = non_sequences,
                    allow_gc = False, strict = True)

        self.hidden_status = hs
        self.atted_context = ac
        self.word_atten = None
        self.cp_idx = cp_idx
        self.dec_z = zs
        self.dec_mu = mu
        self.dec_var = var
        self.z_params = z_params
