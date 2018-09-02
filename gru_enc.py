#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class GRUEncLayer(object):
    def __init__(self, layer_id, x, x_mask, shape):
        prefix = "GRUEnc_"
        layer_id = "_" + layer_id
        self.x_mask = x_mask
        self.in_size, self.out_size, self.batch_size = shape

        self.W = init_weights((self.in_size, self.out_size), prefix + "W" + layer_id, num_concatenate = 2, axis_concatenate = 1)
        self.U = init_weights((self.out_size, self.out_size), prefix + "U" + layer_id, "ortho", num_concatenate = 2, axis_concatenate = 1)
        self.b = init_bias(self.out_size, prefix + "b" + layer_id, num_concatenate = 2)
        
        self.Wx = init_weights((self.in_size, self.out_size), prefix + "Wx" + layer_id)
        self.Ux = init_weights((self.out_size, self.out_size), prefix + "Ux" + layer_id, "ortho")
        self.bx = init_bias(self.out_size, prefix + "bx" + layer_id)

        self.params = [self.W, self.U, self.b, self.Wx, self.Ux, self.bx]

        self.x = T.dot(x, self.W) + self.b
        self.xx = T.dot(x, self.Wx) + self.bx

        def _slice(x, n):
            if x.ndim == 3:
                return x[:, :, n * self.out_size : (n + 1) * self.out_size]
            elif x.ndim == 2:
                return x[:, n * self.out_size : (n + 1) * self.out_size]

        def _active(x, xx, m, pre_h, U, Ux):
            tmp = T.nnet.sigmoid(T.dot(pre_h, U) + x)
            r = _slice(tmp, 0)
            u = _slice(tmp, 1)
            h = T.tanh(T.dot(pre_h, Ux) * r + xx)
            h = u * pre_h + (1.0 - u) * h
            h = m * h + (1.0 - m) * pre_h
            return h

        outputs, _ = theano.scan(_active, 
                sequences = [self.x, self.xx, self.x_mask],
                outputs_info = [T.zeros((self.batch_size, self.out_size), dtype = theano.config.floatX)],
                non_sequences = [self.U, self.Ux],
                allow_gc = False, strict = True)

        self.activation = outputs
        
