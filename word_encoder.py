#pylint: skip-file
from gru_enc import *

class WordEncoderLayer(object):
    def __init__(self, layer_input, x_mask, shape, hidden_size, name):
        self.prefix = "WordEncoder_" + name + "_"
        self.x = layer_input
        self.x_mask = x_mask
        self.x_size, self.num_docs = shape
        self.hidden_size = hidden_size
        self.layers = []
        self.params = []

        for i in xrange(len(self.hidden_size)):
            if i == 0:
                layer_input = self.x
                shape = (self.x_size, self.hidden_size[0], self.num_docs) 
            else:
                layer_input = self.layers[i - 1].activation
                shape = (self.layers[i - 1].out_size, self.hidden_size[i], self.num_docs) 

            hidden_layer = GRUEncLayer(self.prefix + str(i), layer_input, self.x_mask, shape)
            self.layers.append(hidden_layer)
            self.params += hidden_layer.params

        self.word_emb = hidden_layer.activation
        self.sent_emb = self.word_emb[-1]
