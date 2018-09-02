#pylint: skip-file
from gru_dec import *

class WordDecoderLayer(object):
    def __init__(self, layer_input, mask, shape, is_predicting, beam_decoding):
        self.prefix = "WordDecoderLayer_"
        self.layers = []
        self.params = []

        hidden_layer = GRUDecLayer(layer_input, mask, shape, is_predicting, beam_decoding)
        self.layers.append(hidden_layer)
        self.params += hidden_layer.params

        self.hidden_status = hidden_layer.hidden_status
        self.atted_context = hidden_layer.atted_context
        self.word_atten = hidden_layer.word_atten
        self.cp_idx = hidden_layer.cp_idx
        self.dec_z = hidden_layer.dec_z
        self.dec_mu = hidden_layer.dec_mu
        self.dec_var = hidden_layer.dec_var
        self.z_params = hidden_layer.z_params
