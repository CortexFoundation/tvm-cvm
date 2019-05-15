import gluon_zoo as gz
import mxnet as mx
from mxnet import ndarray as nd

import sym_pass as spass
import utils

gz.save_model("alexnet")


def load_fname(version, suffix=None):
    suffix = "."+suffix if suffix is not None else ""
    return "./data/alexnet%s%s.json"%(version, suffix), \
        "./data/alexnet%s%s.params"%(version, suffix)

inputs_ext = { 'data': {
    'shape': (16, 3, 224, 224)
}}

sym_file, param_file = load_fname("")
sym, params = mx.sym.load(sym_file), nd.load(param_file)
# sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)

utils.log_init()
spass.sym_calculate_ops(sym, params, inputs_ext)
