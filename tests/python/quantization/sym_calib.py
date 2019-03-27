import logging

import mxnet as mx
from mxnet.gluon import nn
import nnvm as nnvm
import tvm

from sym_utils import *


class _ArgsWrapper():
    def __init__(self, **kwargs):
        self.args = {k:v for k,v in kwargs}

def get_node(sym, graph):
    name = sym.attr('name')
    if name not in graph:
        assert False, "Unrecognized layer:%s in sym_post_quant"%name
    return graph[name]

def _calib_output_shift_bits(args_wrapper):
    logger = logging.getLogger("log.calib.output.shift_bits")

    added_params_name, deleted_params_name = [], []
    for sym in _topo_sort(symbol):
        mx_graph = nn.SymbolBlock(sym, [inputs])



def calibrate_params(symbol, params, graph, quant_flag):
    logger = logging.getLogger("log.quant.calib")

    pass

