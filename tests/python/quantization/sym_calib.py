import logging

import mxnet as mx
from mxnet.gluon import nn
import nnvm as nnvm
import tvm

from sym_utils import *
from quant_utils import *
from utils import *


class _ArgsWrapper():
    def __init__(self, **kwargs):
        self.args = {k:v for k,v in kwargs}

def get_node(sym, graph):
    name = sym.attr('name')
    if name not in graph:
        assert False, "Unrecognized layer:%s in sym_post_quant"%name
    return graph[name]

def _is_changeable_ops(sym):
    name = sym.attr('name')
    attr = sym.list_attr()

    no_changeable_ops = ['null', 'Flatten', 'Activation']
    if name in no_changeable_ops:
        return False

    if name == 'Pooling' and attr['pool_type'] == 'max':
        return False

    return True

def _calib_ops_shift_bit(symbol, params, ctx, calib_data, quant_flag):
    logger = logging.getLogger("log.calib.out.shift_bits")
    logger.setLevel(quant_flag.log_level)

    inputs = mx.sym.var('data')
    image_data = calib_data.data[0]
    _, input_shift_bits = quant_helper(image_data)

    OUT_SB_SUFFIX = '_out_shift_bits'
    IN_SB_SUFFIX = '_in_shift_bits'

    collect_res = {}
    for sym in topo_sort(symbol):
        name = sym.attr('name')
        op_name = sym.attr('op_name')
        childs = sym.get_children()

        assert op_name in mx_identity_ext

        if not _is_no_changeable_ops(sym):
            continue

        mx_graph = nn.SymbolBlock(sym, [inputs])
        load_parameters(mx_graph, params, ctx=ctx)
        qres = mx_graph.forward(image_data.as_in_context(ctx))
        quant_res, out_sb = quant_helper(qres)

        out_sb_name = name + OUT_SB_SUFFIX
        collect_res[out_sb_name] = {
            "header": quant_res[0].asnumpy().flatten()[0],
            "max": quant_res.max().asnumpy(),
            "min": quant_res.min().asnumpy(),
            "shape": quant_res.shape,
            "shift_bits": out_sb,
        }

    # calibrate input&output shift bits
    # TODO
    for sym in topo_sort(symbol):
        name = sym.attr('name')
        op_name = sym.attr('op_name')
        childs = sym.get_children()

        in_sb_name = name + IN_SB_SUFFIX
        out_sb_name = name + OUT_SB_SUFFIX

        if childs is None:
            continue

        inputs = [c for c in childs \
            if c.attr('op_name') != 'null' or c.attr('name') == 'data']

        assert len(inputs) > 0
        if len(inputs) == 1:
            collect_res

    return symbol, params, collect_res

def calibrate_params(symbol, params, quant_flag, graph={}):
    logger = logging.getLogger("log.quant.calib")

    pass

