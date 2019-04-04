import logging

import mxnet as mx
from mxnet.gluon import nn, SymbolBlock
import nnvm as nnvm
import tvm

from sym_utils import *
from quant_utils import *
from utils import *

def _is_changeable_ops(sym):
    name = sym.attr('name')
    attr = sym.list_attr()

    no_changeable_ops = ['null', 'Flatten', 'Activation']
    if name in no_changeable_ops:
        return False

    if name == 'Pooling' and attr['pool_type'] == 'max':
        return False

    return True

def _calib_sym_real_shift_bits(sym, params, graph, inputs_ext,
        out_sbits, ctx=mx.gpu()):
    logger = logging.getLogger('log.calib.out_shift_bits')

    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())

    if op_name == 'null':
        return sym, params

    # calculate layer output shift bits
    args = sym.list_inputs()
    inputs = [mx.sym.var(n) for n in inputs_ext if n in args]
    graph = SymbolBlock(sym, inputs)
    load_parameters(graph, params, ctx=ctx)
    qres = graph.forward(calib_data.as_in_context(ctx))
    _, out_sbits[name] = quant_helper(qres)

    logger.debug("calibrate layer(%s) out shift_bits: %s",
            name, out_sbits[name].asnumpy())

    return sym, params

def _sym_rewrite_and_update_in_sbits(sym, params, graph, inputs_ext,
        in_sbits, out_sbits)
    logger = logging.getLogger('log.calib.out_shift_bits')

    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())

    if childs is None:
        return sym, params

    # calculate layer input shift bits
    childs_name = [c.attr('name') for c in childs]
    inputs_sb = {}
    for c in childs:
        cname = c.attr('name')
        if c.attr('op_name') == 'null' and name in inputs_ext:
            inputs_sb[name] = inputs_ext[name]['shift_bits']
        elif c.attr('op_name') != 'null':
            assert cname in out_sbits, "Unknown error with inputs name(%s) \
                not exists in symbol(%s) out_sbits"%(cname, name)
            inputs_sb[name] = out_sbits[name]

    assert len(inputs_sb) > 0

    node = sym
    if len(inputs_sb) == 1:
        in_sbits[name] = inputs_sb[0]
    elif len(inputs_sb) == 2:
        assert op_name in ['elemwise_add']

        f_name, f_sb = inputs_sb.items()[0]
        f_sb_name = f_name + '_shift_bits'
        assert f_sb_name not in graph
        f_sym = mx.sym.var(f_sb_name)
        graph[f_sb_name] = f_sym

        s_name, s_sb = inputs_sb.items()[1]
        s_sb_name = s_name + '_shift_bits'
        assert s_sb_name not in graph
        s_sym = mx.sym.var(s_sb_name)
        graph[s_sb_name] = s_sym

        assert f_sb.shape == (1,) and s_sb.shape == (1,)
        if any(f_sb > s_sb):
            qparams[f_sb_name], qparams[s_sb_name] = nd.zeros(1,), f_sb - s_sb
        else:
            qparams[f_sb_name], qparams[s_sb_name] = s_sb - f_sb, nd.zeros(1,)
    else:
        logger.critical("Unsupported layer %s with inputs<%s> larger than 2",
                name, childs_name)
        assert False

    return node, params



def _calib_ops_shift_bit(symbol, params, calib_data, inputs_ext, **kwargs):
    logger = logging.getLogger("log.calib.out.shift_bits")

    inputs = mx.sym.var('data')
    image_data = calib_data.data[0]
    _, input_shift_bits = quant_helper(image_data)


    _, inputs_ext['data']['shift_bits'] = quant_helper(calib_data)
    in_sbits, out_sbits = {}, {}
    sym, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_calib_sym_real_shift_bits,
            in_sbits=in_sbits, out_sbits=out_sbits, **kwargs)

    print (sbits)
    exit()

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

