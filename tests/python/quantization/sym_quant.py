import mxnet as mx
from mxnet import ndarray as nd

from sym_utils import *

max_bit = 32 # INT32
default_target_bit = 8 # INT8
input_bit = 8
bias_target_bit = default_target_bit * 4 - 1
disable_requant_ops = [
    'Activation', 'relu',
    'Pooling',
    'slice', 'slice_like', 'slice_axis',
    'clip', 'negative',
    'repeat', 'tile',
    'Reshape', 'transpose', 'Flatten',
]

def _infer_fixed_precs(sym, params, graph, inputs_ext, precs):
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    if op_name == 'null':
        precs[name] = input_bit if name in inputs_ext else default_target_bit
    elif op_name in disable_requant_ops:
        A_name = childs[0].attr('name')
        if A_name in precs:
            precs[name] = precs[A_name]
    elif op_name in ['sigmoid', 'exp']:
        A_name = childs[0].attr('name')
        precs[A_name] = precs[A_name] if A_name in precs else 16
        assert precs[A_name] <= 16, "%s out of INT16" % (A_name)
        precs[name] = 16
    elif op_name in ['Convolution', 'FullyConnected']:
        X, W = childs[0], childs[1]
        X_name, W_name = X.attr('name'), W.attr('name')
        precs[X_name] = precs[X_name] if X_name in precs else 8
        assert precs[X_name] <= 8, "%s out of INT8" % (X_name)
        precs[W_name] = 8
        if eval(attr['no_bias']) == False:
            precs[childs[2].attr('name')] = 31
    elif op_name in ['broadcast_add', 'broadcast_sub', 'elemwise_add',
            'elemwise_sub']:
        A, B = childs[0], childs[1]
        A_name, B_name = A.attr('name'), B.attr('name')
        precs[A_name] = precs[A_name] if A_name in precs else 31
        assert precs[A_name] <= 31, "%s out of INT31" % (A_name)
        precs[B_name] = precs[B_name] if B_name in precs else 31
        assert precs[B_name] <= 31, "%s out of INT31" % (B_name)
    elif op_name in ['broadcast_mul']:
        A, B = childs[0], childs[1]
        A_name, B_name = A.attr('name'), B.attr('name')
        precs[A_name] = precs[A_name] if A_name in precs else 16
        assert A_name <= 32, "%s out of INT32" % (A_name)
        precs[B_name] = precs[B_name] if B_name in precs else 16
        assert B_name <= 32, "%s out of INT32" % (B_name)
        assert precs[A_name] + precs[B_name] <= 32, \
                "%s forward out of INT32" % (name)
    if childs is not None:
        for c in childs:
            _infer_backward_precs(c, precs)
    return sym, params

def _infer_backward_precs(sym, precs):
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    if op_name in disable_requant_ops:
        A_name = childs[0].attr('name')
        if A_name in precs:
            assert precs[A_name] == precs[name]
        else:
            precs[A_name] = precs[name]
        _infer_backward_precs(childs[0], precs)

def sym_infer_precision(symbol, params, inputs_ext):
    logger = logging.getLogger('log.infer.precision')
    precs = {}
    topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_infer_fixed_precs, precs=precs)

    return symbol, params


def sym_simulate(symbol, params, inputs_ext, data):
    return symbol, params
