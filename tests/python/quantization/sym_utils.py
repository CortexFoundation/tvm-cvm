from mxnet.symbol import _internal
from mxnet import symbol as _sym
from mxnet import ndarray as nd
import mxnet as mx
import nnvm
import logging

INT32_MIN, INT32_MAX = -2147483647, 2147483647
INT8_MIN, INT8_MAX = -127, 127

INT8_TYPE, INT32_TYPE= ('int8', 'int32')

class OpExt():
    def __init__(self, op_name='null',
            in_types=[], out_types=[]):
        self.op_name = op_name
        self.in_types = in_types
        self.out_types = out_types

def combile_name(n1, n2):
    t1, t2 = n1.split("_"), n2.split("_")
    len1, len2 = len(t1), len(t2)
    min_len = min(len1, len2)
    begin, end = 0, 0
    for i in range(min_len):
        if t1[i] != t2[i]:
            break
        begin = i + 1
    for i in range(min_len):
        if t1[len1-1-i] != t2[len2-1-i]:
            break
        end = i + 1
    res = t1[:begin]
    res.extend([n for n in t1[begin:len1-end]])
    res.extend([n for n in t2[begin:len2-end]])
    res.extend(t1[len1-end:])
    return "_".join(res)

def get_nd_op(op_name):
    op = getattr(nd, op_name, None)
    if op is None:
        op = getattr(nd._internal, op_name, None)

    if op is None:
        raise RuntimeError("Unable to map op_name {} to mxnet.ndarray".format(op_name))
    return op

_MX_OP_CONTRIB_PREFIX = '_contrib_'
def get_mxnet_op(op_name):
    op = getattr(_internal, op_name, None)
    if op is None:
        op = getattr(_sym, op_name, None)
    if op_name.startswith(_MX_OP_CONTRIB_PREFIX):
        op = getattr(_sym.contrib, op_name[len(_MX_OP_CONTRIB_PREFIX):], None)

    if op is None:
        raise RuntimeError("Unable to map op_name {} to mxnet.sym".format(op_name))
    return op

def get_nnvm_op(op_name):
    op = getattr(nnvm.sym, op_name)

    if not op:
        raise RuntimeError("Unable to map op_name {} to nnvm.sym".format(op_name))
    return op

def sym_iter(sym):
    if sym is None:
        return None

    if isinstance(sym, mx.sym.Symbol):
        sym = [sym[i] for i in range(len(sym))]

    sym = [s for s in sym]
    return sym

def examine_parameters(symbol, params, inputs_ext, allows=[], callback=None):
    args, new_params = symbol.list_inputs(), {}
    for arg in args:
        if arg not in inputs_ext:
            assert arg in params, 'arg(%s) not exists in params dict(%s)' \
                % (arg, params.keys())

            if callback is not None:
                callback(params, arg)

            new_params[arg] = params[arg]

    for name in allows:
        if name in params:
            new_params[name] = params[name]
    return new_params

def op_const(number, graph, var=mx.sym.var):
    name = 'const_var_' + str(number)
    if name not in graph:
        graph[name] = var(name, shape=(1,))
    return graph[name], name

def topo_sort(symbol, logger=logging):
    """Sort all symbols in the mxnet graph in topological order.

    Parameters
    ----------
    symbol : mxnet.sym.Symbol

    Returns:
    -------
    list
        List of mxnet symbol
    """
    queue = []
    symbol_map = {}
    deps = {}
    dep_cnts = {}
    for s in symbol:
        symbol_map[s.attr('name')] = s
        queue.append(s)

    while queue:
        sym = queue.pop(0)
        name = sym.attr('name')
        childs = sym.get_children()
        if childs is None:
            dep_cnts[name] = 0
        else:
            childs = sym_iter(childs)
            # remove duplication dependency
            dep_cnts[name] = len({c.attr('name') for c in childs})
            for child in childs:
                child_name = child.attr('name')
                if child_name not in deps:
                    deps[child_name] = set()
                deps[child_name].add(name)
                if child_name not in symbol_map:
                    symbol_map[child_name] = child
                    queue.append(child)

    order = []
    reduce_flag = True
    while dep_cnts:
        if not reduce_flag:
            logger.critical("deps cannot reduce -> %s", dep_cnts)
            assert False

        remove = []
        reduce_flag = False
        for name in dep_cnts:
            if dep_cnts[name] == 0:
                order.append(symbol_map[name])
                remove.append(name)
                if name in deps:
                    for other in deps[name]:
                        dep_cnts[other] -= 1

                reduce_flag = True
        for name in remove:
            del dep_cnts[name]
    return order

def sym_collect_attr(symbol, attr_name='op_name'):
    return {sym.attr(attr_name) for sym in topo_sort(symbol)}

def get_node(sym, graph):
    name = sym.attr('name')
    if name not in graph:
        assert False, "Unrecognized layer:%s in graph"%name
    return graph[name]

def topo_visit(symbol, params, inputs_ext={}, get_op=get_mxnet_op,
        logger=logging, callback=None, **kwargs):
    graph = {}
    params = {k:v[:] for k,v in params.items()}
    for sym in topo_sort(symbol, logger):
        name = sym.attr('name')
        op_name = sym.attr('op_name')
        childs = sym_iter(sym.get_children())
        attr = sym.list_attr()

        node = sym
        # TODO: add support for _cond op
        if op_name in ['_cond'] and logger:
            logger.warn(
                    "topo_visit do not support op %s:%s(%s), \
attention used in non-changable graph pass",
                    op_name, name, [c.attr('name') for c in childs])
        elif childs is not None:
            # update childs in graph
            childs = [get_node(c, graph) for c in childs]
            node = get_op(op_name)(*childs, **attr, name=name)

            # check params dict
            for c in childs:
                if c.attr('op_name') != 'null':
                    continue
                cname = c.attr('name')
                assert cname in params or cname in inputs_ext, \
                    'symbol:%s(%s) parameter:%s is missing in params dict:%s' \
                    % (name, [c.attr('name') for c in childs],
                        cname, params.keys())

        if callback is not None:
            # process symbol and params
            node, params = callback(node, params, graph, inputs_ext, **kwargs)

        graph[name] = node

    nodes = []
    for sym in symbol:
        node = get_node(sym, graph)
        nodes.append(node)

    ret_sym = nodes[0]
    if len(nodes) > 1:
        ret_sym = mx.sym.Group(nodes)

    return ret_sym, params


"""Deterministic Op Description
The specific op for quantization with Int8 or Int32, more details
described as belows:

In: inputs variable, maybe followed with int counter.
Out: output variable, maybe followed with int counter.
P_X: params variable, load from params file.
C_X: constant variable, fixed in graph.

Activation: specific indicate relu.
    In[Int8] -> Out[Int8]
Pooling: sepcific indicate max pool.
    In[Int8] -> Out[Int8]
Convolution:
    In[Int8] * P_weight[Int8] + P_bias[Int32] -> Out[Int64]
FullyConnected|Dense:
    In[Int8] * P_weight[Int8] + P_bias[Int32] -> Out[Int64]
elemwise_add:
    In1[Int8] + In2[Int8] -> Out[Int32]
sum: reduce op over specific axis, sum(data, axis=[1, 2])
    In[Int8] -> Out[Int32]

Reshape:
    In[Int32] -> Out[Int32]
Flatten:
    In[Int32] -> Out[Int32]

broadcast_add:
    In1[Int32] + In2[Int32] -> Out[Int64]
    In1[Int8]  + In2[Int8]  -> Out[Int32]
broadcast_sub:
    In1[Int32] + In2[Int32] -> Out[Int64]
    In1[Int8]  - In2[Int8]  -> Out[Int32]
broadcast_mul:
    In1[Int32] * In2[Int32] -> Out[Int64]
    In1[Int8]  * In2[Int8]  -> Out[Int32]
broadcast_div:
    In1[Int32] / In2[Int32] -> Out[Int32]
    In1[Int8]  / In2[Int8]  -> Out[Int8]

_plus_scalar:
    In[Int32] + C_scale[Int32] -> Out[Int64]
_sub_scalar:
    In[Int32] - C_scale[Int32] -> Out[Int64]
_mul_scalar:
    In[Int32] * C_scale[Int32] -> Out[Int64]
_div_scalar:
    In[Int32] / C_scale[Int32] -> Out[Int32]

# Requant Op
cvm_right_shift:
    assert P_shift_bits > 0
    In[Int8|Int32|Int64] >> P_shift_bits[Int8] -> Out[Int8]
cvm_left_shift:
    assert 0 <= P_shift_bits < 24
    In[Int8|Int32|Int64] << P_shift_bits[Int8] -> Out[Int8]

"""
nnvm_identity_ext = {
    'null': OpExt(out_types=[INT8_TYPE, INT32_TYPE]),

    'relu': OpExt('relu', [INT8_TYPE], [INT8_TYPE]),
    'max_pool2d': OpExt('max_pool2d', [INT8_TYPE], [INT8_TYPE]),

    'conv2d': OpExt('conv2d', [INT8_TYPE], [INT32_TYPE]),
    'dense': OpExt('dense', [INT8_TYPE], [INT32_TYPE]),
    'sum': OpExt('sum', [INT8_TYPE], [INT32_TYPE]),
    'elemwise_add': OpExt('elemwise_add', [INT8_TYPE], [INT32_TYPE]),

    'reshape': OpExt('reshape', [INT8_TYPE, INT32_TYPE], [INT8_TYPE, INT32_TYPE]),
    'flatten': OpExt('flatten', [INT8_TYPE, INT32_TYPE], [INT8_TYPE, INT32_TYPE]),
    'strided_slice': OpExt('strided_slice', [INT8_TYPE], [INT8_TYPE]),

    'broadcast_right_shift': OpExt('broadcast_right_shift', [INT32_TYPE], [INT8_TYPE]),
    'broadcast_left_shift': OpExt('broadcast_left_shift', [INT32_TYPE], [INT8_TYPE]),
    'broadcast_div': OpExt('broadcast_div', [INT32_TYPE], [INT32_TYPE]),
    'broadcast_mul': OpExt('broadcast_mul', [INT32_TYPE], [INT32_TYPE]),
    'broadcast_add': OpExt('broadcast_add', [INT32_TYPE], [INT32_TYPE]),
    'broadcast_sub': OpExt('broadcast_sub', [INT32_TYPE], [INT32_TYPE]),
    'broadcast_max': OpExt('broadcast_max', [INT32_TYPE], [INT32_TYPE]),

    '__add_scalar__': {},

    'max': {},
    'abs': {},
    'log2': {},

    'clip': OpExt('clip', [INT32_TYPE], [INT8_TYPE]),
    'concatenate': {},

    'cvm_clip': {},
    'cvm_left_shift': {},
    'cvm_right_shift': {},
}

"""Mxnet Symbol Operator Extension
Attribute Options:
    0   : whether flag by default is support
    1...: optional type
"""
mx_identity_ext = {
    'null': {},
    'Convolution': {},
    'BatchNorm': {},
    'Pooling': {
        'pool_type': [False, 'max', 'avg'],
        'count_include_pad': [True, 'True'],
        # 'pooling_convention': [True, 'valid'],
    },
    'Flatten': {},
    'FullyConnected': {},
    'Activation': {
        'act_type': [False, 'relu'], # Only supported relu
    },
    'Dropout': {
        'mode': [True, 'training'],
    },
    'Concat': {},
    'elemwise_add': {},
    'elemwise_sub': {},
    'LeakyReLU': {
        'act_type': [True, 'leaky']
    },
    'slice_like': {},
    'slice_axis': {},
    'repeat': {},
    'Reshape': {},
    'transpose': {},
    'tile': {},
    'expand_dims': {},

    '_arange': {},

    # Not supported broadcast_div
    'broadcast_mul': {},
    'broadcast_add': {},
    'broadcast_sub': {},

    '_mul_scalar': {},
    '_div_scalar': {},
}
