import logging
import json

import quant_pass as qpass

import mxnet as mx
from mxnet.symbol import _internal
from mxnet import symbol as _sym

INT32_MIN, INT32_MAX = -2147483647, 2147483647
INT8_MIN, INT8_MAX = -127, 127

def _topo_sort(symbol):
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
    while dep_cnts:
        remove = []
        for name in dep_cnts:
            if dep_cnts[name] == 0:
                order.append(symbol_map[name])
                remove.append(name)
                if name in deps:
                    for other in deps[name]:
                        dep_cnts[other] -= 1
        for name in remove:
            del dep_cnts[name]
    return order

def _get_mxnet_op(op_name):
    try:
        op = getattr(_internal, op_name)
    except:
        op = getattr(_sym, op_name)

    if not op:
        raise RuntimeError("Unable to map op_name {} to nnvm.sym".format(op_name))
    return op

def fold_cond(symbol, params, graph, quant_flag):
    def get_node(name):
        if isinstance(sym, mx.sym.Symbol):
            name = name.attr('name')

        if name not in graph:
            assert False, "Op:%s haven't been processed"%name

        #  output_index = json.loads(sym.tojson())['heads'][0][1]
        return graph[name]

    def left_shift(inputs, scale_sym, shift_bits):
        scale = 2 ** (-shift_bits)
        #  out = mx.sym.broadcast_mul(inputs, scale_sym)
        out = inputs / int(scale.asnumpy()[0])
        out = mx.sym.floor(out)
        return out, scale

    def right_shift(inputs, scale_sym, shift_bits):
        scale = 2 ** (shift_bits-1)

        #  out = mx.sym.broadcast_div(inputs, scale_sym)
        out = inputs / int(scale.asnumpy()[0])
        out = mx.sym.floor(out + 1)
        out = mx.sym.floor(out / 2)

        return out, scale

    for sym in _topo_sort(symbol):
        name = sym.attr('name')
        attr = sym.list_attr()
        op_name = sym.attr('op_name')
        childs = sym.get_children()

        # update inputs layer symbol
        if childs is not None:
            childs = [get_node(child) for child in childs]
            # update childs inputs
            op = _get_mxnet_op(op_name)
            node = op(*childs, **attr)
        elif op_name != 'null':
            assert False, "Unrecognized op without input"
        else:
            # inputs or params
            node = sym

        if op_name == '_cond':
            # cond_func, then_func, else_func = sym.attr('subgraph')
            sb_param_idx, lesser_scalar_idx, others = None, None, []
            for idx, child in enumerate(childs):
                child_op_name = child.attr('op_name')
                if child_op_name == 'null':
                    assert sb_param_idx is None
                    sb_param_idx = idx
                elif child_op_name == '_lesser_scalar':
                    lesser_scalar_idx = idx
                else:
                    others.append(idx)

            shift_bits_sym = childs[sb_param_idx]
            sb_param_name = shift_bits_sym.attr('name')
            assert sb_param_name in params

            assert len(others) == 2
            # _cond op must be created by same input
            assert childs[others[0]].attr('name') == childs[others[1]].attr('name')
            input_sym = childs[others[0]]

            shift_bits = params[sb_param_name]
            scale_name = sb_param_name.replace('_shift_bits', '_scale')
            scale_sym = mx.sym.var(scale_name, shape=(1,))
            if any(shift_bits < 1):
                node, scale = left_shift(input_sym, scale_sym, shift_bits)
            else:
                node, scale = right_shift(input_sym, scale_sym, shift_bits)

            params[scale_name] = scale
            del params[sb_param_name]

        graph[name] = node

    nodes = []
    for sym in symbol:
        node = get_node(sym)
        nodes.append(node)

    ret_sym = nodes[0]
    if len(nodes) > 1:
        ret_sym = mx.sym.Group(nodes)

    return ret_sym, params


def fuse_conv_bn(symbol, params, graph, quant_flag):
    logger = logging.getLogger("log.fuse.conv_bn")

    def get_node(sym):
        name = sym.attr('name')
        if name not in graph:
            assert False, "Unrecognized error in fuse_conv_bn"
        output_index = json.loads(sym.tojson())['heads'][0][1]
        return graph[name][output_index]

    logger.info("Fuse conv & bn")
    qparams = qpass.fuse_bn_parameters(params, quant_flag)
    graph = {}
    fuse_maps = {}
    remove_nodes = []
    for sym in _topo_sort(symbol):
        name = sym.attr('name')
        attr = sym.list_attr()
        op_name = sym.attr('op_name')
        childs = sym.get_children()

        if op_name == "BatchNorm":
            assert childs is not None
            childs = [get_node(child) for child in childs]
            childs_op_name = [child.attr('op_name') for child in childs]

            remove_nodes.append(name)

            for child in childs:
                if child.attr('op_name') == 'Convolution':
                    fuse_maps[name] = child.attr('name')
                    continue

                assert child.attr('op_name') == 'null'
                remove_nodes.append(child.attr('name'))

            assert "Convolution" in childs_op_name

            print (name, op_name, attr, childs_op_name)

        graph[name] = sym

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
    In[Int8] * P_weight[Int8] + P_bias[Int32] -> Out[Int32]
FullyConnected|Dense:
    In[Int8] * P_weight[Int8] + P_bias[Int32] -> Out[Int32]
elemwise_add: forward with Int8 input, which means the previous layer
must be ClipInt.
    In1[Int8] + In2[Int8] -> Out[Int32]
sum: reduce op over specific axis, sum(data, axis=[1, 2])
    In[Int8] -> Out[Int32]

Reshape:
    In[Int32] -> Out[Int32]
Flatten:
    In[Int32] -> Out[Int32]

# op for requant
{broadcast_div}:
    In[Int32] / P_scale[Int32] -> Out[Int32]
broadcast_mul|_mul_scalar:
    In[Int32] * P_scale[Int32] -> Out[Int32]
    In[Int32] * C_scale[Int32] -> Out[Int32]
_div_scalar:
    In[Int32] / C_scale[Int32] -> Out[Int32]
_plus_scalar:
    In[Int32] + C_scale[Int32] -> Out[Int32]

ClipInt: the only operator to forward Int32 input to Int8 output.
    In[Int32] -> Out[Int8]

"""
IdentitySymbols = {
    'null': [],
    'Activation': [],
    'Pooling': [],
    'Convolution': [],
    'FullyConnected': [],

    'Reshape': [],

    'sum': [],
    'broadcast_div': [],
    'broadcast_mul': [],

    'clip': [],

    'elemwise_add': [],

    '_div_scalar': [],
    '_plus_scalar': [],
}

def sym_post_quant(symbol, params, graph, quant_flag):
    logger = logging.getLogger("log.quant.post")

    def get_node(sym):
        name = sym.attr('name')
        if name not in graph:
            assert False, "Unrecognized layer:%s in sym_post_quant"%name
        return graph[name]

    # symbol
    for sym in _topo_sort(symbol):
        name = sym.attr('name')
        attr = sym.list_attr()
        op_name = sym.attr('op_name')
        childs = sym.get_children()

        # update inputs layer symbol
        if childs is not None:
            childs = [get_node(child) for child in childs]
            # update childs inputs
            op = _get_mxnet_op(op_name)
            node = op(*childs, **attr)
        elif op_name != 'null':
            assert False, "Unrecognized op without input"
        else:
            # inputs or params
            node = sym

        # remove layer: floor in int8
        if op_name == "floor":
            childs_name = [c.attr('name') for c in childs]
            assert len(childs) == 1
            node = childs[0]

        graph[name] = node

    nodes = []
    for sym in symbol:
        node = graph[sym.attr('name')]
        nodes.append(node)
    ret_sym = nodes[0]
    if len(nodes) > 1:
        ret_sym = mx.sym.Group(nodes)

    ops = set()
    for sym in _topo_sort(ret_sym):
        op_name = sym.attr('op_name')
        childs = sym.get_children()

        v = []
        if childs is not None:
            v = [(c.attr('name'), c.attr('op_name')) for c in childs]

        ops.add(op_name)

    # params
    args = ret_sym.list_arguments()
    ret_params, params_dtype = {}, {}
    for key, value in params.items():
        if key in args:
            msg = "key:%s value:%s"%(key, value)
            flat = value.asnumpy().flatten()

            dtype = 'int8' if all(flat >= INT8_MIN) and all(flat <= INT8_MAX) \
                    else 'int32'

            assert all(flat.astype('int32').astype('float32') == flat), msg
            params_dtype[key] = dtype
            ret_params[key] = value.astype(dtype)

    for arg in args:
        if arg == 'data':
            continue
        assert arg in ret_params, 'arg:%s in symbol not exists params'%arg

    logger.info("Existing operators: %s", sorted(ops))

    return ret_sym, ret_params

