import logging
import json
import math
import numpy as np

import quant_pass as qpass

import mxnet as mx
from mxnet.symbol import _internal
from mxnet import symbol as _sym

import nnvm as nnvm
import tvm

INT32_MIN, INT32_MAX = -2147483647, 2147483647
INT8_MIN, INT8_MAX = -127, 127

class _GraphHelper(object):
    def __init__(self, graph={}, gtype=_sym.Symbol):
        self.graph = graph
        self.gtype = gtype

    def _get_name(self, name):
        if isinstance(name, self.gtype):
            name = name.attr('name')

        assert isinstance(name, str)
        return name

    def get_node(self, sym, default=None):
        name = self._get_name(sym)

        if name not in self.graph:
            if default is None:
                assert False, "op:%s haven't been processed in graph"%name
            else:
                assert isinstance(default, self.gtype)
                self.graph[name] = default

        return self.graph[name]

    def set_node(self, sym, default):
        name = self._get_name(sym)

        assert name not in self.graph

        self.graph[name] = default
        return default



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
        raise RuntimeError("Unable to map op_name {} to mxnet.sym".format(op_name))
    return op

def _get_nnvm_op(op_name):
    op = getattr(nnvm.sym, op_name)

    if not op:
        raise RuntimeError("Unable to map op_name {} to nnvm.sym".format(op_name))
    return op

def fold_cond(symbol, params, graph, quant_flag):
    logger = logging.getLogger("log.quant.fold.condition")
    logger.setLevel(quant_flag.log_level)
    logger.info("fold _cond op in graph")

    gh = _GraphHelper(graph)

    added_params_name, deleted_params_name = set(), []
    for sym in _topo_sort(symbol):
        name = sym.attr('name')
        attr = sym.list_attr()
        op_name = sym.attr('op_name')
        childs = sym.get_children()

        # update inputs layer symbol
        if childs is not None:
            childs = [gh.get_node(child) for child in childs]
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
            assert shift_bits.shape == (1,)

            if not quant_flag.use_scalar:
                assert "_shift_bits" in sb_param_name
                scale_name = sb_param_name.replace("_shift_bits", "_scale")
                scale_sym = mx.sym.var(scale_name, shape=(1,))

                one_name, two_name = "const_var_one", "const_var_two"
                const_var_one = gh.get_node(one_name,
                        mx.sym.var(one_name, shape=(1,)))
                const_var_two = gh.get_node(two_name,
                        mx.sym.var(two_name, shape=(1,)))

                if shift_bits < 1:
                    scale = 2 ** (-shift_bits)
                    node = mx.sym.broadcast_mul(input_sym, scale_sym)
                else:
                    scale = 2 ** (shift_bits - 1)
                    node = mx.sym.broadcast_div(input_sym, scale_sym)
                    node = mx.sym.broadcast_add(node, const_var_one)
                    node = mx.sym.floor(node)
                    node = mx.sym.broadcast_div(node, const_var_two)

                params[one_name] = mx.ndarray.array([1])
                params[two_name] = mx.ndarray.array([2])
                params[scale_name] = scale

                added_params_name.update([scale_name, one_name, two_name])

            else:
                shift_bits = shift_bits.asnumpy()[0]
                if shift_bits < 1:
                    scale = 2 ** (-shift_bits)
                    node = mx.sym.floor(input_sym * scale)
                else:
                    scale = 2 ** (shift_bits-1)
                    node = mx.sym.floor(input_sym / scale)
                    node = mx.sym.floor((node+1) / 2)

            node = mx.sym.floor(node)

            del params[sb_param_name]
            deleted_params_name.append(sb_param_name)

        graph[name] = node
    logger.debug("[ added_params_name       ]: %s", added_params_name)
    logger.debug("[ deleted_params_name     ]: %s", deleted_params_name)

    nodes = []
    for sym in symbol:
        node = gh.get_node(sym)
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

def quant_realize(symbol, params, graph, quant_flag):
    """Transform Sim-Quant(Float32 Simulate Int8) to Int8-Inference Graph
        Works:
        *) Remove floor layer in Int8 graph
        *) Cast _*_scalar op to Int32
        *) Remove unused params in graph
        *) Check&cast params type from Float32 to Int8|Int32
        *) Check supported op in cvm engine
        *) Cast broadcast_div to broadcast_right_shift


    Parameters:
    ===========
    symbol: nnvm.Symbol
    params: mxnet.ndarray.NDArray

    Returns:
    ========
    symbol: nnvm.Symbol
    params: tvm.nd.Array
    """
    logger = logging.getLogger("log.quant.post")

    def get_node(sym):
        name = sym.attr('name')
        if name not in graph:
            assert False, "Unrecognized layer:%s in sym_post_quant"%name
        return graph[name]

    # symbol
    added_params_name, deleted_params_name = [], set()
    for sym in _topo_sort(symbol):
        name = sym.attr('name')
        attr = sym.list_attr()
        op_name = sym.attr('op_name')
        childs = sym.get_children()

        if 'scalar' in attr:
            scalar = float(attr['scalar'])

            msg = "name:%s, op_name:%s, scalar:%s"%(name, op_name, attr)
            assert scalar >= INT32_MIN and scalar <= INT32_MAX, msg
            assert float(int(scalar)) == scalar, msg

            #  attr['scalar'] = max(min(int(scalar), INT8_MAX), INT8_MIN)
            attr['scalar'] = int(scalar)

        # update inputs layer symbol
        if childs is not None:
            childs = [get_node(child) for child in childs]
            # update childs inputs
            op = _get_nnvm_op(op_name)
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
        elif op_name == "broadcast_div":
            msg = '%s(op=%s, inputs=%s)'%(name, op_name, [c.attr('name') for c in childs])
            assert (len(childs) == 2)

            input_sym = childs[0]
            div_sym = childs[1]
            assert div_sym.attr('op_name') == 'null' # params or constant
            div_sym_name = div_sym.attr('name')

            assert div_sym_name in params, msg

            div = params[div_sym_name]
            shift_bits = mx.ndarray.log2(div).astype('float32')
            assert all(div >= 0)
            assert shift_bits.astype('int8').astype('float32') == shift_bits, msg

            logger.debug("broadcast_div to right_shift: %s -> (%s, %s)",
                    msg, div.asnumpy(), shift_bits.asnumpy())

            sb_sym_name = div_sym_name.replace('_scale', '') + '_shift_bits'
            if sb_sym_name in graph:
                sb_sym = graph[sb_sym_name]
            else:
                sb_sym = nnvm.sym.Variable(sb_sym_name, shape=(1,))
                graph[sb_sym_name] = sb_sym

                params[sb_sym_name] = shift_bits
                added_params_name.append(sb_sym_name)

            node = nnvm.sym.broadcast_right_shift(input_sym, sb_sym)
            deleted_params_name.add(div_sym_name)

        graph[name] = node

    for name in deleted_params_name:
        del params[name]
    logger.debug("[ added_params_name       ]: %s", added_params_name)
    logger.debug("[ deleted_params_name     ]: %s", deleted_params_name)

    nodes = []
    for sym in symbol:
        node = graph[sym.attr('name')]
        nodes.append(node)
    ret_sym = nodes[0]
    if len(nodes) > 1:
        ret_sym = nnvm.sym.Group(nodes)

    ops = set()
    for sym in _topo_sort(ret_sym):
        op_name = sym.attr('op_name')
        childs = sym.get_children()
        v = []
        if childs is not None:
            v = [(c.attr('name'), c.attr('op_name')) for c in childs]

        ops.add(op_name)

    # params
    args = ret_sym.list_input_names()
    ret_params, params_dtype = {}, {}
    for key, value in params.items():
        if key in args:
            msg = "key:%s value:%s"%(key, value)
            flat = value.asnumpy().flatten()

            assert all(flat >= INT32_MIN) and all(flat <= INT32_MAX), msg

            dtype = 'int8' if all(flat >= INT8_MIN) and all(flat <= INT8_MAX) \
                    else 'int32'

            assert all(flat.astype('int32').astype('float32') == flat), msg
            params_dtype[key] = dtype
            ret_params[key] = tvm.nd.array(value.astype(dtype).asnumpy())

    for arg in args:
        if arg == 'data':
            continue
        assert arg in ret_params, 'arg:%s in symbol not exists params'%arg

    logger.info("Existing operators: %s", sorted(ops))

    return ret_sym, ret_params

