from mxnet.symbol import _internal
from mxnet import symbol as _sym
import nnvm as nnvm

__all__ = ["_topo_sort", "_GraphHelper", "OpExt",
        "_get_mxnet_op", "_get_nnvm_op", "_identity_ext",
        "INT8_MIN", "INT8_MAX", "INT32_MIN", "INT32_MAX"]

INT32_MIN, INT32_MAX = -2147483647, 2147483647
INT8_MIN, INT8_MAX = -127, 127

INT8_TYPE, INT32_TYPE= ('int8', 'int32')


class OpExt():
    def __init__(self, op_name='null',
            in_types=[], out_types=[]):
        self.op_name = op_name
        self.in_types = in_types
        self.out_types = out_types


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
            childs_ = (childs) if isinstance(childs, nnvm.symbol.Symbol) else (childs.list_outputs())
            dep_cnts[name] = len({childs[idx].attr('name') for idx, c  in enumerate(childs_)})
            for idx, _  in enumerate(childs_):
                child = childs[idx]
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
_identity_ext = {
    'null': OpExt(out_types=[INT8_TYPE, INT32_TYPE]),

    'relu': OpExt('relu', [INT8_TYPE], [INT8_TYPE]),
    'max_pool2d': OpExt('max_pool2d', [INT8_TYPE], [INT8_TYPE]),

    'conv2d': OpExt('conv2d', [INT8_TYPE], [INT32_TYPE]),
    'dense': OpExt('dense', [INT8_TYPE], [INT32_TYPE]),
    'sum': OpExt('sum', [INT8_TYPE], [INT32_TYPE]),
    'elemwise_add': OpExt('elemwise_add', [INT8_TYPE], [INT32_TYPE]),

    'reshape': OpExt('reshape', [INT8_TYPE, INT32_TYPE], [INT8_TYPE, INT32_TYPE]),
    'flatten': OpExt('flatten', [INT8_TYPE, INT32_TYPE], [INT8_TYPE, INT32_TYPE]),

    'broadcast_right_shift': OpExt('broadcast_right_shift', [INT32_TYPE], [INT8_TYPE]),
    'broadcast_left_shift': OpExt('broadcast_left_shift', [INT32_TYPE], [INT8_TYPE]),
    'broadcast_div': OpExt('broadcast_div', [INT32_TYPE], [INT32_TYPE]),
    'broadcast_mul': OpExt('broadcast_mul', [INT32_TYPE], [INT32_TYPE]),
    'broadcast_add': OpExt('broadcast_mul', [INT32_TYPE], [INT32_TYPE]),

    'clip': OpExt('clip', [INT32_TYPE], [INT8_TYPE]),
}
