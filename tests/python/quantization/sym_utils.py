from mxnet.symbol import _internal
from mxnet import symbol as _sym
import nnvm as nnvm

__all__ = ["_topo_sort", "_GraphHelper", "OpExt",
        "_get_mxnet_op", "_get_nnvm_op"]


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

