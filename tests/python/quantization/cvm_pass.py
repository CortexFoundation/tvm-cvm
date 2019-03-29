import tvm

import logging

from sym_utils import *

def _get_node(sym, graph):
    name = sym.attr('name')
    if name not in graph:
        assert False, "Unrecognized layer:%s in sym_post_quant"%name
    return graph[name]

requant_op_name = ['broadcast_left_shift', 'broadcast_right_shift']

def _extract_requant_op(sym, params):
    """
        For divide:
            data = broadcast_right_shift(data, sb)
            data = broadcast_add(data, 1)
            data = broadcast_right_shift(data, 1)
            data = clip(data, min=-127, max=127)

        For multiply:
            data = brocast_left_shift(data, sb)
            data = clip(data, min=-127, max=127)
    """
    op_name = sym.attr('op_name')
    childs = sym.get_children()

    const_one_array = tvm.nd.array([1])

    if op_name == requant_op_name[0]:
        return sym
    elif op_name == requant_op_name[1]:
        if childs[1].attr('op_name') != 'null':
            return None

        name = childs[1].attr('name')
        assert name in params and \
            all(params[name] == const_one_array)

        psym = childs[0]
        pname, pchilds = psym.attr('op_name'), psym.get_children()
        if pname != 'broadcast_add':
            return None

        if pchilds[1].attr('op_name') != 'null':
            return None

        name = pchilds[1].attr('name')
        assert name in params and \
            all(params[name] == const_one_array)

        dsym = pchilds[0]
        dname, dchilds = dsym.attr('op_name'), dsym.get_children()
        if dname != 'broadcast_right_shift':
            return None

        name = dchilds[1].attr('name')
        assert name == params
        params[name] = tvm.nd.array(params[name].asnumpy() + 1)
        return dchilds[0]

    return None


def create(symbol, params, quant_flag, g={}):
    logger = logging.getLogger("log.quant.post")

    added_params_name, deleted_params_name = [], []
    for sym in _topo_sort(symbol):
        name = sym.attr('name')
        attr = sym.list_attr()
        op_name = sym.attr('op_name')
        childs = sym.get_children()

        # update inputs layer symbol
        if childs is not None:
            childs = [_get_node(child, g) for child in childs]
            op = _get_nnvm_op(op_name)
            node = op(*childs, **attr)
        elif op_name != 'null':
            assert False, "Unrecognized op without input"
        else:
            # inputs or params
            node = sym

        requant_sym = _extract_requant_op(sym)
        if requant_sym is not None:
            node = requant_sym

        if op_name == 'clip':
            node = childs[0]

        g[name] = node

    nodes = []
    for sym in symbol:
        node = graph[sym.attr('name')]
        nodes.append(node)
    ret_sym = nodes[0]
    if len(nodes) > 1:
        ret_sym = nnvm.sym.Group(nodes)

    return ret_sym, params


