from mxnet import ndarray as nd
import math
import numpy as np
from sym_utils import *
from tfm_base import *


out_key = 'out_key'
target_key = 'target_key'

# === symbol pass == 

def calculate_ops(symbol, params, normalize=True):
    ops, infer_shapes = [0], infer_shape(symbol, params)
    def _impl(op, **kwargs):
        ops[0] += apply_pass("calculate_ops")(op, **kwargs)
    topo_visit_transformer(symbol, params, _impl,
            infer_shapes=infer_shapes)

    ops = ops[0]
    if normalize:
        LEVELS = ['', 'K', 'M', 'G', 'T', 'P']
        idx = 0
        while ops > 1000:
            ops /= 1000
            idx += 1
        ops = "{:5.2f}{}".format(ops, LEVELS[idx])
    return ops

@N.register_nm("fmi")
def transfer_multiple_inputs(sym, params):
    infer_shapes = infer_shape(sym, params)
    dim_sum, dim_per, dims = 0, {}, {}
    def _sum_input(node, params, **kwargs):
        name = node.attr('name')
        nonlocal dim_sum, dim_per, dims
        if is_inputs(node, params):
            dims[name] = infer_shapes[name][0]
            dot = np.product(dims[name])
            dim_per[name] = dot
            dim_sum += dot
    topo_visit_transformer(sym, params, _sum_input)

    assert len(dim_per) > 0, "Graph has no input"
    if len(dim_per) == 1:
        return sym, params

    data_sum = mx.sym.var('data', shape=(dim_sum,))
    first, last = 0, 0
    def _change_node(op, params, graph, **kwargs):
        name = op.attr('name')
        if is_inputs(op, params):
            nonlocal first, last
            last = first + dim_per[name]
            op = mx.sym.slice(data_sum, name=N.n('slice'),
                    begin=(first,), end=(last,))
            op = mx.sym.reshape(op, name=N.n('reshape'),
                    shape=dims[name])
            first = last
        return op
    sym, params = topo_visit_transformer(sym, params, _change_node)
    return sym, params

@N.register_nm("fuse_transpose")
def fuse_transpose(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("fuse_transpose", infer_shapes=infer_shapes))

@N.register_nm("validate")
def validate(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("validate", infer_shapes=infer_shapes))

@N.register_nm("rewrite")
def rewrite(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("rewrite", infer_shapes=infer_shapes))

@N.register_nm("quantize")
def quantize(symbol, params, th_dict, precs, scales, **kwargs):
    th_dict = check_fixed()
    assert th_dict is not None, "Please calibrate thresholds first."
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("quantize", infer_shapes=infer_shapes,
            th_dict=th_dict, precs=precs, scales=scales))

@N.register_nm("cvm")
def compile(symbol, params):
    def _as_list(arr):
        return arr if isinstance(arr, list) else [arr]

    infer_shapes = infer_shape(symbol, params)
    graph = {}
    for op in topo_sort(symbol):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        childs = [] if childs is None else childs
        childs = [get_node(c, graph) for c in childs]
        childs = [x for y in childs for x in _as_list(y)]
        op = apply_pass("compile", infer_shapes=infer_shapes)(
                op, childs=childs, attr=attr)
        graph[name] = op

    nodes = []
    for sym in symbol:
        node = get_node(sym, graph)
        nodes.append(node)
    if len(nodes) > 1:
        return nnvm.sym.Group(nodes), params
    return nodes[0], params

# === symbol helper ===

@N.register_nm("gv")
def graph_validate(symbol, params):
    # Check no duplicate name
    names = set()
    for sym in topo_sort(symbol):
        name = sym.attr('name')
        assert name not in names, "duplicated name in graph: %s" % name
        names.add(name)

    # Repalce names
    def _name_replace(op, params, graph):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if is_inputs(op, params):
            assert "data" not in graph, "multiple inputs"
            graph["data"] = op = mx.sym.var("data", attr=attr)
        elif is_params(op, params):
            pass
        elif childs is None:
            op = get_mxnet_op(op_name)(name=name, **attr)
        else:
            op = get_mxnet_op(op_name)(*childs, name=name, **attr)
        return op
    sym, params = topo_visit_transformer(symbol, params, _name_replace)

    new_params = {s.attr('name'):params[s.attr('name')] \
            for s in topo_sort(sym) if is_params(s, params)}

    return sym, new_params

@N.register_nm("fc")
def fuse_constant(symbol, params):
    def _impl(op, params, graph):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if is_var(op, params):
            pass
        elif childs is None:
            params[name] = get_nd_op(op_name)(**attr)
            op = mx.sym.var(name, shape=params[name].shape)
        elif all([is_params(c, params) for c in childs]):
            in_params = [params[c.attr('name')] for c in childs]
            params[name] = get_nd_op(op_name)(*in_params, **attr)
            op = mx.sym.var(name, shape=params[name].shape)
        return op
    return topo_visit_transformer(symbol, params, _impl)

@N.register_nm("ais")
def attach_input_shape(symbol, params, input_shape):
    def _impl(op, params, graph):
        if is_inputs(op, params):
            op = mx.sym.var(op.attr('name'), shape=input_shape)
        return op
    return topo_visit_transformer(symbol, params, _impl)

def infer_shape(symbol, params, input_shape=None):
    infer_shapes = {}
    def _impl(op, params, graph):
        name, op_name = op.attr('name'), op.attr('op_name')
        _, oshp, _ = op.infer_shape()

        if is_params(op, params):
            if oshp is None:
                oshp = [params[name].shape]
                op = mx.sym.var(name, shape=oshp[0])
            assert params[name].shape == oshp[0], \
                    "Parameter %s's shape %s is inconsistent with \
                    params dict %s" % (name, oshp[0], params[name].shape)
        elif is_inputs(op, params):
            if input_shape is None:
                assert oshp is not None, "It seems that graph doesn't set \
                        input_shape, please invoke attach_input_shape first."
            else:
                oshp = [input_shape]
                op = mx.sym.var(name, shape=oshp[0])
        infer_shapes[name] = oshp
        return op
    topo_visit_transformer(symbol, params, _impl)
    return infer_shapes

def _collect_attribute(op, **kwargs):
    attr_name, func = kwargs['attr_name'], kwargs['func']
    func(op.attr(attr_name))
    return op

def collect_op_names(symbol, params):
    op_names = set()
    _ = topo_visit_transformer(symbol, params, _collect_attribute,
            attr_name='op_name', func=op_names.add)
    return op_names

# === MRT ===

def _get_opt(out, lambd):
    absmax = out.abs().max().asscalar()
    if lambd is None:
        return absmax
    mean = nd.mean(out).asscalar()
    std = nd.norm(out - mean).asscalar() / math.sqrt(np.product(out.shape))
    alpha = abs(mean) + lambd * std
    if alpha < 0.95 * absmax:
        print ("[", mean, std, "]", alpha, absmax)
        return alpha
    return absmax

def _get_bit(opt):
    if isinstance(opt, nd.NDArray):
        opt = opt.abs().max().asscalar()
    if opt == 0:
        return 1
    return math.ceil(math.log2(opt)) + 1

def sym_calibrate(symbol, params, data, **kwargs):
    logger = logging.getLogger('log.mrt')
    _, deps = topo_sort(symbol, logger=logger, with_deps=True)
    th_dict, out_cache = {}, {}
    ctx = kwargs.get('ctx', mx.cpu())
    logger.info("calibrate model outputs")

    def _impl(op, params, graph, **kwargs):
        deps, old_ths = kwargs['deps'], kwargs['old_ths']
        logger = logging.getLogger('log.mrt.calibrate')
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if op_name == 'null':
            out = data if is_inputs(op, params) else params[name]
        elif childs is None:
            out = get_nd_op(op_name)(**attr)
        else:
            cinfos = [(c.attr('name'), get_entry_id(c)) for c in childs]
            nd_inputs = [out_cache[n[0]][n[1]] for n in cinfos]
            out = get_nd_op(op_name)(*nd_inputs, **attr)
            for n, _ in cinfos:
                assert n in deps
                deps[n].remove(name)
                if len(deps[n]) == 0:
                    del out_cache[n]
        out = [out] if len(op) == 1 else out
        out_cache[name] = [o.as_in_context(ctx) for o in out]
        opts = float(_get_opt(out[0], kwargs['lambd']))
        if old_ths and name in old_ths:
            th_dict[name] = max(old_ths[name], opts)
        else:
            th_dict[name] = opts
            p = logger.debug if opts < 30 else logger.warn
            p("collect symbol %-40s out_shape=%-20s th_dict: (%s)",
                    name, [o.shape for o in out], th_dict[name])

    topo_visit_transformer(symbol, params, _impl, logger=logger,
            deps=deps, data=data, **kwargs)
    out_cache.clear()

    return th_dict


def check_fixed(symbol, params, th_dict):
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
    return th_dict

def requant_operator():
    pass

def requant_operator():
    pass

def requant():
    pass

