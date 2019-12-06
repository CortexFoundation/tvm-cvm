from mxnet import ndarray as nd
import math
import numpy as np

from sym_utils import *
from tfm_base import *
import dataset as ds
import utils
import sim_quant_helper as sim

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
def quantize(symbol, params, th_dict, precs, scales, op_input_precs):
    infer_shapes = infer_shape(symbol, params)
    sym, params = topo_visit_transformer(symbol, params,
            apply_pass(
                "quantize",
                infer_shapes=infer_shapes,
            ),
            th_dict=th_dict,
            precs=precs, scales=scales,
            op_input_precs=op_input_precs)

    return sym, params

    def quantize_output(op, **kwargs):
        name = op.attr('name')
        th_dict = kwargs['th_dict']
        precs, scales = kwargs['precs'], kwargs['scales']

        # Requantize output symbol
        if name in precs and name in precs[name]:
            print (precs[name])
            oprec = precs[name][name]
            os = scales[name] = scale(th_dict[name], oprec)
            op, oprec, os = requant(op, oprec, os, oname=name, **kwargs)

            oname = op.attr('name')
            th_dict[oname] = th_dict[name]
            precs[oname] = oprec
            scales[oname] = os
        return op
    return topo_visit_transformer(sym, params,
            quantize_output, th_dict=th_dict,
            precs=precs, scales=scales)


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

    assert len(dim_per) > 0, "no input in graph"
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

@N.register_nm("gv")
def graph_validate(symbol, params):
    """ Graph Validate pass do some checks:
            1. examine unique names in model
            2. fuse multiple inputs into single one
            3. named the single input node `data`
            4. remove unused params
    """
    names = set()
    for sym in topo_sort(symbol):
        name = sym.attr('name')
        assert name not in names, "duplicated name in graph: %s" % name
        names.add(name)

    sym, params = transfer_multiple_inputs(symbol, params)

    def _name_replace(op, params, graph):
        name, attr = op.attr('name'), op.list_attr()
        if is_inputs(op, params):
            op = mx.sym.var("data", attr=attr)
        return op
    sym, params = topo_visit_transformer(sym, params, _name_replace)

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
def attach_input_shape(symbol, params, input_shapes):
    def _impl(op, params, graph):
        name, attr = op.attr('name'), op.list_attr()
        if is_inputs(op, params) and name in input_shapes:
            op = mx.sym.var(name, shape=input_shapes[name], attr=attr)
        return op
    return topo_visit_transformer(symbol, params, _impl)

# TODO: reduce graph for adjacent broadcast_mul
def reduce_graph(symbol, params):
    pass

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

def mx_const(number, graph, params):
    name = N.n('const_var')
    prec = math.ceil(math.log2(number)) + 1
    if name not in graph:
        attr = { 'precision': str(prec) }
        graph[name] = mx.sym.var(name, shape=(1,), attr=attr)
        params[name] = nd.array([number])
    return graph[name]

def get_bit(opt):
    if isinstance(opt, nd.NDArray):
        opt = opt.abs().max().asscalar()
    if opt == 0:
        return 1
    return math.ceil(math.log2(opt)) + 1

def get_range(prec):
    return (2 ** (prec - 1)) - 1

def scale(threshold, precision):
    assert threshold >= 0
    if threshold == 0:
        return 1
    alpha = (2 ** (precision - 1)) - 1
    return alpha / threshold

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

def realize(X, sb, prec, name=None):
    name = name if name else N.n('realize')
    if sb == 0:
        sym = mx.sym.Custom(X, precision=prec,
                name=name, op_type='cvm_clip')
    elif sb < 0:
        sym = mx.sym.Custom(X, shift_bit=-sb, precision=prec,
                name=name, op_type='cvm_left_shift')
    else:
        sym = mx.sym.Custom(X, shift_bit=sb, precision=prec,
                name=name, op_type='cvm_right_shift')
    return sym

def requant_operator(X, oprec, oscale=None, **kwargs):
    logger = logging.getLogger('log.mrt.realize')
    params, graph = kwargs['params'], kwargs['graph']
    th_dict, precs = kwargs['th_dict'], kwargs['precs']
    xopn, xn = X.attr('op_name'), X.attr('name')

    exactly = True if oscale else False
    oprec = precs[xn].get(kwargs['oname'], oprec)
    oscale = oscale if oscale else scale(th_dict[xn], oprec)
    iprec = precs[xn][OUT_KEY]
    iscale = kwargs['scales'][xn]

    if exactly:
        sb = get_bit(th_dict[xn]*iscale) - oprec
        if sb > 1:
            iprec -= sb
            X = realize(X, sb, iprec)
            iscale = iscale / (2**sb)
            logger.debug(
                "Operator  %-20s name=%-40s exactly quantize with sb=%s" +
                " scale=%s, prec=%s",
                    xopn, xn, sb, iscale, iprec)
    if exactly or iprec > oprec:
        rescale = oscale / iscale
        frac, exp = sim.cvm_float(rescale, MAX_BIT - iprec)
        sim_scale = frac * (2 ** exp)
        scale_err = abs((sim_scale - rescale) / rescale)
        if exactly and scale_err > 0.001:
            logger.warn(
                "Operator  %-20s name=%-40s requantize to scale=%s " +
                "with <%s, %d, %d>, error=%s",
                    xopn, xn, rescale, sim_scale, frac, exp, scale_err)
        oscale = iscale * frac * (2 ** exp)
        if frac > 1:
            X = realize(X, 0, iprec)
            var = mx_const(frac, graph, params)
            X = mx.sym.broadcast_mul(X, var, name=N.n("mrt_quantize_scale"))
        X = realize(X, -exp, oprec)
        logger.debug(
            "Operator  %-20s name=%-40s requantize with scale=%-16.8f<%d, %d>" +
            " iprec=%s, iscale=%-10.5f, oprec=%s, oscale=%-10.5f",
                xopn, xn, rescale, frac, exp, iprec, iscale, oprec, oscale)
    else:
        X = realize(X, 0, oprec)
        oscale = iscale
        logger.debug(
            "Operator  %-20s name=%-40s clip with iprec=%s, oprec=%s",
                xopn, xn, iprec, oprec)
    return X, oprec, oscale

def requant_parameter(wname, oprec, oscale=None, **kwargs):
    params, th_dict = kwargs['params'], kwargs['th_dict']
    logger = logging.getLogger('log.mrt.realize')
    Wn = N.n(wname)

    oprec = kwargs['precs'][wname].get(kwargs['oname'], oprec)
    oscale = oscale if oscale else scale(th_dict[wname], oprec)
    params[Wn] = sim.int_realize(params[wname] * oscale, oprec, logger=logger)
    attr = { 'precision': str(oprec) }
    W = mx.sym.var(Wn, shape=params[Wn].shape, attr=attr)

    logger.debug(
        "Parameter th_dict=%-12.8f name=%-40s requantize with scale=%-16.8f to" +
        " prec=%s",
            th_dict[wname], wname, oscale, oprec)
    return W, oprec, oscale

def requant(sym, oprec, oscale=None, **kwargs):
    if is_params(sym, kwargs['params']):
        return requant_parameter(sym.attr('name'), oprec, oscale, **kwargs)
    else:
        return requant_operator(sym, oprec, oscale, **kwargs)

def requant_output(op, name, **kwargs):
    infer_shapes, th_dict = kwargs['infer_shapes'], kwargs['th_dict']
    precs, scales = kwargs['precs'], kwargs['scales']

    oname = op.attr('name')
    infer_shapes[oname] = infer_shapes[name]
    th_dict[oname] = th_dict[name]
    precs[oname] = precs[name]
    scales[oname] = scales[name]

    # Requantize output symbol
    # if name in precs[name]:
    #     print (precs[name])
    #     oprec = precs[name][name]
    #     os = scale(th_dict[name], oprec)
    #     op, oprec, os = requant_operator(op, oprec, os, oname=name, **kwargs)

    #     oname = op.attr('name')
    #     infer_shapes[oname] = infer_shapes[name]
    #     th_dict[oname] = th_dict[name]
    #     precs[oname] = oprec
    #     scales[oname] = os
    return op

