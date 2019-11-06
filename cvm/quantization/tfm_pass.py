from mxnet import ndarray as nd
import math
import numpy as np
from sym_utils import *
from tfm_base import *
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
def quantize(symbol, params, th_dict, precs, scales, op_input_precs):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("quantize", infer_shapes=infer_shapes),
            th_dict=th_dict, precs=precs, scales=scales,
            op_input_precs=op_input_precs)

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

OUT_KEY = "out_key"
TARGET_KEY = "target_key"
MAX_BIT = 32

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

def scale(threshold, precision):
    assert threshold >= 0
    if threshold == 0:
        return 1
    alpha = (2 ** (precision - 1)) - 1
    return alpha / threshold

def _mrt_sim_quantize(sym, sb, params, graph, prec):
    name = "%s_%d_%d" % (sym.attr('name'), sb, prec)
    if name not in graph:
        graph[name] = mx.sym.Custom(sym, sb=sb, prec=prec,
                name=name, op_type='mrt_sim_quant')
    return graph[name]

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

def requant_operator(X, oname, def_prec, oscale=None, **kwargs):
    logger = logging.getLogger('log.mrt.simulate')
    params, graph = kwargs['params'], kwargs['graph']
    th_dict, precs = kwargs['th_dict'], kwargs['precs']
    xopn, xn = X.attr('op_name'), X.attr('name')

    exactly = True if oscale else False
    oprec = precs[xn].get(oname, def_prec)
    oscale = oscale if oscale else scale(th_dict[xn], oprec)
    iscale, iprec = kwargs['scales'][xn], precs[xn][OUT_KEY]
    '''
    if exactly:
        in_prec = _get_bit(kwargs['th_dict'][xn] * iscale)
        out_prec = prec.p
        sb = in_prec - out_prec if in_prec > out_prec else 0
        if sb > 1:
            iprec = PREC(iprec.p - sb)
            X = _mrt_sim_quantize(X, sb, params, graph, iprec.p)
            iscale = iscale / (2 ** sb)
            logger.debug(
                "Operator  %-20s name=%-40s exactly quantize with sb=%s" +
                " scale=%s, prec=%s",
                    xopn, xn, sb, iscale, iprec)
    '''

    if exactly or (iprec > oprec and iscale > oscale):
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
            X = _mrt_sim_quantize(X, 0, params, graph, iprec.p)
            var = mx_const(frac, graph, params)
            mul_name = _uniq_name("mrt_quantize_scale")
            X = mx.sym.broadcast_mul(X, var, name=mul_name)
        X = _mrt_sim_quantize(X, (-exp), params, graph, oprec)
        logger.debug(
            "Operator  %-20s name=%-40s requantize with scale=%-16.8f<%d, %d>" +
            " iprec=%s, iscale=%-10.5f, oprec=%s, oscale=%-10.5f",
                xopn, xn, rescale, frac, exp, iprec, iscale,
                oprec, oscale)
    else:
        X = _mrt_sim_quantize(X, 0, params, graph, oprec)
        oscale = iscale
        logger.debug(
            "Operator  %-20s name=%-40s clip with iprec=%s, oprec=%s",
                xopn, xn, iprec, oprec)
    return X, oprec, oscale

def requant_parameter(pname, def_prec, oscale=None, **kwargs):
    precs = kwargs['precs']
    P_name = N.n(pname)
    P_prec = precs[pname].get(name, def_prec)
    xs = oscale if oscale else scale(th_dict[pname], P_prec.p)
    params[P_name] = params[pname] * xs
    P_attr = { 'precision': str(P_prec.p) }
    graph[P_name] = mx.sym.var(P_name,
            shape=params[P_name].shape, attr=P_attr)
    logger.debug(
        "Parameter th_dict=%-12.8f name=%-40s requantize with scale=%-16.8f to prec=%s",
            th_dict[pname], pname, xs, P_prec)
    return graph[P_name], P_prec, xs

def requant():
    pass

