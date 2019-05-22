import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn, SymbolBlock
from mxnet import gluon
import numpy as np
import math

from sym_utils import *
from utils import *
import sym_pass as spass
import sim_quant_helper as sim

PLACE_HOLDER = 32 # INT32
out_key = 'out_key'
target_key = 'target_key'
disable_requant_ops = [
    'Activation', 'relu',
    'Pooling',
    'slice', 'slice_like', 'slice_axis',
    'clip', 'negative',
    'repeat', 'tile', 'expand_dims',
    'Reshape', 'transpose', 'Flatten',
    '_contrib_box_nms',
]
class ANNO_TYPE():
    REQUANT = '_requant'
    IN_PREC_SQUEEZE = '_in_prec_squeeze'
    IN_PREC_SCALE = '_in_prec_scale'

def _infer_fixed_precs(sym, params, graph, inputs_ext, precs):
    logger = logging.getLogger("log.infer.fixed.precision")
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    cprecs = [precs[c.attr('name')] for c in childs] if childs else []
    precs[name] = precs[name] if name in precs else {}
    if op_name == 'null':
        return sym, params
    elif op_name in disable_requant_ops:
        pass
    elif op_name in ['sigmoid', 'exp']:
        cprecs[0][name] = 16
    elif op_name in ['Convolution', 'FullyConnected']:
        cprecs[0][name], cprecs[1][name] = 8, 8
        if eval(attr['no_bias']) == False:
            cprecs[2][name] = PLACE_HOLDER-1
    elif op_name in ['broadcast_add', 'broadcast_sub', 'elemwise_add',
            'elemwise_sub']:
        cprecs[0][name], cprecs[1][name] = PLACE_HOLDER-1, PLACE_HOLDER-1
    elif op_name in ['broadcast_mul']:
        cprecs[0][name], cprecs[1][name] = PLACE_HOLDER//2, PLACE_HOLDER//2
    elif op_name in ['Concat']:
        for prec in cprecs:
            prec[name] = PLACE_HOLDER
    elif op_name in ['sum']:
        cprecs[0][name] = 8
    else:
        logger.critical("%s name=%-40s has not been considered.",
                op_name, name)
    return sym, params

def _update_input_precs(precs, in_bit, inputs_ext):
    for k in inputs_ext:
        precs[k][out_key] = in_bit
        for n, v in precs[k].items():
            assert v >= in_bit, "input %s out of bit %s vs. %s" \
                    % (k, v, in_bit)
            precs[k][n] = in_bit

def _infer_dynamic_precs(sym, params, graph, inputs_ext, infer_shapes, precs,
        fix_param=False):
    logger = logging.getLogger("log.infer.dynamic.precision")
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    cprecs = [precs[c.attr('name')] for c in childs] if childs else []
    if op_name == 'null':
        return sym, params
    elif op_name in disable_requant_ops:
        cprecs[0][name] = cprecs[0][out_key]
        precs[name][out_key] = cprecs[0][name]
        return sym, params

    # update childs precision
    for i, c in enumerate(childs):
        c_tb = cprecs[i][name]
        c_bit = cprecs[i][out_key] if out_key in cprecs[i] else cprecs[i][name]
        if c_tb >= c_bit:
            cprecs[i][name] = c_bit

    cbits = [prec[name] for prec in cprecs]
    if op_name in ['sigmoid', 'exp']:
        precs[name][out_key] = 16
    elif op_name in ['Convolution', 'FullyConnected']:
        W_shape = infer_shapes[childs[1].attr('name')]
        sum_len = np.product(W_shape[1:])
        sum_bit = math.ceil(math.log2(sum_len))
        out_bit = cbits[0] + cbits[1] + sum_bit
        if get_attr(attr, 'no_bias', False) == False:
            if fix_param:
                out_bit = max(out_bit, cprecs[2][name])
            else:
                cprecs[2][name] = out_bit
            out_bit += 1
        precs[name][out_key] = out_bit
    elif op_name in ['broadcast_add', 'broadcast_sub', 'elemwise_add',
            'elemwise_sub', 'Concat']:
        is_params = lambda s : s.attr('op_name')=='null' and \
                s.attr('name') not in inputs_ext
        params_idx = [idx for idx,s in enumerate(childs) if is_params(s)]
        inputs_idx = [idx for idx,s in enumerate(childs) if not is_params(s)]
        assert len(inputs_idx) > 0, "Forget apply fuse constant pass first"
        out_bit = max([cbits[i] for i in inputs_idx])
        for idx in params_idx:
            if cbits[idx] > out_bit:
                if fix_param:
                    out_bit = max(out_bit, cprecs[idx][name])
                else:
                    cprecs[idx][name] = out_bit
        precs[name][out_key] = out_bit if op_name in ['Concat'] else out_bit+1
    elif op_name in ['broadcast_mul']:
        precs[name][out_key] = cbits[0] + cbits[1]
    elif op_name in ['sum']:
        axis = eval(attr['axis'])
        dshape = infer_shapes[childs[0].attr('name')]
        sum_len = np.product([dshape[i] for i in axis])
        sum_bit = math.ceil(math.log2(sum_len))
        precs[name][out_key] = cbits[0] + sum_bit

    assert precs[name][out_key] <= PLACE_HOLDER, \
        "%s name=%-40s out of PLACE_HOLDER %s" % (op_name, name, PLACE_HOLDER)
    return sym, params

def _infer_parameter_precs(sym, params, graph, inputs_ext, precs, outputs_ext):
    logger = logging.getLogger('log.infer.parameters.precision')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    if op_name != 'null' or name in inputs_ext:
        return sym, params
    param_prec = precs[name]
    if name in outputs_ext:
        ext = outputs_ext[name]
        if 'fixed' in ext and ext['fixed']:
            alpha = params[name].abs().max()
            precs[name][out_key] = math.ceil(math.log2(alpha)) + 1
    else:
        min_prec = min(list(param_prec.values()))
        assert min_prec >= 8, "%s precision=%s from %s" % (name, min_prec, param_prec)
        assert out_key not in param_prec
        param_prec[out_key] = min_prec
    logger.debug("Fixed parameter %-40s out precision: %2d from %s",
        name, min_prec, param_prec)
    return sym, params

def _annotate(sym, graph, precs, out_bit, out_tb, anno_type, logger):
    name, op_name = sym.attr('name'), sym.attr('op_name')
    logger.info("Requantize layer %-20s name=%-40s out of bit %s vs. %s",
        op_name, name, out_tb, out_bit)
    tmp_name = name + '_requant_' + str(out_tb)
    if tmp_name not in graph:
        graph[tmp_name] = mx.sym.Custom(sym, in_prec=out_bit,
                out_prec=out_tb, anno_type=anno_type,
                name=tmp_name, op_type='cvm_annotate')
        if tmp_name not in precs:
            precs[tmp_name] = { out_key: out_tb }
        precs[name][tmp_name] = out_bit
    return graph[tmp_name]
def _is_annotate_op(sym):
    op_name, attr = sym.attr('op_name'), sym.list_attr()
    if op_name == 'Custom' and attr['op_type'] == 'cvm_annotate':
        return True
    return False
def _sym_annotate(sym, params, graph, inputs_ext, precs):
    logger = logging.getLogger("log.sym.annotate")
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    if childs is None:
        return sym, params
    cprecs = [precs[c.attr('name')] for c in childs]
    new_childs = []
    for i, c in enumerate(childs):
        c_tb = cprecs[i][name]
        c_bit = cprecs[i][out_key] if out_key in cprecs[i] else cprecs[i][name]
        tmp = c
        if c_tb < c_bit:
            tmp = _annotate(c, graph, precs, c_bit, c_tb,
                    ANNO_TYPE.REQUANT, logger)
            precs[tmp.attr('name')][name] = c_tb
            precs[tmp.attr('name')][out_key] = c_tb
        new_childs.append(tmp)
    node = get_mxnet_op(op_name)(*new_childs, **attr, name=name)

    if op_name in ['sigmoid', 'exp']:
       c_tb = cprecs[0][name]
       c_bit = cprecs[0][out_key] if out_key in cprecs[0] else cprecs[0][name]
       tmp = _annotate(childs[0], graph, precs, c_bit, c_tb,
               ANNO_TYPE.IN_PREC_SCALE, logger)
       precs[tmp.attr('name')][name] = c_tb
       precs[tmp.attr('name')][out_key] = c_tb
       node = get_mxnet_op(op_name)(tmp, **attr, name=name)

    if target_key in precs[name]:
        out_tb, out_bit = precs[name][target_key], precs[name][out_key]
        if out_tb < out_bit:
            node = _annotate(node, graph, precs, out_bit, out_tb,
                    ANNO_TYPE.REQUANT, logger)
            precs[node.attr('name')][name] = out_tb
            precs[tmp.attr('name')][out_key] = out_tb
    return node, params

def _get_thresholds(out):
    return (out.min().asscalar(), out.max().asscalar())
def _sym_calibrate_th_dict(symbol, params, inputs_ext,
        calib_len=8, ctx=[mx.cpu()]):
    logger = logging.getLogger("log.sim.calibrate")
    th_dict = {}
    def _run_layer(calib_sym):
        print ([c.attr('name') for c in calib_sym])
        group = mx.sym.Group(calib_sym)
        args = group.list_inputs()
        inputs = [mx.sym.var(n) for n in inputs_ext if n in args]
        data = [inputs_ext[n.attr('name')]['data'] for n in inputs]
        ctx_data = []
        for d in data:
            tmp = gluon.utils.split_and_load(d, ctx_list=ctx, batch_axis=0, even_split=False)
            ctx_data.append(tmp)
        dlen, clen = len(ctx_data), len(ctx_data[0])
        ctx_data = [[ctx_data[i][j] for i in range(dlen)] for j in range(clen)]

        net = SymbolBlock(group, inputs)
        load_parameters(net, params, ctx=ctx)
        res = [net.forward(*d) for d in ctx_data]
        for out in res:
            for idx, s in enumerate(calib_sym):
                amin, amax = _get_thresholds(out[idx])
                sname = s.attr('name')
                if sname in th_dict:
                    amin = min(th_dict[sname][0], amin)
                    amax = max(th_dict[sname][1], amax)
                th_dict[sname] = (amin, amax)
        for s in calib_sym:
            logger.debug("calibrate layer: %-40s thresholds=%s",
                   s.attr('name'), th_dict[s.attr('name')])

    graph, calib_sym = [], []
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs = sym_iter(sym.get_children())
        if op_name == 'null':
            out = inputs_ext[name]['data'] if name in inputs_ext else params[name]
            amin, amax = _get_thresholds(out)
            th_dict[name] = (amin, amax)
            graph.append(name)
        if _is_annotate_op(sym) or \
            op_name in ['broadcast_add', 'broadcast_sub',
                'elemwise_add', 'elemwise_sub', 'Concat']:
                # 'sigmoid', 'exp']:
            calib_sym.extend(childs)

        if len(calib_sym) > calib_len:
            # remove duplicate
            calib_sym = {s.attr('name'):s for s in calib_sym if s.attr('name') not in graph}
            if len(calib_sym) <= calib_len:
                continue
            graph.extend([s.attr('name') for s in calib_sym])
            _run_layer(calib_sym)
            calib_sym = []
    if len(calib_sym) > 0:
        calib_sym = {s.attr('name'):s for s in calib_sym if s.attr('name') not in graph}
        _run_layer(calib_sym)
    return th_dict

def _update_scale_and_precs(symbol, params, inputs_ext, th_dict, precs, scales):
    logger = logging.getLogger('log.simulate.update.scale')
    def _get_scale(amin, amax, prec):
        alpha = max(abs(amin), abs(amax))
        tb_max = 2 ** (prec - 1) - 1
        return tb_max / alpha
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        cscales = [scales[c.attr('name')] for c in childs] if childs else []
        if name in scales:
            logger.debug("Fixed scale of %s with value: %s",
                    name, scales[name])
        elif op_name == 'null':
            prec = precs[name][out_key]
            amin, amax = th_dict[name]
            scales[name] = _get_scale(amin, amax, prec)
            if name in ['yolov30_yolooutputv30_broadcast_add1',
                        'yolov30_yolooutputv31_broadcast_add1',
                        'yolov30_yolooutputv32_broadcast_add1']:
                scales[name] = 1
        elif op_name in ['Convolution', 'FullyConnected']:
            if get_attr(attr, 'no_bias', False) == False:
                B_name = childs[2].attr('name')
                scales[B_name] = cscales[0] * cscales[1]
            scales[name] = cscales[0] * cscales[1]
        elif op_name in ['broadcast_mul']:
            scales[name] = cscales[0] * cscales[1]
        elif op_name in ['elemwise_add', 'elemwise_sub',
            'broadcast_add', 'broadcast_sub', 'Concat']:
            is_params = lambda s : s.attr('op_name')=='null' and \
                    s.attr('name') not in inputs_ext
            params_idx = [idx for idx,s in enumerate(childs) if is_params(s)]
            inputs_idx = [idx for idx,s in enumerate(childs) if not is_params(s)]
            assert len(inputs_idx) > 0, "Forget apply fuse constant pass first"
            out_scale = min([cscales[i] for i in inputs_idx])
            for i in params_idx:
                cname = childs[i].attr('name')
                if scales[cname] > out_scale:
                    scales[cname] = cscales[i] = out_scale
            scales[name] = min(cscales)
            for c in childs:
               cname = c.attr('name')
               amin, amax = th_dict[cname]
               alpha = max(abs(amin), abs(amax))
               int_alpha = alpha * scales[cname]
               prec = math.ceil(math.log2(int_alpha)) + 1
               assert prec <= precs[cname][out_key], \
                        "Update %s for %s precision %s vs. %s" \
                        % (cname, name, prec, precs[cname][out_key])
               precs[cname][out_key] = prec
        elif op_name in ['sum']:
            scales[name] = cscales[0]
        elif op_name in disable_requant_ops:
            scales[name] = cscales[0]
        elif _is_annotate_op(sym):
            cname = childs[0].attr('name')
            amin, amax = _get_ths(cname)
            amin, amax = th_dict[cname]
            alpha = max(abs(amin), abs(amax))
            int_alpha = alpha * scales[cname]
            prec = math.ceil(math.log2(int_alpha)) + 1
            assert prec <= precs[cname][out_key], \
                    "Update %s for %s precision %s vs. %s" \
                    % (cname, name, prec, precs[cname][out_key])
            precs[cname][out_key] = prec
            scales[name] = _get_scale(amin, amax, precs[name][out_key])
            if attr['anno_type'] == ANNO_TYPE.REQUANT:
                if prec <= precs[name][out_key]:
                    scales[name] = scales[cname]
        elif op_name in ['sigmoid', 'exp']:
            cname = childs[0].attr('name')
            in_prec = precs[cname][name]
            alpha = (2 ** (in_prec - 1)) - 1
            data = nd.array([-alpha, alpha])
            out = get_nd_op(op_name)(data / cscales[0])
            amin, amax = _get_thresholds(out)
            scales[name] = _get_scale(amin, amax, precs[name][out_key])
        else:
            logger.critical('Unrecognized op:%s(%s) . attrs(%s)', op_name, name, attr)
        logger.debug("collect layer %-20s name=%-40s scales: out_scale=%-15.5f in_scales=%s",
                op_name, name, scales[name], cscales)

def _simulate(sym, scale, in_prec, out_prec, name):
    node = mx.sym.Custom(sym, in_prec=in_prec, out_prec=out_prec,
            scale=scale, name=name, op_type='cvm_sim_quant')
    return node

    name = sym.attr('name') if prefix is None else prefix
    scale_name = name + '_scale'
    assert scale_name not in graph, "scale name %s has existed in graph" \
            % (scale_name)
    node = mx.sym.Custom(sym, in_prec)
    scale_sym = graph[scale_name] = mx.sym.var(scale_name, shape=(1,))
    params[scale_name] = nd.array([scale])

    requant_op_name = name + '_requantize'
    assert requant_op_name not in graph
    node = mx.sym.broadcast_mul(sym, scale_sym, name=requant_op_name)
    graph[requant_op_name] = node
    return node
def _simulate_layer(sym, params, graph, inputs_ext, scales, precs):
    logger = logging.getLogger('log.calib.sym.sim.requant')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()

    node = sym
    cscales = [scales[c.attr('name')] for c in childs] if childs else []
    def _restore():
        new_childs = []
        for idx, c in enumerate(childs):
            tmp = c / cscales[idx]
            new_childs.append(tmp)
        out = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
        out = out * scales[name]
        return out
    if _is_annotate_op(sym):
        # TODO: if requant_scale < 1: should not do requantize
        X_name = childs[0].attr('name')
        requant_scale = scales[name] / scales[childs[0].attr('name')]
        in_prec, out_prec = precs[X_name][out_key], precs[name][out_key]
        node = _simulate(childs[0], requant_scale, in_prec, out_prec, name)
        logger.debug("layer %-40s requant scale=%-16.8f  out=%-16.8f in=%s",
                name, requant_scale, scales[name],
                [scales[c.attr('name')] for c in childs] if childs else [])
    elif op_name in ['broadcast_add', 'broadcast_sub',
            'elemwise_add', 'elemwise_sub', 'Concat']:
        cscales = [scales[c.attr('name')] for c in childs]
        new_childs = []
        out_scale = min(cscales)
        for idx, c in enumerate(childs):
            relative_scale = out_scale / cscales[idx]
            if relative_scale != 1:
                cname = c.attr('name')
                in_prec, out_prec = precs[cname][out_key], precs[name][out_key]
                c = _simulate(c, relative_scale, in_prec, out_prec,
                        "%s_in%d_squeeze"%(name, idx))
                logger.debug("layer %-40s  adjust scale=%-16.8f orig=%-16.8f" + \
                        " for requant %-40s input scale %-16.8f",
                        c.attr('name'), relative_scale,
                        cscales[idx], name, out_scale)
            new_childs.append(c)
        node = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
    elif op_name in ['sigmoid', 'exp']:
        node = _restore()

        # cname = childs[0].attr('name')
        # in_prec = precs[cname][name]
        # alpha = (2 ** (in_prec - 1)) - 1
        # data = nd.arange(-alpha, alpha+1)
        # out = get_nd_op(op_name)(data / cscales[0])
        # weight = (out * scales[name]).round().reshape(2*alpha, 1)
        # W_name = name + '_weight'
        # assert W_name not in graph
        # W = graph[W_name] = mx.sym.var(W_name, shape=weight.shape)
        # params[W_name] = weight

        # alpha_sym, alpha_name = op_const(alpha, graph, var=mx.sym.var)
        # params[alpha_name] = nd.array([alpha])
        # X = mx.sym.broadcast_add(childs[0], alpha_sym)
        # node = mx.sym.Custom(X, W, in_dim=2*alpha,
        #         name=name, op_type='cvm_lut')
    elif op_name in ['_contrib_box_nms']:
        node = _restore()
        # scale = cscales[0]
        # overlap_thresh = get_attr(attr, 'overlap_thresh', 0.5)
        # valid_thresh = get_attr(attr, 'valid_thresh', 0)
        # attr['overlap_thresh'] = overlap_thresh * scale
        # attr['valid_thresh'] = valid_thresh * scale
        # node = get_mxnet_op(op_name)(*childs, **attr, name=name)
    # elif childs is not None:
    #     node = _restore()
    scales[node.attr('name')] = scales[name]
    precs[node.attr('name')] = precs[name]
    return node, params
def _simulate_parameters(sym, params, graph, inputs_ext, scales):
    logger = logging.getLogger('log.annotate.parameters')
    if sym.attr('op_name') != 'null':
        return sym, params
    name = sym.attr('name')
    if name in inputs_ext:
        inputs_ext[name]['scale'] = float(scales[name])
    elif name in scales:
        params[name] = params[name] * scales[name]
    return sym, params

def _extract_symbol(symbol, params, outputs_ext):
    bases = []
    graph = {}
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [graph[c.attr('name')] for c in childs]
            node = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in outputs_ext:
            bases.append(sym)
            node = mx.sym.var(name)
        graph[name] = node
    base = bases[0] if len(bases) == 1 else mx.sym.Group(bases)
    base_params = {k:params[k] for k in base.list_inputs() if k in params}
    tops = [graph[sym.attr('name')] for sym in symbol]
    top = tops[0] if len(tops) == 1 else mx.sym.Group(tops)
    top_params = {k:params[k] for k in top.list_inputs() if k in params}
    return base, base_params, top, top_params
def _merge_symbol(base, base_params, top, top_params, maps):
    graph = {maps[c.attr('name')]:c for c in base}
    for sym in topo_sort(top):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [graph[c.attr('name')] for c in childs]
            node = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in graph:
            print ("Replace top symbol %-40s with base" % (name))
            node = graph[name]
        graph[name] = node
    symbols = [graph[s.attr('name')] for s in top]
    symbol = symbols[0] if len(symbols) == 1 else mx.sym.Group(symbols)
    params = base_params
    params.update(top_params)
    params = {k:params[k] for k in symbol.list_inputs() if k in params}
    return symbol, params

def sym_annotate(symbol, params, inputs_ext, outputs_ext, in_bit=8, out_bit=8):
    logger = logging.getLogger('log.infer.precision')
    precs = {}
    topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_infer_fixed_precs, precs=precs)
    _update_input_precs(precs, in_bit, inputs_ext)
    infer_shapes = spass.sym_infer_shape(symbol, params, inputs_ext)
    topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_infer_dynamic_precs,
            infer_shapes=infer_shapes, precs=precs, fix_param=False)
    topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_infer_parameter_precs,
            precs=precs, outputs_ext=outputs_ext)
    topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_infer_dynamic_precs,
            infer_shapes=infer_shapes, precs=precs, fix_param=True)

    for sym in symbol:
        precs[sym.attr('name')][target_key] = out_bit
    symbol, params = topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_sym_annotate, precs=precs)

    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs = sym_iter(sym.get_children())
        childs = childs if childs else []
        logger.debug("%-20s name=%-40s out_prec=%s in_precs=%s",
                op_name, name, precs[name][out_key],
                [precs[c.attr('name')][name] for c in childs])
    return symbol, params, precs

def sym_simulate(symbol, params, inputs_ext, outputs_ext, precs, ctx):
    logger = logging.getLogger('log.simulate')
    for k, v in inputs_ext.items():
        assert 'data' in v, "inputs %s has not supply attribute data: %s"%(k, v)

    infer_shapes = spass.sym_infer_shape(symbol, params, inputs_ext)
    th_dict = _sym_calibrate_th_dict(symbol, params, inputs_ext, ctx=ctx)
    scales = {}
    for k, v in outputs_ext.items():
        if 'thresholds' in v:
            logger.debug("Update thresholds of output %s", k)
            th_dict[k] = v['thresholds']
        if 'fixed' in v and v['fixed']:
            scales[k] = 1
    _update_scale_and_precs(symbol, params, inputs_ext,
            th_dict, precs, scales)
    print (th_dict, scales)

    symbol, params = topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_simulate_layer,
            scales=scales, precs=precs)
    print ("Parameters: ", params.keys())
    _, params = topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_simulate_parameters, scales=scales)

    out_scales = [scales[s.attr('name')] for s in symbol]

    for k, v in inputs_ext.items():
        del v['data']
    return symbol, params, out_scales

def mixed_precision(symbol, params, inputs_ext,
        in_bit=8, out_bit=8, outputs_ext=None, ctx=[mx.cpu()]):
    if outputs_ext is None:
        outputs_ext = {s.attr('name'):{} for s in symbol}
    base, base_params, top, top_params = _extract_symbol(symbol, params,
            outputs_ext)
    inputs = [mx.sym.var(n) for n in inputs_ext]
    net1 = nn.SymbolBlock(base, inputs)
    load_parameters(net1, base_params, ctx=ctx)
    data = [inputs_ext[n]['data'].as_in_context(ctx[0]) for n in inputs_ext]
    out = net1(*data)
    out_range = [o.abs().max().asscalar() for o in out]
    out_name = [c.attr('name') for c in base]
    print (list(zip(out_name, out_range)))

    base, base_params, precs = sym_annotate(base, base_params, inputs_ext,
            outputs_ext, in_bit=in_bit, out_bit=out_bit)
    exit()
    qbase, qbase_params, out_scales = sym_simulate(base, base_params,
            inputs_ext, outputs_ext, precs, ctx)

    maps = zip([c.attr('name') for c in base], [c.attr('name') for c in qbase])
    maps = {k[0]:k[1] for k in maps}
    qsym, qparams = _merge_symbol(qbase, qbase_params, top, top_params, maps)
    return qsym, qparams, out_scales






