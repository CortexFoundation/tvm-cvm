import logging
import os
import numpy as np

import mxnet as mx
from mxnet.gluon import nn, SymbolBlock
from mxnet import ndarray as nd
import nnvm as nnvm
import tvm

from sym_utils import *
from sym_pass import *
from quant_utils import *
from utils import *
import sim_quant_helper as sim
import cvm_op as cvm

max_bit = 32 # INT32
input_target_bit = 8
default_target_bit = 16 # INT8
bias_target_bit = default_target_bit * 4 - 1
disable_requant_ops = [
    'Activation', 'relu',
    'Pooling',
    'slice', 'slice_like', 'slice_axis',
    'clip', 'negative',
    'repeat', 'tile', 'expand_dims',
    'Reshape', 'transpose', 'Flatten',
    '_contrib_box_nms',
]

def _collect_symbol_ext(sym, params, graph, inputs_ext, scale_shapes):
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()

    scale_shapes[name] = (1,)
    # cshapes = [scale_shapes[c.attr('name')] for c in childs] if childs else []
    # if op_name == 'Convolution' and attr['num_group'] == attr['num_filter']:
    #     channel = int(attr['num_filter'])
    #     scale_shapes[name] = (1, channel, 1, 1)
    # elif op_name == 'Convolution' and len(cshapes[0]) > 1:
    #     scale_shapes[childs[1].attr('name')] = cshapes[0]
    # if op_name == 'Convolution' and attr['num_group'] == attr['num_filter']:
    #     X, W = childs[0], childs[1]
    #     channel = int(attr['num_filter'])
    #     scale_shapes[W.attr('name')] = (channel, 1, 1, 1)
    #     scale_shapes[name] = (1, channel, 1, 1)
    #     # while X.attr('name') in disable_requant_ops:
    #     #     scale_shapes[X.attr('name')] = (1, channel, 1, 1)
    #     #     X = sym_iter(X.get_children())[0]
    #     # scale_shapes[X.attr('name')] = (1, channel, 1, 1)
    #     if attr['no_bias'] == 'False':
    #         B_name = childs[2].attr('name')
    #         scale_shapes[B_name] = (channel,)
    if op_name in disable_requant_ops:
        scale_shapes[name] = scale_shapes[childs[0].attr('name')]

    return sym, params

def _get_thresholds(output, calib_mode='naive'):
    min_range = output.min().asscalar()
    max_range = output.max().asscalar()
    return (min_range, max_range)
def _calib_sym_collect_thresholds(sym, params, graph, inputs_ext,
        scale_shapes, th_dict, calib_data,
        calib_mode='naive', ctx=mx.gpu()):
    logger = logging.getLogger('log.calib.sym.collect.thresholds')

    name, op_name = sym.attr('name'), sym.attr('op_name')
    attr, childs = sym.list_attr(), sym_iter(sym.get_children())

    if op_name == 'null':
        if name in inputs_ext:
            if calib_data is not None:
                output = calib_data
            else:
                output = inputs_ext[name]['data']
        else:
            output = params[name]
    elif op_name in disable_requant_ops:
        th_dict[name] = th_dict[childs[0].attr('name')]
        return sym, params
    else:
        args = sym.list_inputs()
        inputs = [mx.sym.var(n) for n in inputs_ext if n in args]
        if calib_data is not None:
            data = [calib_data.as_in_context(ctx)]
        else:
            data = [inputs_ext[n.attr('name')]['data'] for n in inputs]
            data = [d.as_in_context(ctx) for d in data]
        graph = SymbolBlock(sym, inputs)
        load_parameters(graph, params, ctx=ctx)
        output = graph.forward(*data)

    slices = [output]
    shape = scale_shapes[name]
    for idx, s in enumerate(shape):
        if s == 1 :
            continue
        begin, end = [None]*len(shape), [None]*len(shape)
        tmp_slices = []
        for sli in slices:
            for start in range(s):
                begin[idx], end[idx] = start, start+1
                tmp = sli.slice(begin=begin, end=end)
                tmp_slices.append(tmp)
        slices = tmp_slices

    th_dict[name] = nd.zeros((len(slices), 2))
    for idx, out in enumerate(slices):
        th_dict[name][idx] = _get_thresholds(out, calib_mode)
    logger.debug("collect symbol %-30s out_shape=%-20s vs. sb_shape=%-20s th_dict: (%s, %s)",
            name, output.shape, shape,
            th_dict[name].min().asscalar(), th_dict[name].max().asscalar())
    return sym, params

def _sim_requantize_op(sym, scale, params, graph, prefix=None):
    name = sym.attr('name') if prefix is None else prefix
    scale_name = name + '_requant_scale'
    assert scale_name not in graph, "scale name %s has existed in graph" \
            % (scale_name)
    scale_sym = graph[scale_name] = mx.sym.var(scale_name, shape=scale.shape)
    params[scale_name] = scale

    requant_op_name = name + '_requant_op'
    assert requant_op_name not in graph
    node = mx.sym.broadcast_mul(sym, scale_sym, name=requant_op_name)
    graph[requant_op_name] = node
    return node
def _is_sim_requantize_op(sym):
    name = sym.attr('name')
    return True if name.endswith('_requant_op') else False
def _realize_tvm_requant_op(sym, sb, params, graph, target_bit):
    """Requantize Op:
        out = round(sym >> sb)  if sb >  0
        out = round(sym)        if sb == 0
        out = round(sym << -sb) if sb <  0

        round(sym >> sb) = int((int(sym >> (sb - 1)) + 1) >> 1)

        out = clip_int(out)
    """
    out = mx.sym.round(sym) # avoid precision loss represented in float32
    sb, tb = sb.asscalar(), target_bit.asscalar()
    if sb < 0:
        out = out * (2 ** -sb)
        out = mx.sym.round(sym)
    elif sb > 0:
        if sb > 1:
            out = out / (2 ** (sb - 1))
            out = mx.sym.floor(out)
        out = out + 1
        out = out / 2
        out = mx.sym.floor(out)
    clip_range = 2 ** (tb - 1) - 1
    out = mx.sym.clip(out, a_min=-clip_range, a_max=clip_range)
    return out
def _realize_cvm_requant_op(sym, sb, params, graph, target_bit):
    name = sym.attr('name')
    requant_op = name + '_cvm_shift'
    assert requant_op not in graph
    sb, tb = int(sb.asscalar()), int(target_bit.asscalar())
    if sb == 0:
        return mx.sym.Custom(sym, precision=tb,
                cvm_name=requant_op,
                name=requant_op, op_type='cvm_clip')
    elif sb < 0:
        return mx.sym.Custom(sym, shift_bit=-sb, precision=tb,
                name=requant_op, op_type='cvm_left_shift')
    else:
        return mx.sym.Custom(sym, shift_bit=sb, precision=tb,
                cvm_name=requant_op,
                name=requant_op, op_type='cvm_right_shift')

def _collect_scale_helper(sym, params, graph, inputs_ext,
        th_dict, scale_shapes, get_scale, scale_helper, target_bits):
    logger = logging.getLogger('log.calib.sim.scale')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()

    scale_helper[name] = get_scale(th_dict[name], default_target_bit)
    scale_helper[name] = scale_helper[name].reshape(scale_shapes[name])
    target_bits[name] = default_target_bit
    if op_name == 'null':
        if name in inputs_ext:
            inputs_ext[name]['target_bit'] = input_target_bit
            target_bits[name] = input_target_bit
            scale_helper[name] = get_scale(th_dict[name], input_target_bit)
            scale_helper[name] = scale_helper[name].reshape(scale_shapes[name])
    elif op_name in ['Convolution', 'FullyConnected']:
        X_name, W_name = childs[0].attr('name'), childs[1].attr('name')
        if attr['no_bias'] == 'False':
            B_name = childs[2].attr('name')
            scale_helper[B_name] = (scale_helper[X_name] * scale_helper[W_name]).min()
            target_bits[B_name] = bias_target_bit
    elif op_name in ['sigmoid', 'exp']:
        X_name = childs[0].attr('name')
        X_bit = target_bits[X_name]
        _range = (2 ** (X_bit - 1)) - 1
        d = nd.concat(nd.arange(0, _range+1), nd.arange(-_range, 0), dim=0)
        lut = get_nd_op(op_name)(d / scale_helper[X_name])
        th = nd.array([[lut.max().asscalar(), lut.min().asscalar()]])
        scale_helper[name] = get_scale(th, default_target_bit)
        params[name + '_lut'] = lut * scale_helper[name]
    return sym, params
def _annotate_layer(sym, params, graph, inputs_ext,
        scale_helper, target_bits, infer_shapes):
    logger = logging.getLogger('log.calib.sym.sim.requant')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()

    node = sym
    cscales = [scale_helper[c.attr('name')] for c in childs] if childs else []
    cbits = [target_bits[c.attr('name')] for c in childs] if childs else []
    if op_name == 'null':
        return node, params
    elif op_name in disable_requant_ops:
        return node, params
    elif op_name in ['sigmoid', 'exp']:
        lut_name = name + '_lut'
        assert lut_name in params
        lut_sym = mx.sym.var(lut_name, shape=params[lut_name].shape)
        node = mx.sym.Custom(childs[0], lut_sym,
                name=name, op_type='cvm_lut')
        requant_scale = scale_helper[name]
    elif op_name == 'Convolution' and len(cscales[0].shape) > 1:
        # Rewrite conv op for depth-wise
        assert attr['kernel'] == "(1, 1)", "Assert failed: " + \
            "depth-wise conv not followed by 1*1 conv(%-40s): %s" % (name, attr)
        X, W = childs[0], childs[1]
        ic = int(attr['num_filter'])
        X = mx.sym.expand_dims(X, axis=1)
        target_bits[X.attr('name')] = cbits[0]
        out = mx.sym.broadcast_mul(X, W)
        target_bit = cbits[0] + cbits[1]
        target_bits[out.attr('name')] = target_bit
        in_scale = (cscales[0] * cscales[1]).min()
        relative_scale = in_scale / (cscales[0] * cscales[1])
        relative_scale = relative_scale.expand_dims(axis=1)
        out = _sim_requantize_op(out, relative_scale, params, graph)
        target_bits[out.attr('name')] = target_bit
        logger.debug("layer %-40s rewrite for depth-wise conv in_scale=%+16.8f" +
                " out_scale=%+16.8f X_shape=%s W_shape=%s", name,
                in_scale.asscalar(), scale_helper[name].asscalar(),
                cscales[0].shape, cscales[1].shape)
        node = mx.sym.sum(out, axis=2)
        sum_bit = math.ceil(math.log2(ic)) + target_bit
        target_bits[node.attr('name')] = sum_bit
        if attr['no_bias'] == 'False':
            B_name = childs[2].attr('name')
            B_shape = params[B_name].shape
            params[B_name] = params[B_name].reshape([1, *B_shape, 1, 1])
            graph[B_name] = B = mx.sym.var(B_name, shape=params[B_name].shape)
            node = mx.sym.broadcast_add(node, B)
            target_bits[node.attr('name')] = 1 + sum_bit
        requant_scale = scale_helper[name] / in_scale
    elif op_name in ['Convolution', 'FullyConnected', 'broadcast_mul']:
        requant_scale = scale_helper[name] / (cscales[0] * cscales[1])
    elif op_name in ['elemwise_add', 'elemwise_sub',
            'broadcast_add', 'broadcast_sub', 'Concat']:
        new_childs = []
        in_scale = min(cscales)
        for idx, c in enumerate(childs):
            relative_scale = in_scale / cscales[idx]
            if relative_scale != 1:
                c = _sim_requantize_op(c, relative_scale, params, graph,
                        "%s_in%d"%(name, idx))
                target_bits[c.attr('name')] = cbits[idx]
                logger.debug("layer %-40s  adjust scale=%-16.8f orig=%-16.8f" + \
                        " for requant %-40s input scale %-16.8f",
                        c.attr('name'), relative_scale.asscalar(),
                        cscales[idx].asscalar(), name, in_scale.asscalar())
            new_childs.append(c)
        requant_scale = scale_helper[name] / in_scale
        node = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
    elif op_name in ['sum']:
        requant_scale = scale_helper[name] / cscales[0]
    else:
        logger.critical('Unrecognized op:%s(%s) . attrs(%s)', op_name, name, attr)

    if not (requant_scale.asnumpy() == 1).all():
        r = (2**(default_target_bit-1)-1) / requant_scale.min().asscalar()
        target_bits[node.attr('name')] = math.ceil(math.log2(r)) + 1
        node = _sim_requantize_op(node, requant_scale, params, graph)
        logger.debug("layer %-40s requant scale=%-16.8f  out=%-16.8f in=%s",
                name, requant_scale.min().asscalar(), scale_helper[name].min().asscalar(),
                [scale_helper[c.attr('name')].min().asscalar() for c in childs] \
                if childs else [])
    scale_helper[node.attr('name')] = scale_helper[name]
    target_bits[node.attr('name')] = default_target_bit
    infer_shapes[node.attr('name')] = infer_shapes[name]
    return node, params
def _annotate_parameters(sym, params, graph, inputs_ext,
        scale_helper, target_bits):
    logger = logging.getLogger('log.annotate.parameters')
    if sym.attr('op_name') != 'null':
        return sym, params
    name = sym.attr('name')
    if name in inputs_ext:
        inputs_ext[name]['scale'] = float(scale_helper[name].asscalar())
    elif name in scale_helper:
        params[name] = params[name] * scale_helper[name]
    return sym, params
def _realize_symbol(sym, params, graph, inputs_ext,
        target_bits, runtime="cvm"):
    logger = logging.getLogger('log.calib.realize')
    if not _is_sim_requantize_op(sym):
        return sym, params

    assert runtime in ["cvm", "tvm"]
    if runtime == "cvm":
        _realize_func = _realize_cvm_requant_op
        # _realize_func = _realize_broadcast_op
    else:
        _realize_func = _realize_tvm_requant_op

    childs = sym_iter(sym.get_children())
    X, B = childs[0], childs[1]
    X_name, B_name = X.attr('name'), B.attr('name')
    name = sym.attr('name')
    assert X_name in target_bits and name in target_bits, \
        "%s(%s, %s) not in precs %s" \
        % (name, X_name, B_name, target_bits.keys())

    if (params[B_name].asnumpy() == 1).all():
        logger.debug("layer %s skip realize requant with scale 1", name)
        return X, params

    def cal_bit(A_bit, B_bit, sb):
        # A_target_bit, B_target_bit = 16, 16
        # A_target_bit = min(A_bit, A_target_bit)
        # B_target_bit = min(B_bit, B_target_bit)
        # A_target_bit = 32 - B_target_bit if B_target_bit < 16 else A_target_bit
        # B_target_bit = 32 - A_target_bit if A_target_bit < 16 else B_target_bit
        # A_target_bit = min(A_bit, A_target_bit)
        # B_target_bit = min(B_bit, B_target_bit)
        max_bit = 32
        total_bit = A_bit + B_bit
        excess_bit = (total_bit - max_bit) // 2 if total_bit > max_bit else 0
        A_target_bit = A_bit - excess_bit
        B_target_bit = min(B_bit - excess_bit, 32 - A_target_bit)
        A_sb, B_sb = A_bit - A_target_bit, B_bit - B_target_bit
        Y_sb = (-sb) - A_sb - B_sb
        return A_sb, A_target_bit, B_sb, B_target_bit, Y_sb

    frac, sb = sim.parse_nd_float(params[B_name])
    shape = params[B_name].shape
    # size = len(frac)
    # A_sb, A_tb = [None] * size, [None] * size
    # Y_sb, Y_tb = [None] * size, [None] * size
    # for i in range(size):
    #     A_bit = target_bits[X_name]
    #     B_range = frac[i].asscalar()
    #     B_bit = math.ceil(math.log2(B_range)) + 1 if B_range > 0 else 0
    #     A_sb[i], A_tb[i], B_sb, B_tb, Y_sb[i] = cal_bit(A_bit, B_bit, sb[i].asscalar())
    #     Y_tb[i] = target_bits[name]

    #     tmp = int(round(B_range / (2 ** B_sb)))
    #     clip = (2 ** (B_tb - 1)) - 1 if B_tb > 0 else 0
    #     frac[i] = max(min(tmp, clip), -clip)
    # params[B_name] = frac.reshape(shape)

    B_range = frac.max().asscalar()
    Y_tb = target_bits[name]
    Y_range = 2 ** (Y_tb - 1) - 1
    A_range = Y_range / params[B_name].min().asscalar()
    A_bit = target_bits[X_name]
    B_bit = math.ceil(math.log2(B_range)) + 1
    A_sb, A_tb, B_sb, B_tb, Y_sb = cal_bit(A_bit, B_bit, sb.asscalar())

    X = _realize_func(X, nd.array(A_sb).reshape(shape), params, graph,
            nd.array(A_tb).reshape(shape))
    params[B_name] = (frac / (2 ** B_sb)).round()
    B_range = 2 ** (B_tb - 1) - 1
    params[B_name] = nd.clip(params[B_name],
            a_min=-B_range, a_max=B_range)
    attr = { 'precision': str(B_tb) }
    graph[B_name] = B = mx.sym.var(B_name, shape=shape, attr=attr)
    node = mx.sym.broadcast_mul(X, B)
    node = _realize_func(node, nd.array(Y_sb).reshape(shape), params, graph,
            nd.array(Y_tb).reshape(shape))
    logger.debug("layer %s Y(INT%s >> %s) X(%s|%s >> %s) B(%s|%s vs. %s %s >> %s)",
           name, Y_tb, Y_sb, A_range, A_bit, A_sb, B_range,
           B_bit, frac.max().asscalar(), sb.asscalar(), B_sb)
    target_bits[node.attr('name')] = target_bits[name]
    return node, params
def _realize_parameters(sym, params, graph, inputs_ext,
        target_bits={}, params_sim={}):
    logger = logging.getLogger('log.calib.realize.parameters')
    name = sym.attr('name')
    attr = sym.list_attr()
    if 'precision' not in attr or name in inputs_ext:
        return sym, params
    target_bit = int(attr['precision'])
    data = params[name]
    params[name] = sim.int_realize(data, target_bit, logger=logger)
    # calculate error
    error = params[name].astype('float32') - data
    error_rate = error / data
    rate = nd.norm(error_rate).asscalar() / np.product(data.shape)
    if rate > 0.001:
        logger.warn("realize parameter %-60s avg error=%10.9f shape=%s",
                name, rate, data.shape)
    else:
        logger.debug("realize parameter %-60s avg error=%10.9f shape=%s",
                name, rate, data.shape)
    return sym, params


# interface API
def sym_simulate(symbol, params, inputs_ext, calib_data, ctx):
    logger = logging.getLogger('log.simulate')
    scale_shapes = {}
    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_collect_symbol_ext, scale_shapes=scale_shapes)

    th_dict = {}
    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_calib_sym_collect_thresholds, scale_shapes=scale_shapes,
            th_dict=th_dict, calib_data=calib_data,
            calib_mode='naive', ctx=ctx)

    scale_helper, target_bits = {}, {}
    _, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_collect_scale_helper, th_dict=th_dict,
            scale_shapes=scale_shapes, get_scale=sim.get_sim_scale,
            scale_helper=scale_helper, target_bits=target_bits)
    infer_shapes = sym_infer_shape(symbol, params, inputs_ext)
    symbol, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_annotate_layer, scale_helper=scale_helper,
            target_bits=target_bits, infer_shapes=infer_shapes)
    _, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_annotate_parameters,
            scale_helper=scale_helper, target_bits=target_bits)

    out_scales = [float(scale_helper[s.attr('name')].asscalar()) for s in symbol]

    params = examine_parameters(symbol, params, inputs_ext)
    symbol, params = sym_attach_attrs(symbol, params, inputs_ext,
            precision=target_bits)
    return symbol, params, target_bits, out_scales

def sym_realize(symbol, params, inputs_ext, target_bits, runtime="cvm"):
    logger = logging.getLogger('log.realize')
    _, params = topo_visit(symbol, params, get_op=get_mxnet_op,
           logger=logger, inputs_ext=inputs_ext,
           callback=_realize_parameters)
    symbol, params = topo_visit(symbol, params, get_op=get_mxnet_op,
           logger=logger, inputs_ext=inputs_ext,
           callback=_realize_symbol,
           target_bits=target_bits, runtime=runtime)

    def _check_int_params(params, arg):
       param = params[arg]
       msg = "key:%s max_val:%s, min_val:%s %s"%(arg, param.max().asscalar(),
               param.min().asscalar(), param)
       flat = param.asnumpy().flatten()
       assert all(flat >= INT32_MIN) and all(flat <= INT32_MAX), msg
       assert all(flat.astype('int32').astype(flat.dtype) == flat), msg

    params = examine_parameters(symbol, params, inputs_ext,
          callback=_check_int_params)
    return symbol, params
