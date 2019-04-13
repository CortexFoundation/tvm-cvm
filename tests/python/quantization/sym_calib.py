import logging
import os

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

default_target_bit = 8 # INT8
bias_target_bit = default_target_bit * 4 - 1
disable_requant_ops = [
    'Activation', 'Pooling', 'Flatten',
    'slice', 'clip',
]

def _calib_sym_collect_thresholds(sym, params, graph, inputs_ext,
        th_dict, calib_data, ctx=mx.gpu()):
    logger = logging.getLogger('log.calib.sym.collect.thresholds')

    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())

    if op_name == 'null':
        if name in inputs_ext:
            #  output = inputs_ext[name]['thresholds']
            output = calib_data
        else:
            output = params[name]
    elif op_name in disable_requant_ops:
        assert len(childs) == 1
        output = nd.array(th_dict[childs[0].attr('name')])
    else:
        args = sym.list_inputs()
        inputs = [mx.sym.var(n) for n in inputs_ext if n in args]
        graph = SymbolBlock(sym, inputs)
        load_parameters(graph, params, ctx=ctx)
        output = graph.forward(calib_data.as_in_context(ctx))

    min_range = output.min().asscalar()
    max_range = output.max().asscalar()
    if name in th_dict:
        th_dict[name] = (
                min(min_range, th_dict[name][0]),
                max(max_range, th_dict[name][1]))
    else:
        th_dict[name] = (min_range, max_range)

    logger.debug("collect symbol %-40s output min_range=%-20s, max_range=%-20s",
            name, min_range, max_range)

    return sym, params

def _calib_sym_zero_symmetric(sym, params, graph, inputs_ext,
        th_dict, in_zeros, out_zeros):
    logger = logging.getLogger('log.calib.sym.requantize.params')

    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()
    cpu = mx.cpu()

    # calculate input zero symmetric
    if childs is not None:
        childs_name = [c.attr('name') for c in childs]
        in_zeros[name] = [out_zeros[n] for n in childs_name]

    # calculate output zero symmetric
    if op_name == 'null':
        out_zeros[name] = 0
        if name in inputs_ext:
            out_zeros[name] = inputs_ext[name]['zero_point']
            # out_zeros[name] = sim.get_zero_symmetric(th_dict[name])
    elif op_name in ['Pooling', 'Flatten', 'slice']:
        assert len(in_zeros[name]) == 1
        out_zeros[name] = in_zeros[name][0]
    else:
        out_zeros[name] = sim.get_zero_symmetric(th_dict[name])

    return sym, params
def _calib_sym_zero_rewrite(sym, params, graph, inputs_ext,
        in_zeros, out_zeros, infer_shapes, idxs):
    logger = logging.getLogger('log.calib.sym.rewrite')

    name = sym.attr('name')
    op_name = sym.attr('op_name')
    attr = sym.list_attr()
    childs = sym_iter(sym.get_children())

    if op_name == 'null':
        return sym, params

    assert childs is not None
    node = sym
    new_childs = []
    out_z, in_z = out_zeros[name], in_zeros[name]

    index = idxs['index']
    if op_name in ['FullyConnected', 'Convolution']:
        X, W = childs[0], childs[1]
        X_shape = infer_shapes[X.attr('name')]
        W_shape = infer_shapes[W.attr('name')]
        Y_shape = infer_shapes[name]
        # logger.debug("%s out_shape: %s, in_shape: %s, weight_shape: %s",
                # index, infer_shapes[name], X_shape, W_shape)

        bias_name = name + '_offset_bias'
        assert bias_name not in graph

        Y_z, X_z = out_zeros[name], in_zeros[name][0]

        data_shape = [1 if i==0 else s for i,s in enumerate(X_shape)]
        data = nd.full(data_shape, -X_z)
        weight = params[W.attr('name')]
        bias = nd.full((W_shape[0]), Y_z)
        if attr['no_bias'] == 'False':
            bias += params[childs[2].attr('name')]
        params[bias_name] = get_nd_op(op_name)(data, weight, bias, **attr)

        attr['no_bias'] = 'True'
        B = graph[bias_name] = mx.sym.var(bias_name,
                shape=params[bias_name].shape)
        node = get_mxnet_op(op_name)(X, W, **attr, name=name)
        node = mx.sym.broadcast_add(node, B)
    elif op_name in ['broadcast_mul']:
        X, W = childs[0], childs[1]
        assert W.attr('op_name') == 'null'

        X_shape = infer_shapes[X.attr('name')]
        W_shape = infer_shapes[W.attr('name')]
        Y_shape = infer_shapes[name]

        bias_name = name + '_offset_bias'
        assert bias_name not in graph

        Y_z, X_z = out_zeros[name], in_zeros[name][0]

        data_shape = [1 if i==0 else s for i,s in enumerate(X_shape)]
        data = nd.full(data_shape, -X_z)
        weight = params[W.attr('name')]
        params[bias_name] = get_nd_op(op_name)(data, weight, **attr)
        params[bias_name] += Y_z

        if np.any(params[bias_name].asnumpy() != 0):
            B = graph[bias_name] = mx.sym.var(bias_name,
                    shape=params[bias_name].shape)
            node = get_mxnet_op(op_name)(X, W, **attr, name=name)
            node = mx.sym.broadcast_add(node, B)
    elif op_name in ['elemwise_add', 'broadcast_add']:
        X, A = childs[0], childs[1]
        Y_shape = infer_shapes[name]
        B_shape = [1 for _ in Y_shape]

        bias_name = name + '_offset_bias'
        assert bias_name not in graph

        Y_z, X_z = out_zeros[name], in_zeros[name]
        params[bias_name] = nd.array([Y_z - X_z[0] - X_z[1]]).reshape(B_shape)

        B = graph[bias_name] = mx.sym.var(bias_name, shape=B_shape)
        node = get_mxnet_op(op_name)(X, A, **attr, name=name)
        node = mx.sym.broadcast_add(node, B)
    elif op_name in ['Pooling', 'Flatten', 'slice']:
        Y_z, X_z = out_zeros[name], in_zeros[name]
        for x_z in X_z:
            assert x_z == Y_z
    elif op_name == 'sum':
        X = childs[0]
        X_shape = infer_shapes[X.attr('name')]
        bias_name = name + '_offset_bias'
        assert bias_name not in graph

        Y_z, X_z = out_zeros[name], in_zeros[name][0]
        data_shape = [1 if i==0 else s for i,s in enumerate(X_shape)]
        data = nd.full(data_shape, -X_z)
        params[bias_name] = get_nd_op(op_name)(data, **attr)
        params[bias_name] += Y_z

        B = graph[bias_name] = mx.sym.var(bias_name,
                shape=params[bias_name].shape)
        node = get_mxnet_op(op_name)(X, **attr, name=name)
        node = mx.sym.broadcast_add(node, B)
    else:
        logger.info("symbol %-40s processed symmertric by default", name)
        for idx, child in enumerate(childs):
            if in_z[idx] != 0:
                cname = child.attr('name')
                cshape = [1 for _ in infer_shapes[cname]]
                restore_name = cname + '_restore'
                if restore_name in graph:
                    logger.warn("symbol %s has childs(%s)[%s] restore in graph",
                            name, [c.attr('name') for c in childs], idx)
                    restore = graph[restore_name]
                else:
                    restore = mx.sym.var(restore_name, shape=cshape)
                graph[restore_name] = restore

                tmp = mx.sym.broadcast_sub(child, restore)
                new_childs.append(tmp)
                params[restore_name] = nd.array([in_z[idx]]).reshape(cshape)
            else:
                new_childs.append(child)

        node = get_mxnet_op(op_name)(*new_childs, **attr)

        offset_shape = [1 for _ in infer_shapes[name]]
        offset_name = name + '_offset'
        assert offset_name not in graph
        offset = mx.sym.var(offset_name, shape=offset_shape)
        node = mx.sym.broadcast_add(node, offset)
        params[offset_name] = nd.array([out_z]).reshape(offset_shape)

    logger.debug("rewrite symbol %-40s -> %-40s with zeros %-50s -> %s",
            name, node.attr('name'), in_z, out_z)

    infer_shapes[node.attr('name')] = infer_shapes[name]
    idxs['index'] = index + 1
    return node, params
def sym_calib_quantize(symbol, params, inputs_ext, calib_data, ctx):
    logger = logging.getLogger("log.calib.quantize")

    ops = sym_collect_attr(symbol)
    print (ops)

    th_dict, in_zeros, out_zeros= {}, {}, {}
    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_calib_sym_collect_thresholds,
            th_dict=th_dict, calib_data=calib_data, ctx=ctx)

    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_calib_sym_zero_symmetric,
            th_dict=th_dict, in_zeros=in_zeros, out_zeros=out_zeros)

    infer_shapes = sym_infer_shape(symbol, params, inputs_ext)
    indexes = {'index': 0}
    sym, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_calib_sym_zero_rewrite,
            in_zeros=in_zeros, out_zeros=out_zeros,
            infer_shapes=infer_shapes, idxs=indexes)

    return sym, params

def _sim_requantize_op(sym, scale, params, graph):
    name = sym.attr('name')
    scale_name = name + '_requant_scale'
    assert scale_name not in graph
    scale_sym = graph[scale_name] = mx.sym.var(scale_name, shape=(1,))
    params[scale_name] = nd.array([scale])

    requant_op_name = name + '_requant_op'
    assert requant_op_name not in graph
    node = mx.sym.broadcast_mul(sym, scale_sym, name=requant_op_name)
    graph[requant_op_name] = node
    return node
def _is_sim_requantize_op(sym):
    name = sym.attr('name')
    return True if name.endswith('_requant_op') else False
def _realize_sim_requant_op(sym, sb, params, graph):
    """Requantize Op:
        out = round(sym >> sb)  if sb >  0
        out = round(sym)        if sb == 0
        out = round(sym << -sb) if sb <  0

        round(sym >> sb) = int((int(sym >> (sb - 1)) + 1) >> 1)

        out = clip_int8(out)
    """
    name = sym.attr('name')
    sb_name = name + '_shift_bit'
    assert sb_name not in graph
    sb_sym = mx.sym.var(sb_name, shape=(1,))

    if sb == 0:
        out = mx.sym.clip(sym, a_min=-127, a_max=127)
        return out
    elif sb < 0:
        params[sb_name] = nd.array([2 ** (-sb)])
        out = mx.sym.broadcast_mul(sym, sb_sym)
        out = mx.sym.clip(sym, a_min=-127, a_max=127)
        return out

    params[sb_name] = nd.array([2 ** (sb - 1)])
    n1, n2 = "const_var_1", 'const_var_2'
    var1 = graph[n1] if n1 in graph else mx.sym.var(n1, shape=(1,))
    var2 = graph[n2] if n2 in graph else mx.sym.var(n2, shape=(1,))
    graph[n1], graph[n2] = var1, var2
    params[n1], params[n2] = nd.array([1]), nd.array([2])

    out = sym
    if sb > 1:
        out = mx.sym.broadcast_div(out, sb_sym)
        out = mx.sym.floor(out)
    out = mx.sym.broadcast_add(out, var1)
    out = mx.sym.broadcast_div(out, var2)
    out = mx.sym.floor(out)
    out = mx.sym.clip(out, a_min=-127, a_max=127)
    return out

# simlutate and realize graph pass
def _realize_parameters(sym, params, graph, inputs_ext, target_bits, params_sim):
    logger = logging.getLogger('log.calib.realize.parameters')
    name = sym.attr('name')
    if name in params_sim:
        assert sym.attr('op_name') == 'null'
        data = params[name]
        params[name] = sim.int_realize(data, target_bits[name], logger=logger)
        error = params[name] - data
        error_rate = error / data
        rate = nd.norm(error_rate).asscalar() / np.product(data.shape)
        if rate > 0.001:
            logger.warn("realize parameter %-60s average rate=%10.9f shape=%s",
                    name, rate, data.shape)
        else:
            logger.debug("realize parameter %-60s average rate=%10.9f shape=%s",
                    name, rate, data.shape)
    return sym, params

def _sim_scale(sym, params, graph, inputs_ext,
        th_dict, scale_helper, target_bits):
    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()

    # calculate simulate scale
    target_bits[name] = default_target_bit
    scale_helper[name] = sim.get_sim_scale(th_dict[name], default_target_bit)
    if op_name in ['Convolution', 'FullyConnected']:
        if attr['no_bias'] == 'False':
            X_name, W_name = childs[0].attr('name'), childs[1].attr('name')
            B_name = childs[2].attr('name')
            scale_helper[B_name] = scale_helper[X_name] * scale_helper[W_name]
            target_bits[B_name] = bias_target_bit
    return sym, params
def _sim_rewrite(sym, params, graph, inputs_ext, scale_helper, params_sim):
    logger = logging.getLogger('log.calib.sym.sim.rewrite')

    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()

    if op_name == 'null':
        if name in inputs_ext:
            inputs_ext[name]['scale'] = scale_helper[name]
            sim.save_data_scale(name, scale_helper[name], params)
        else:
            params[name] = params[name] * scale_helper[name]
            params_sim.append(name)
        return sym, params
    elif op_name in disable_requant_ops:
        return sym, params

    node = sym
    if op_name in ['Convolution', 'FullyConnected', 'broadcast_mul']:
        X, W = childs[0], childs[1]
        X_name, W_name = X.attr('name'), W.attr('name')
        in_scale = scale_helper[X_name] * scale_helper[W_name]
        out_scale = scale_helper[name]
        requant_scale = out_scale / in_scale
    elif op_name in ['elemwise_add', 'broadcast_add', 'broadcast_sub']:
        A, B = childs[0], childs[1]
        A_name, B_name = A.attr('name'), B.attr('name')

        A_scale, B_scale = scale_helper[A_name], scale_helper[B_name]
        if A_scale > B_scale:
            in_scale, offset = B_scale, B_scale / A_scale
            A = _sim_requantize_op(A, offset, params, graph)
        elif A_scale < B_scale:
            in_scale, offset = A_scale, A_scale / B_scale
            B = _sim_requantize_op(B, offset, params, graph)
        else:
            in_scale = A_scale

        requant_scale = scale_helper[name] / in_scale
        node = get_mxnet_op(op_name)(A, B, **attr, name=name)
    elif op_name in ['sum']:
        X_name = childs[0].attr('name')
        requant_scale = scale_helper[name] / scale_helper[X_name]
    else:
        logger.critical('Unrecognized op:%s(%s)', op_name, name)
        new_childs = []
        for child in childs:
            child_name = child.attr('name')
            scale_name = child_name + '_restore'
            if scale_name in graph:
                scale = graph[scale_name]
            else:
                scale = graph[scale_name] = mx.sym.var(scale_name, shape=(1,))

            restore = scale_helper[child_name]
            params[scale_name] = nd.array([restore])
            tmp = mx.sym.broadcast_div(child, scale)
            new_childs.append(tmp)

        node = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
        requant_scale = scale_helper[name]

    if requant_scale != 1:
        node = _sim_requantize_op(node, requant_scale, params, graph)
        logger.debug("symbol %s requant scale=%s out=%s in=%s",
                name, requant_scale, scale_helper[name],
                [scale_helper[c.attr('name')] for c in childs])

    scale_helper[node.attr('name')] = scale_helper[name]
    return node, params
def _sim_realize_requantize_op(sym, params, graph, inputs_ext):
    logger = logging.getLogger('log.calib.realize')
    childs = sym_iter(sym.get_children())
    node = sym
    if _is_sim_requantize_op(sym):
        A, B = childs[0], childs[1]
        A_name, B_name = A.attr('name'), B.attr('name')

        scale = params[B_name].asscalar()
        frac, sb = sim.extract_float(scale)
        Y_range = 2 ** (default_target_bit - 1) - 1
        A_range = Y_range / scale

        A_sb = math.ceil(math.log2(A_range))
        A_sb = A_sb - (default_target_bit - 1)
        B_sb = math.ceil(math.log2(frac))
        B_sb = B_sb - (default_target_bit - 1)
        Y_sb = - (A_sb + B_sb + sb)

        A = _realize_sim_requant_op(A, A_sb, params, graph)
        params[B_name] = nd.array([round(frac / (2 ** B_sb))])
        node = mx.sym.broadcast_mul(A, B)
        node = _realize_sim_requant_op(node, Y_sb, params, graph)
        logger.debug("layer %-60s Y_range=%-10s>>%-2s X_range=%-20s>>%-2s " +
                "scale=%-20s fraction=%-10s>>%-2s[%-20s] shift bit=%-20s",
                sym.attr('name'), Y_range, Y_sb, A_range, A_sb, scale, frac,
                B_sb, params[B_name].asscalar(), sb)

    return node, params

def _simple_sim_scale(sym, params, graph, inputs_ext,
        th_dict, scale_helper, target_bits):
    logger = logging.getLogger('log.calib.sym.out_shift_bits')
    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()

    # update params bias shift_bits
    if op_name in ['Convolution', 'FullyConnected']:
        if attr['no_bias'] == 'False':
            X_name = childs[0].attr('name')
            W_name = childs[1].attr('name')
            B_name = childs[2].attr('name')
            scale_helper[B_name] = scale_helper[X_name] * scale_helper[W_name]
            target_bits[B_name] = bias_target_bit
    # calculate output shift_bits
    scale_helper[name] = sim.get_simple_sim_scale(th_dict[name],
            default_target_bit)
    target_bits[name] = default_target_bit
    return sym, params
def _simple_sim_realize_requantize_op(sym, params, graph, inputs_ext):
    logger = logging.getLogger('log.calib.realize')
    childs = sym_iter(sym.get_children())
    node = sym
    if _is_sim_requantize_op(sym):
        A, B = childs[0], childs[1]
        A_name, B_name = A.attr('name'), B.attr('name')

        scale = params[B_name].asscalar()
        frac, sb = sim.extract_float(scale)
        assert frac == 1, \
            "extract parameter:%s float:%s fraction:%s shift bit:%s" \
                % (sym.attr('name'), scale, frac, sb)

        sb = - int(sb)
        node = _realize_sim_requant_op(A, sb, params, graph)
        logger.debug("realize requant operator %-60s scale=%-20s fraction=%s shift bit=%s",
                sym.attr('name'), scale, frac, sb)
    return node, params

# interface API
def sym_calib_sim_quant(symbol, params, inputs_ext, calib_data, ctx):
    logger = logging.getLogger('log.simulate')

    th_dict, target_bits, scale_helper, params_sim = {}, {}, {}, []
    # simulate
    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_calib_sym_collect_thresholds,
            th_dict=th_dict, calib_data=calib_data, ctx=ctx)
    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_sim_scale, th_dict=th_dict,
            scale_helper=scale_helper, target_bits=target_bits)
    sym, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_sim_rewrite,
            scale_helper=scale_helper, params_sim=params_sim)

    # realize
    _, params = topo_visit(sym, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_realize_parameters,
            target_bits=target_bits, params_sim=params_sim)
    sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_sim_realize_requantize_op)

    return sym, params, th_dict

def sym_calib_simple_sim_quant(symbol, params, inputs_ext,
        calib_data=None, th_dict={}, ctx=mx.cpu()):
    logger = logging.getLogger("log.calib.sym")
    if not th_dict:
        topo_visit(symbol, params, get_op=get_mxnet_op,
                logger=logger, inputs_ext=inputs_ext,
                callback=_calib_sym_collect_thresholds,
                th_dict=th_dict, calib_data=calib_data, ctx=ctx)

    scale_helper, target_bits, params_sim = {}, {}, []
    # simulate
    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_simple_sim_scale, th_dict=th_dict,
            scale_helper=scale_helper, target_bits=target_bits)
    sym, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_sim_rewrite,
            scale_helper=scale_helper, params_sim=params_sim)

    # realize
    _, params = topo_visit(sym, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_realize_parameters,
            target_bits=target_bits, params_sim=params_sim)
    sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_simple_sim_realize_requantize_op)

    def _check_int_params(params, arg):
       param = params[arg]
       msg = "key:%s value:%s"%(arg, param)
       flat = param.asnumpy().flatten()
       assert all(flat >= INT32_MIN) and all(flat <= INT32_MAX), msg
       assert all(flat.astype('int32').astype(flat.dtype) == flat), msg

    params = examine_parameters(sym, params, inputs_ext,
            allows=['data_scale'], callback=_check_int_params)

    return sym, params
