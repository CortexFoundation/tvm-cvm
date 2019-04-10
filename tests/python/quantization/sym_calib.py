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
        output = calib_data if name in inputs_ext else params[name]
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

def _calib_sym_rewrite(sym, params, graph, inputs_ext,
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
            callback=_calib_sym_rewrite,
            in_zeros=in_zeros, out_zeros=out_zeros,
            infer_shapes=infer_shapes, idxs=indexes)

    return sym, params

def _calib_sym_real_shift_bits(sym, params, graph, inputs_ext,
        in_sbits, out_sbits, target_bits, calib_data, ctx=mx.gpu()):
    logger = logging.getLogger('log.calib.sym.out_shift_bits')

    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()

    cpu = mx.cpu()

    # update params bias shift_bits
    if op_name in ['Convolution', 'FullyConnected']:
        if attr['no_bias'] == 'False':
            assert len(childs) == 3

            data_name = childs[0].attr('name')
            weight_name = childs[1].attr('name')
            bias_name = childs[2].attr('name')

            bias_sb = out_sbits[data_name] + out_sbits[weight_name]
            out_sbits[bias_name] = bias_sb
            target_bits[bias_name] = bias_target_bit

    # calculate input shift_bits
    if childs is not None:
        childs_name = [c.attr('name') for c in childs]
        in_sbits[name] = [out_sbits[n] for n in childs_name]

    # calculate output shift_bits
    if op_name == 'null':
        data = calib_data if name in inputs_ext else params[name]
        _, out_sbits[name] = sim.nd_quant(data.as_in_context(cpu),
                target_bit=default_target_bit, logger=None)
    elif op_name in ['Activation', 'Pooling', 'slice', 'Flatten']:
        assert len(in_sbits[name]) == 1
        out_sbits[name] = in_sbits[name][0]
    else:
        # calculate layer output shift bits
        args = sym.list_inputs()
        inputs = [mx.sym.var(n) for n in inputs_ext if n in args]
        graph = SymbolBlock(sym, inputs)
        load_parameters(graph, params, ctx=ctx)
        qres = graph.forward(calib_data.as_in_context(ctx))
        _, out_sbits[name] = sim.nd_quant(qres.as_in_context(cpu),
                target_bit=default_target_bit, logger=None)
    target_bits[name] = default_target_bit

    logger.debug("calibrate %-40s %-20s in_sbits %-20s -> out_sbits %-20s",
            name, 'Parameter' if op_name == 'null' else op_name,
            [sb.asnumpy()[0] for sb in in_sbits[name]] \
                if name in in_sbits else [],
            out_sbits[name].asnumpy() if name in out_sbits else [])

    return sym, params

def _calib_sym_requant(sym, params, graph, inputs_ext,
        in_sbits, out_sbits, target_bits):
    logger = logging.getLogger('log.calib.sym.requant')

    name = sym.attr('name')
    op_name = sym.attr('op_name')
    attr = sym.list_attr()
    childs = sym.get_children()

    if op_name == 'null':
        # requant params into int with target bits decided before
        if name in out_sbits and name in params:
            assert name in target_bits
            params[name], sb = sim.nd_quant(params[name],
                    shift_bits=out_sbits[name],
                    target_bit=target_bits[name], logger=logger)
            logger.debug("requant params(%s) to int%s with shift bits(%s)",
                    name, target_bits[name]+1, sb.asnumpy())

        return sym, params

    inputs_sb = in_sbits[name]
    out_sb = out_sbits[name]
    requant_sb_name = name + '_requant_shift_bits'
    assert requant_sb_name not in params

    node = sym
    if op_name in ['FullyConnected', 'Convolution', 'broadcast_mul']:
        assert len(inputs_sb) >= 2
        params[requant_sb_name] = out_sb - inputs_sb[0] - inputs_sb[1]
    elif op_name in ['elemwise_add', 'broadcast_add', 'broadcast_sub']:
        assert len(inputs_sb) == 2

        f_name, f_sb = childs[0].attr('name'), inputs_sb[0]
        f_sb_name = f_name + '_plus_shift_bits'
        assert f_sb_name not in graph
        f_sym = mx.sym.var(f_sb_name, shape=(1,))
        graph[f_sb_name] = f_sym

        s_name, s_sb = childs[1].attr('name'), inputs_sb[1]
        s_sb_name = s_name + '_plus_shift_bits'
        assert s_sb_name not in graph
        s_sym = mx.sym.var(s_sb_name, shape=(1,))
        graph[s_sb_name] = s_sym

        assert f_sb.shape == (1,) and s_sb.shape == (1,)
        input_0, input_1 = childs[0], childs[1]
        if any(f_sb > s_sb):
            params[s_sb_name], in_sb = f_sb - s_sb, f_sb
            input_1, params = sim.sym_quant(input_1, params, graph,
                    shift_bits=params[s_sb_name],
                    target_bit=target_bits[input_1.attr('name')])
        else:
            params[f_sb_name], in_sb = s_sb - f_sb, s_sb
            input_0, params = sim.sym_quant(input_0, params, graph,
                    shift_bits=params[f_sb_name],
                    target_bit=target_bits[input_0.attr('name')])

        node = get_mxnet_op(op_name)(input_0, input_1, **attr, name=name)
        params[requant_sb_name] = out_sb - in_sb
    elif op_name in ['sum']:
        assert len(inputs_sb) == 1
        params[requant_sb_name] = out_sb - inputs_sb[0]
    elif op_name in disable_requant_ops:
        pass
    else:
        logger.critical('Unrecognized op:%s(%s)', op_name, name)

    if requant_sb_name in params:
        logger.debug('requant %s(%s) with shift bits: %s',
                op_name, name, params[requant_sb_name].asnumpy())

        requant_sym = mx.sym.var(requant_sb_name, shape=(1,))
        graph[requant_sb_name] = requant_sym
        node, params = sim.sym_quant(node, params, graph,
                shift_bits=params[requant_sb_name],
                target_bit=target_bits[name])

    target_bits[node.attr('name')] = target_bits[name]
    return node, params

def sym_calib_quant(symbol, params, inputs_ext, calib_data, ctx):
    logger = logging.getLogger("log.calib.sym")
    params = examine_parameters(symbol, params, inputs_ext)

    in_sbits, out_sbits, target_bits = {}, {}, {}
    sym, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_calib_sym_real_shift_bits,
            in_sbits=in_sbits, out_sbits=out_sbits,
            target_bits=target_bits,
            calib_data=calib_data, ctx=ctx)
    params = examine_parameters(sym, params, inputs_ext)

    sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_calib_sym_requant,
            in_sbits=in_sbits, out_sbits=out_sbits,
            target_bits=target_bits)

    def _check_int_params(params, arg):
        param = params[arg]
        msg = "key:%s value:%s"%(arg, param)
        flat = param.asnumpy().flatten()
        assert all(flat >= INT32_MIN) and all(flat <= INT32_MAX), msg
        assert all(flat.astype('int32').astype(flat.dtype) == flat), msg

    params = examine_parameters(sym, params, inputs_ext,
            callback=_check_int_params)

    return sym, params, {k:v for k,v in out_sbits.items() if k in inputs_ext}
