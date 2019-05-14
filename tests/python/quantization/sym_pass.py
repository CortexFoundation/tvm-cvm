import logging
import math
import numpy as np

import mxnet as mx
from mxnet.gluon import nn, SymbolBlock
from mxnet import ndarray as nd
import nnvm as nnvm
import tvm

from sym_utils import *
from utils import *

def fold_cond_op(symbol, params, graph, quant_flag):
    logger = logging.getLogger("log.quant.fold.condition")
    logger.setLevel(quant_flag.log_level)
    logger.info("fold _cond op in graph")
    gh = GraphHelper(graph)
    added_params_name, deleted_params_name = set(), []
    for sym in topo_sort(symbol, logger):
        name = sym.attr('name')
        attr = sym.list_attr()
        op_name = sym.attr('op_name')
        childs = sym_iter(sym.get_children())
        # update inputs layer symbol
        if childs is not None:
            childs = [gh.get_node(childs[idx]) for idx in range(len(childs))]
            # update childs inputs
            op = get_mxnet_op(op_name)
            node = op(*childs, **attr)
        elif op_name != 'null':
            assert False, "Unrecognized op without input"
        else:
            # inputs or params
            node = sym
        if op_name == '_cond':
            logger.debug("Fold condition op:%s(%s)", name,
                    [c.attr('name') for c in childs])
            # cond_func, then_func, else_func = sym.attr('subgraph')
            sb_param_idx, lesser_scalar_idx, others = None, None, []
            for idx, child in enumerate(childs):
                child_op_name = child.attr('op_name')
                if child_op_name == 'null':
                    assert sb_param_idx is None
                    sb_param_idx = idx
                elif child_op_name == '_lesser_scalar':
                    lesser_scalar_idx = idx
                else:
                    others.append(idx)
            shift_bits_sym = childs[sb_param_idx]
            sb_param_name = shift_bits_sym.attr('name')
            assert sb_param_name in params, sb_param_name
            assert len(others) == 2
            # _cond op must be created by same input
            assert childs[others[0]].attr('name') == childs[others[1]].attr('name')
            input_sym = childs[others[0]]
            shift_bits = params[sb_param_name]
            assert shift_bits.shape == (1,)
            if not quant_flag.use_scalar:
                assert "_shift_bits" in sb_param_name
                scale_name = sb_param_name.replace("_shift_bits", "_scale")
                scale_sym = mx.sym.var(scale_name, shape=(1,))
                one_name, two_name = "const_var_one", "const_var_two"
                const_var_one = gh.get_node(one_name,
                        mx.sym.var(one_name, shape=(1,)))
                const_var_two = gh.get_node(two_name,
                        mx.sym.var(two_name, shape=(1,)))
                if shift_bits < 1:
                    scale = 2 ** (-shift_bits)
                    node = mx.sym.broadcast_mul(input_sym, scale_sym)
                else:
                    scale = 2 ** (shift_bits - 1)
                    node = mx.sym.broadcast_div(input_sym, scale_sym)
                    node = mx.sym.broadcast_add(node, const_var_one)
                    node = mx.sym.floor(node)
                    node = mx.sym.broadcast_div(node, const_var_two)
                params[one_name] = mx.ndarray.array([1])
                params[two_name] = mx.ndarray.array([2])
                params[scale_name] = scale
                added_params_name.update([scale_name, one_name, two_name])
            else:
                shift_bits = shift_bits.asnumpy()[0]
                if shift_bits < 1:
                    scale = 2 ** (-shift_bits)
                    node = mx.sym.floor(input_sym * scale)
                else:
                    scale = 2 ** (shift_bits-1)
                    node = mx.sym.floor(input_sym / scale)
                    node = mx.sym.floor((node+1) / 2)
            node = mx.sym.floor(node)
            del params[sb_param_name]
            deleted_params_name.append(sb_param_name)
        graph[name] = node
    logger.debug("[ added_params_name       ]: %s", added_params_name)
    logger.debug("[ deleted_params_name     ]: %s", deleted_params_name)
    nodes = []
    for sym in symbol:
        node = gh.get_node(sym)
        nodes.append(node)
    ret_sym = nodes[0]
    if len(nodes) > 1:
        ret_sym = mx.sym.Group(nodes)
    return ret_sym, params

def yxnet_realize(symbol, params, inputs_ext):
    logger = logging.getLogger("log.quant.nnvm.realize")

    def _realize(sym, params, graph, inputs_ext):
        name = sym.attr('name')
        attr = sym.list_attr()
        op_name = sym.attr('op_name')
        childs = sym_iter(sym.get_children())

        node = sym
        if 'scalar' in attr:
            scalar = float(attr['scalar'])

            msg = "name:%s, op_name:%s, scalar:%s"%(name, op_name, attr)
            assert scalar >= INT32_MIN and scalar <= INT32_MAX, msg
            assert float(int(scalar)) == scalar, msg

            attr['scalar'] = int(scalar)
            node = get_nnvm_op(op_name)(*childs, **attr)

        # remove layer: floor in int8
        if op_name in ['floor', 'ceil', 'fix']:
            node = childs[0]
        elif op_name == '__rpow_scalar__':
            base = int(attr['scalar'])
            if base == 2:
                const_1, const_name = op_const(1, graph, var=nnvm.sym.Variable)
                params[const_name] = nd.array([1])
                node = nnvm.sym.broadcast_left_shift(const_1, childs[0])
        elif op_name not in nnvm_identity_ext:
            logger.critical(
                "Unsupported op:%s(name=%s, attr=%s) in INT8 Inference network",
                op_name, name, attr)
            pass
        return node, params

    ops = sym_collect_attr(symbol)
    print (ops)
    ret_sym, params = topo_visit(symbol, params, get_op=get_nnvm_op,
            logger=logger, inputs_ext=inputs_ext, callback=_realize)
    args = ret_sym.list_input_names()
    ret_params = {}
    for key, value in params.items():
        if key not in args:
            logger.warn("key:%s not exists in graph", key)
            ret_params[key] = value
        else:
            msg = "key:%s value:%s"%(key, value)
            flat = value.asnumpy().flatten()
            assert all(flat >= INT32_MIN) and all(flat <= INT32_MAX), msg
            assert all(flat.astype('int32').astype('float32') == flat), msg
            ret_params[key] = tvm.nd.array(value.astype('int32').asnumpy())
    return ret_sym, ret_params

def nnvm_realize(symbol, params, inputs_ext):
    """Transform Sim-Quant(Float32 Simulate Int8) to Int8-Inference Graph
        Works:
        *) Remove floor|ceil|round layer in Int8 graph
        *) Cast __div_scalar__ op to right shift
        *) Remove unused params in graph
        *) Check&cast params type from Float32 to Int8|Int32
        *) Check supported op in cvm engine
        *) Cast broadcast_div to broadcast_right_shift


    Parameters:
    ===========
    symbol: nnvm.Symbol
    params: mxnet.ndarray.NDArray

    Returns:
    ========
    symbol: nnvm.Symbol
    params: tvm.nd.Array
    """
    logger = logging.getLogger("log.quant.nnvm.realize")

    def _realize(sym, params, graph, inputs_ext):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        attr, childs = sym.list_attr(), sym_iter(sym.get_children())
        node = sym
        if 'scalar' in attr:
            scalar = float(attr['scalar'])

            msg = "name:%s, op_name:%s, scalar:%s"%(name, op_name, attr)
            assert scalar >= INT32_MIN and scalar <= INT32_MAX, msg
            assert float(int(scalar)) == scalar, msg

            attr['scalar'] = int(scalar)
            node = get_nnvm_op(op_name)(*childs, **attr)

        if op_name in ['floor', 'ceil', 'round']:
            node = childs[0]
        elif op_name == '__div_scalar__':
            scalar = int(attr['scalar'])
            sb = math.log2(scalar)
            assert int(sb) == sb, "op(%s name=%s) scalar (%s vs. %s)" \
                % (op_name, name, scalar, sb)

            X = childs[0]
            sb_sym, sb_name = op_const(int(sb), graph, var=nnvm.sym.Variable)
            params[sb_name] = nd.array([int(sb)])
            node = nnvm.sym.broadcast_right_shift(X, sb_sym)
        elif op_name not in nnvm_identity_ext:
            logger.critical(
                "Unsupported op:%s(name=%s, attr=%s) in INT8 Inference network",
                op_name, name, attr)

        return node, params

    print (sym_collect_attr(symbol))
    ret_sym, params = topo_visit(symbol, params, get_op=get_nnvm_op,
            logger=logger, inputs_ext=inputs_ext, callback=_realize)

    args = ret_sym.list_input_names()
    ret_params = {}
    for key, value in params.items():
        if key not in args:
            logger.warn("key:%s not exists in graph", key)
        else:
            msg = "key:%s value:%s"%(key, value)
            flat = value.asnumpy().flatten()
            assert all(flat >= INT32_MIN) and all(flat <= INT32_MAX), msg
            assert all(flat.astype('int32').astype('float32') == flat), msg
            ret_params[key] = tvm.nd.array(value.astype('int32').asnumpy())
    return ret_sym, ret_params

def tvm_params_reduce(symbol, params, inputs_ext, ctx):
    for sym in topo_sort(symbol):
        name, attr = sym.attr('name'), sym.list_attr()
        if sym.attr('op_name') == 'null' and name not in inputs_ext:
            precision = eval(attr['precision'])
            val = params[name]
            if precision > 8:
                params[name] = tvm.nd.array(val.asnumpy().astype('int32'), ctx)
            else:
                params[name] = tvm.nd.array(val.asnumpy().astype('int8'), ctx)
    return params

MATRIX_MAXIMUM_SIZE = 65536 # 2 ** 16
def _matrix_decomposition(sym, params, graph, inputs_ext, infer_shapes):
    logger = logging.getLogger('log.sym.pass.matrix_decomposition')
    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()

    node = sym
    if op_name == 'Convolution':
        # TODO: do matrix decomposition for conv op
        childs_name = [c.attr('name') for c in childs]
        childs_shape = [infer_shapes[n] for n in childs_name]

        for idx, cshape in enumerate(childs_shape):
            cname = childs_name[idx]
            if cname in params and cshape != params[cname].shape:
                logger.critical(
                    "parameter(%s): infer shape(%s) in graph isn't consistent \
                    with params dict(%s)",
                    cshape, params[cname].shape)

        assert 'layout' not in attr or attr['layout'] == 'NCHW'
        # conv input is NCHW format
        data_shape = childs_shape[0] # (batch, channel, height, weight)
        weight_shape = childs_shape[1] # (filter, channel, kernel, kernel)

        channel = data_shape[1] # channel
        kernel = [weight_shape[2], weight_shape[3]] # kernel size
        matrix_len = channel * kernel[0] * kernel[1]
        # print (data_shape, weight_shape, matrix_len)

    elif op_name == 'FullyConnected':
        childs_name = [c.attr('name') for c in childs]
        childs_shape = [infer_shapes[n] for n in childs_name]

        for idx, cshape in enumerate(childs_shape):
            cname = childs_name[idx]
            if cname in params and cshape != params[cname].shape:
                logger.critical(
                    "parameter(%s): infer shape(%s) in graph isn't consistent \
                    with params dict(%s)",
                    cshape, params[cname].shape)

        batch, matrix_len = childs_shape[1]
        if matrix_len > MATRIX_MAXIMUM_SIZE:
            weight_name_prefix = childs[1].attr('name')
            bias = childs[2] if attr['no_bias']=='False' else None

            X, W = childs[0], childs[1]
            if X.attr('op_name') != 'Flatten':
                X = mx.sym.flatten(X)
            weight_params = params[weight_name_prefix]

            nodes = []
            start, step, idx = 0, MATRIX_MAXIMUM_SIZE, 0
            while start < matrix_len:
                stop = min(start + step, matrix_len)

                weight_name = weight_name_prefix + '_split' + str(idx)
                assert weight_name not in graph
                weight = mx.sym.var(weight_name)
                graph[weight_name] = weight

                # TODO: use slice_axis instead of slice
                tmp = mx.sym.slice(X, begin=(0, start), end=(batch, stop))
                tmp = mx.sym.FullyConnected(tmp, weight, bias, **attr)
                nodes.append(tmp)

                params[weight_name] = weight_params.slice(
                        begin=(0, start), end=(batch, stop))
                start, idx = stop, idx+1

            while len(nodes) > 1:
                a, b = nodes.pop(0), nodes.pop(0)
                tmp = a + b
                nodes.append(tmp)
            node = nodes[0]

            logger.info("split %s(%s) with shape (%s, %s -> %s(%s)) array",
                    op_name, name, batch, matrix_len, idx, step)

    return node, params

def sym_infer_shape(symbol, params, inputs_ext):
    logger = logging.getLogger('log.symbol.infer_shape')

    def _infer_shape(sym, params, graph, inputs_ext, infer_shapes):
        logger = logging.getLogger('log.symbol.infer_shape')
        name = sym.attr('name')
        op_name = sym.attr('op_name')
        args = sym.list_inputs()

        if op_name == 'null':
            if name in params:
                assert params[name].shape == infer_shapes[name], \
                        "parameter %s shape %s is inconsistent with \
                        params dict %s"%(name, out_shapes[0], params[name].shape)
            return sym, params

        inputs_shape = {k:v['shape'] for k,v in inputs_ext.items() if k in args}
        _, out_shapes, _ = sym.infer_shape(**inputs_shape)
        assert len(out_shapes) == 1, 'Infer shape %s'%(name)
        if name in infer_shapes:
            logger.warn("Symbol:%s has been infered shape in graph", out_shapes)
            assert infer_shapes[name] == out_shapes[0], "%s shape %s vs. %s" \
                    % (name, infer_shapes[name], out_shapes)

        infer_shapes[name] = out_shapes[0]

        return sym, params

    inputs_shape = {k:v['shape'] for k, v in inputs_ext.items()}
    arg_shapes, _, aux_shapes = symbol.infer_shape(**inputs_shape)
    args, auxs = symbol.list_arguments(), symbol.list_auxiliary_states()
    infer_shapes = {args[i]:arg_shapes[i] for i in range(len(args))}
    infer_shapes.update({auxs[i]:aux_shapes[i] for i in range(len(auxs))})

    _, _ = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_infer_shape, infer_shapes=infer_shapes)

    return infer_shapes

def _sym_check(sym, params, graph, inputs_ext):
    logger = logging.getLogger('log.prepare.symbol.check')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    if op_name not in mx_identity_ext:
        logger.error("%s(%s) has not been considered in quantization",
                name, op_name)
        return sym, params
    attr = sym.list_attr()
    std_attr = mx_identity_ext[op_name]
    for k,v in std_attr.items():
        if k in attr:
            assert attr[k] in v, \
                "%s(%s attr=%s) not match attribute %s (%s vs. %s)" \
                % (name, op_name, attr, k, attr[k], v)
        else:
            assert v[0], "%s(%s attr=%s) not contains attribute %s" \
                % (name, op_name, attr, k)

    if op_name == 'Pooling':
        msg = "%s(%s attr=%s) not match attribute %s (%s vs. %s)"
        if 'pooling_convention' in attr:
            pooling_convention = attr['pooling_convention']
            if pooling_convention == 'full':
                assert 'global_pool' in attr and \
                    attr['global_pool'] == 'True', msg \
                    % (name, op_name, attr, 'pooling_convention&global_pool',
                    [attr['pooling_convention'], attr['global_pool']],
                    ['full', 'True'])
            else:
                assert pooling_convention == 'valid', msg \
                    % (name, op_name, attr, 'pooling_convention',
                    attr['pooling_convention'], 'valid')
    return sym, params

def _sym_rewrite(sym, params, graph, inputs_ext, infer_shapes):
    logger = logging.getLogger('log.prepare.symbol.rewrite')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    node = sym
    if op_name == 'Pooling':
        pool_type = attr['pool_type']
        is_global = attr["global_pool"]
        if pool_type == 'avg' and is_global == 'True':
            input_name = childs[0].attr('name')
            input_shape = infer_shapes[input_name]
            assert len(input_shape) == 4

            scale_name = input_name + '_avg_scale'
            assert scale_name not in graph
            scale_sym = mx.sym.var(scale_name, shape=(1,))
            graph[scale_name] = scale_sym

            params[scale_name] = nd.array([1. /
                    (input_shape[2] * input_shape[3])])

            node = mx.sym.sum(childs[0], axis=(2, 3))
            node = mx.sym.broadcast_mul(node, scale_sym)
        elif pool_type == 'avg':
            X = childs[0]
            X_shape = infer_shapes[X.attr('name')]
            in_channel = X_shape[1]
            conv_attr = {
                'no_bias': 'True',
                'dilate': '(1, 1)',
                'kernel': attr['kernel'],
                'stride': attr['stride'],
                'pad': attr['pad'],
                'layout': 'NCHW',
                'num_filter': in_channel,
                'num_group': in_channel,
            }
            kernel = attr['kernel'][1:-1].split(',')
            kernel = [int(s) for s in kernel]
            conv_name = name.replace('_fwd', '') + '_conv'
            W_name = conv_name + '_weight'
            assert W_name not in graph
            W_shape = (in_channel, 1, kernel[0], kernel[1])
            graph[W_name] = W = mx.sym.var(W_name, shape=W_shape)
            params[W_name] = nd.full(shape=W_shape, val=(1/np.product(kernel)))
            node = mx.sym.Convolution(X, W, **conv_attr, name=conv_name)
        else:
            assert pool_type == 'max', "Unsupported Pooling \
                    %s(%s, pool_type=%s)"%(op_name, name, pool_type)
    elif op_name == 'LeakyReLU':
        act = attr['act_type']
        slope = eval(attr['slope'])
        assert act == 'leaky', "Unsupported LeakyReLU %s for act_type: %s" \
                % (name, act)
        X = childs[0]
        posi_X = mx.sym.relu(X)
        nega_X = mx.sym.negative(X)
        nega_X = mx.sym.relu(nega_X)
        slope_name = name + "_slope"
        params[slope_name] = nd.array([slope])
        graph[slope_name] = slope_sym = mx.sym.var(slope_name, shape=(1,))
        scale_X = mx.sym.broadcast_mul(nega_X, slope_sym)
        node = posi_X - scale_X
    elif op_name == 'BatchNorm':
        # data, gamma, beta, data_mean, data_var
        assert len(childs) == 5
        conv_sym = childs[0]
        gamma = params[childs[1].attr('name')]
        beta = params[childs[2].attr('name')]
        data_mean = params[childs[3].attr('name')]
        data_var = params[childs[4].attr('name')]

        assert conv_sym.attr('op_name') == 'Convolution'
        conv_attr = conv_sym.list_attr()
        conv_childs = sym_iter(conv_sym.get_children())

        epsilon = float(attr['eps']) if 'eps' in attr else 1e-5
        scale = gamma / nd.sqrt(data_var + epsilon)

        weight_name = conv_childs[1].attr('name')
        weight = params[weight_name]
        weight_scale = scale.repeat(np.product(
                    weight.shape[1:])).reshape(weight.shape)
        params[weight_name] = weight * weight_scale

        bias_name = conv_sym.attr('name') + '_bias'
        assert bias_name not in graph, "bias name %s has existed in graph %s" \
            % (name, graph.keys())
        bias = beta - scale * data_mean
        if conv_attr['no_bias'] == 'False':
            bias += params[conv_childs[2].attr('name')]
        params[bias_name] = bias

        conv_name = conv_sym.attr('name')
        suffix = [n for n in name.split("_") if n not in conv_name.split("_")]
        conv_name = "%s_%s" % (conv_name, "_".join(suffix))
        conv_attr['no_bias'] = 'False'
        bias_sym = graph[bias_name] = mx.sym.var(bias_name, shape=bias.shape)
        node = mx.sym.Convolution(conv_childs[0], conv_childs[1],
                bias_sym, **conv_attr, name=conv_name)

        logger.info("fuse Convolution=%-40s and batchnorm=%-40s",
                conv_sym.attr('name'), name)
    elif op_name == 'Dropout':
    # dropout is identity during testing
        node = childs[0]
    elif op_name == '_mul_scalar':
        X = childs[0]
        scalar = eval(attr['scalar'])
        if scalar == 0:
            params[name] = nd.zeros(infer_shapes[name])
            node = mx.sym.var(name, shape=infer_shapes[name])
        else:
            sname = name + '_scalar'
            params[sname] = nd.array([scalar])
            graph[sname] = scale = mx.sym.var(sname, shape=(1,))
            node = mx.sym.broadcast_mul(X, scale, name=name)
    elif op_name == '_div_scalar':
        X = childs[0]
        scalar = eval(attr['scalar'])
        sname = name + '_scalar'
        params[sname] = nd.array([1 / scalar])
        graph[sname] = scale = mx.sym.var(sname, shape=(1,))
        node = mx.sym.broadcast_mul(X, scale, name=name)
    elif op_name == 'slice_like':
        A, B = childs[0], childs[1]
        A_name, B_name = A.attr('name'), B.attr('name')
        axes = eval(attr['axes'])
        A_shape, B_shape = infer_shapes[A_name], infer_shapes[B_name]
        oshape = [None] * len(A_shape)
        begin, end = [None] * len(A_shape), [None] * len(A_shape)
        for ax in axes:
            assert B_shape[ax] <= A_shape[ax]
            begin[ax], end[ax] = 0, B_shape[ax]
        node = mx.sym.slice(A, begin=begin, end=end)
    infer_shapes[node.attr('name')] = infer_shapes[name]
    return node, params

def _fuse_bias(sym, params, graph, inputs_ext, infer_shapes):
    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym_iter(sym.get_children())
    attr = sym.list_attr()

    node = sym
    if op_name in ['FullyConnected', 'Convolution']:
        if attr['no_bias'] == 'False':
            attr['no_bias'] = 'True'

            bias_name = childs[2].attr('name')
            bias = params[bias_name]

            shape = list(infer_shapes[name])
            assert len(bias.shape) == 1
            assert shape [1] == bias.shape[0]
            shape = [1 if i!=1 else s for i,s in enumerate(shape)]

            params[bias_name] = bias.reshape(shape)
            bias_sym = mx.sym.var(bias_name, shape=shape)
            graph[bias_name] = bias_name

            node = get_mxnet_op(op_name)(childs[0], childs[1],
                    **attr, name=name)
            node = mx.sym.broadcast_add(node, bias_sym, name=name+'_add')

    return node, params

def _fuse_constant(sym, params, graph, inputs_ext):
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    node = sym
    if op_name == 'null':
        return node, params
    elif childs is None:
        out = get_nd_op(op_name)(**attr)
        params[name] = out
        node = mx.sym.var(name, shape=out.shape)
    else:
        is_param = lambda c: (c.attr('op_name')=='null') and \
                        (c.attr('name') not in inputs_ext)
        flag = all([is_param(c) for c in childs])
        if flag:
            in_params = [params[c.attr('name')] for c in childs]
            out = get_nd_op(op_name)(*in_params, **attr)
            params[name] = out
            node = mx.sym.var(name, shape=out.shape)
    return node, params

def _reduce_graph(sym, params, graph, inputs_ext):
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    is_param = lambda c: (c.attr('op_name')=='null') and \
                    (c.attr('name') not in inputs_ext)
    # node param1
    #   \   /            reduce      node op(param1, param2)
    #  operator param1   =======>      \   /
    #        \   /                    operator
    #       operator
    node = sym
    struct = [
        ['broadcast_mul'],
        ['broadcast_add', 'broadcast_sub'],
    ]
    if op_name in ['broadcast_mul']:
        A, B = childs[0], childs[1]
        if A.attr('op_name') not in ['broadcast_mul']:
            return node, params
        if not is_param(B):
            return node, params
        A_A, A_B = sym_iter(A.get_children())
        if not is_param(A_B):
            return node, params
        B_name, A_B_name = B.attr('name'), A_B.attr('name')
        if params[B_name].shape != (1,) and params[A_B_name].shape != (1,):
            return node, params
        fuse_name = B_name.split("_")
        fuse_name = "%s_%s"%("_".join(fuse_name),
                "_".join([n for n in A_B_name.split("_") if n not in fuse_name]))
        params[fuse_name] = get_nd_op(op_name)(params[B_name], params[A_B_name])
        fuse_sym = mx.sym.var(fuse_name, shape=params[fuse_name].shape)
        node = get_mxnet_op(op_name)(A_A, fuse_sym, **attr, name=name)
    return node, params

def sym_quant_prepare(symbol, params, inputs_ext):
    logger = logging.getLogger('log.sym.pass.prepare')

    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_sym_check)

    infer_shapes = sym_infer_shape(symbol, params, inputs_ext)
    sym, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_sym_rewrite, infer_shapes=infer_shapes)

    # infer_shapes = sym_infer_shape(sym, params, inputs_ext)
    # sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
    #         logger=logger, inputs_ext=inputs_ext,
    #         callback=_fuse_bias, infer_shapes=infer_shapes)

    infer_shapes = sym_infer_shape(sym, params, inputs_ext)
    sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_matrix_decomposition, infer_shapes=infer_shapes)

    sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_fuse_constant)

    # sym, params = topo_visit(sym, params, get_op=get_mxnet_op,
    #         logger=logger, inputs_ext=inputs_ext,
    #         callback=_reduce_graph)

    params = examine_parameters(sym, params, inputs_ext)
    return sym, params

def sym_attach_attrs(symbol, params, inputs_ext, **kwargs):
    logger = logging.getLogger('log.sym.attach.attrs')
    def _attach_attr(sym, params, graph, inputs_ext, **kwargs):
        name = sym.attr('name')
        op_name = sym.attr('op_name')
        attr = sym.list_attr()
        childs = sym_iter(sym.get_children())
        for k,v in kwargs.items():
            if name not in v:
                continue
            attr[k] = str(v[name])

        if op_name == 'null':
            sym = mx.sym.var(name, attr=attr)
        return sym, params

    return topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_attach_attr, **kwargs)

def sym_dump_layer_outputs(symbol, params, inputs_ext,
        data, allows, datadir, dtype='float64', out_dtype='int32', ctx=mx.gpu()):
    logger = logging.getLogger('log.sym.dump.internals')
    def _run_layer(sym, params, inputs_ext):
        args = sym.list_inputs()
        inputs = [mx.sym.var(n) for n in inputs_ext if n in args]
        graph = SymbolBlock(sym, inputs)
        load_parameters(graph, params, ctx=ctx, dtype=dtype)
        return graph.forward(data.astype(dtype).as_in_context(ctx))
    def _str_output(out, max_num=None):
        out = out.asnumpy().flatten()
        out = out[:max_num] if max_num else out
        dump = ' '.join(str(d) for d in out)
        return dump
    def _str_feature(out):
        a_max = out.max().asscalar()
        a_min = out.min().asscalar()
        return "max: %s, min: %s" % (a_max, a_min)

    in_file, out_file = datadir+'/in.txt', datadir+'/out.txt'
    fin, fout = open(in_file, "w+"), open(out_file, "w+")
    for sym in topo_sort(symbol, logger):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        if op_name not in allows:
            continue
        if name != 'C2_conv2_fwd_C2_batchnorm2_fwd':
            continue

        logger.info("Dump layer %-40s output", name)
        childs = sym_iter(sym.get_children())
        prefix = datadir + '/' + name
        for idx, c in enumerate(childs):
            if c.attr('name') in inputs_ext:
                out = data.astype(out_dtype)
            elif c.attr('op_name') == 'null':
                out = params[c.attr('name')].astype(out_dtype)
            else:
                out = _run_layer(c, params, inputs_ext).astype(out_dtype)
            dump_str = name + '_' + op_name + '_in' + str(idx) + ':\n'
            dump_str += _str_output(out, 100) + '\n'
            fin.write(dump_str)

            if name == 'C2_conv2_fwd_C2_batchnorm2_fwd':
                np.save(datadir+c.attr('name'), out.asnumpy().astype('int32'))

        out = _run_layer(sym, params, inputs_ext).astype(out_dtype)
        dump_str = name + '_' + op_name + '_out:' + _str_feature(out) + '\n'
        dump_str += _str_output(out, 100) + ' ' + '\n'
        fout.write(dump_str)

        if name == 'C2_conv2_fwd_C2_batchnorm2_fwd':
            np.save(datadir+name, out.asnumpy().astype('int32'))

    fin.close()
    fout.close()

def sym_calculate_ops(symbol, params, inputs_ext):
    logger = logging.getLogger("log.calculate.ops")
    ops = {}
    infer_shapes = sym_infer_shape(symbol, params, inputs_ext)
    def _cal_ops(sym, params, graph, inputs_ext):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        msg = "%-20s name=%-40s ops=%-15s oshape=%-20s ishape=%-50s attr=%s"
        cshapes = [infer_shapes[c.attr('name')] for c in childs] if childs else []
        if op_name == 'null':
            return sym, params
        base_ops, ext = 1, "{}"
        if op_name in ['Convolution', 'FullyConnected']:
            W_shape = cshapes[1]
            base_ops = np.product(W_shape[1:]) * 2
            if eval(attr['no_bias']) == False:
                base_ops += 1
        elif op_name in ['Activation']:
            if attr['act_type'] != "relu":
                assert False
        elif op_name in ['Pooling']:
            pool_type = attr['pool_type']
            is_global = eval(attr["global_pool"])
            if is_global:
                _, _, K1, K2 = cshapes[0]
            else:
                K1, K2 = eval(attr['kernel'])
            assert pool_type in ['avg', 'max']
            base_ops = K1 * K2
            if pool_type == 'avg':
                base_ops += 1
            ext = "{'kernel': %s}"%attr['kernel']
        elif op_name in ['Custom']:
            op_type = attr['op_type']
            assert op_type in ['cvm_clip', 'cvm_left_shift', 'cvm_right_shift']
        elif op_name in ['broadcast_mul', 'broadcast_add', 'broadcast_sub', 'Flatten',
            'elemwise_add', 'elemwise_sub', 'relu', 'slice', 'clip', 'negative',
            'slice_like', 'slice_axis', 'repeat', 'tile', 'expand_dims',
            'Reshape', 'transpose', 'Flatten', 'Concat']:
            # base op is 1, do nothing
            pass
        elif op_name in ['sum']:
            axis = eval(attr['axis'])
            base_ops = np.product([cshapes[0][i] for i in axis])
            ext = "{'axis': %s}"%attr['axis']
        else:
            logger.critical("%s(%s) has not been considered", op_name, name)
        count = np.product(infer_shapes[name][1:]) * base_ops
        ops[name] = count
        logger.debug(msg, op_name, name, count,
                infer_shapes[name], cshapes, ext)
        return sym, params

    topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_cal_ops)
    total_ops = 0
    for k,v in ops.items():
        total_ops += v
    logger.info("Graph Total OPs: %s", total_ops)
    top_k = 5
    logger.info("========== Top %d OPs ==========", top_k)
    sorted_ops = sorted(ops.items(), key=lambda item: item[1], reverse=True)
    for i in range(top_k):
        k, v = sorted_ops[i]
        logger.info("{:3d} | name={:40s} ops={:<15d} percent={:6.2%}".format(
                i, k, v, v / total_ops))
    return total_ops






