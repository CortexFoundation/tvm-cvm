import logging
import math

import mxnet as mx
import nnvm as nnvm
import tvm

from sym_utils import *

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
        childs = sym.get_children()

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
            logger.debug("Fold condition op:%s", name)
            # cond_func, then_func, else_func = sym.attr('subgraph')
            sb_param_idx, lesser_scalar_idx, others = None, None, []
            for idx, child in enumerate(childs):
                child = childs[idx]
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
            assert sb_param_name in params

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

def nnvm_realize(symbol, params, graph, quant_flag):
    """Transform Sim-Quant(Float32 Simulate Int8) to Int8-Inference Graph
        Works:
        *) Remove floor layer in Int8 graph
        *) Cast _*_scalar op to Int32
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

    def _realize(sym, params, graph, inputs_ext, ops):
        name = sym.attr('name')
        attr = sym.list_attr()
        op_name = sym.attr('op_name')
        childs = sym.get_children()

        node = sym
        if 'scalar' in attr:
            scalar = float(attr['scalar'])

            msg = "name:%s, op_name:%s, scalar:%s"%(name, op_name, attr)
            assert scalar >= INT32_MIN and scalar <= INT32_MAX, msg
            assert float(int(scalar)) == scalar, msg

            attr['scalar'] = int(scalar)
            node = get_nnvm_op(op_name)(*childs, **attr)

        # remove layer: floor in int8
        if op_name == "floor":
            node = childs[0]
        elif op_name == "broadcast_div":
            msg = '%s(op=%s, inputs=%s)'%(name, op_name, [c.attr('name') for c in childs])
            input_sym = childs[0]
            div_sym = childs[1]
            assert div_sym.attr('op_name') == 'null' # params or constant
            div_sym_name = div_sym.attr('name')

            div = params[div_sym_name]
            shift_bits = mx.ndarray.log2(div).astype('float32')
            assert all(div >= 0)
            assert shift_bits.astype('int8').astype('float32') == shift_bits, msg

            sb_sym_name = div_sym_name.replace('_scale', '') + '_shift_bits'
            if sb_sym_name in graph:
                sb_sym = graph[sb_sym_name]
            else:
                sb_sym = nnvm.sym.Variable(sb_sym_name, shape=(1,))
                graph[sb_sym_name] = sb_sym
                params[sb_sym_name] = shift_bits
            node = nnvm.sym.broadcast_right_shift(input_sym, sb_sym)
        elif op_name not in nnvm_identity_ext:
            logger.critical(
                "Unsupported op:%s(name=%s, attr=%s) in INT8 Inference network",
                op_name, name, attr)
            assert False

        return node, params

    ret_sym, params = topo_visit(symbol, params, graph, get_op=get_nnvm_op,
            logger=logger, inputs_ext={'data':{}}, callback=_realize)

    ops = set()
    for sym in topo_sort(ret_sym):
        op_name = sym.attr('op_name')
        ops.add(op_name)
    logger.info("Created graph operators: %s", sorted(ops))

    args = ret_sym.list_input_names()
    ret_params = {}
    for key, value in params.items():
        if key in args:
            msg = "key:%s value:%s"%(key, value)
            flat = value.asnumpy().flatten()
            assert all(flat >= INT32_MIN) and all(flat <= INT32_MAX), msg
            assert all(flat.astype('int32').astype('float32') == flat), msg

            ret_params[key] = tvm.nd.array(value.astype('int32').asnumpy())

    return ret_sym, ret_params

# matrix decomposition
MATRIX_MAXIMUM_SIZE = 65536 # 2 ** 16
def _find_usable_split(n, amax):
    """Find usable split length with integer n.

    Returns split_size, split_number
    """
    assert n > 1

    start = max((n // amax), 2)
    stop = int(math.sqrt(n)) + 1
    for i in range(start, stop):
        if n % i == 0:
            return n/i, i

    return 1, n

def _matrix_decomposition(sym, params, graph, inputs_ext, infer_shapes):
    logger = logging.getLogger('log.quant.op.rewrite.matrix_decomposition')
    name = sym.attr('name')
    op_name = sym.attr('op_name')
    childs = sym.get_children()
    attr = sym.list_attr()

    # Infer internal symbol output shape and save it in infer_shapes.
    inputs_shape = {k:v['shape'] for k, v in inputs_ext.items()}
    if op_name != 'null':
        _, out_shapes, _ = sym.infer_shape(**inputs_shape)
        if name in infer_shapes:
            logger.warn("Symbol:%s has been infered shape in graph", out_shapes)
            assert infer_shapes[name] == out_shapes
        infer_shapes[name] = out_shapes

    node = sym
    if op_name == 'Convolution':
        logger.info('Convolution: %s', name)
        childs = sym_iter(childs)
        sym_ops = [c for c in childs if c.attr('op_name') != 'null']
        sym_params= [c for c in childs if c.attr('op_name') == 'null']

        params_name = [p.attr('name') for p in sym_params]
        params_shape = [params[n].shape for n in params_name if n in params]

    elif op_name == 'FullyConnected':
        logger.info('FullyConnected: %s', name)
        childs = sym_iter(childs)
        childs_name = [c.attr('name') for c in childs]
        childs_shape = [infer_shapes[n] for n in childs_name]

        for idx, cshape in enumerate(childs_shape):
            cname = childs_name[idx]
            if cname in params and cshape != params[cname].shape:
                logger.critical(
                    "parameter(%s): infer shape(%s) in graph isn't consistent \
                    with params dict(%s)",
                    cshape, params[cname])

        if '_weight' not in childs_name[1]:
            logger.warn("'_weight' not in symbol(%s) weight parameter(%s), \
                    may be wrong",
                    childs_name[1])
        if attr['no_bias'] == 'False' and '_bias' not in childs_name[2]:
            logger.warn("'_bias' not in symbol(%s) bias parameter(%s), \
                    may be wrong",
                    childs_name[2])

        weight_shape = childs_shape[1]
        matrix_len = weight_shape[1]
        if matrix_len > MATRIX_MAXIMUM_SIZE:
            weight_name_prefix = childs[1].attr('name')
            bias = childs[2] if attr['no_bias']=='False' else None

            size, number = _find_usable_split(matrix_len, MATRIX_MAXIMUM_SIZE)
            if size == 1:
                logger.error(
                    "cannot find usable split shape for %s in symbol(%s), \
                    split by size 1 eventually",
                    childs_shape, name)

            X, W = childs[0], childs[1]
            out = mx.sym.flatten(X)
            out = mx.sym.split(out, axis=1, num_outputs=number)

            # update params
            weight_params = params[weight_name_prefix]
            weights_split = weight_params.split(axis=1, num_outputs=number)

            node = None
            for idx in range(number):
                weight_name = weight_name_prefix + '_split' + str(idx)
                assert weight_name not in graph
                weight = mx.sym.var(weight_name)

                tmp = mx.sym.FullyConnected(out[idx], weight, bias, **attr)
                node = tmp if node is None else (node + tmp)

                params[weight_name] = weights_split[idx]

            del params[weight_name_prefix]

    return node, params

def mx_sym_rewrite(symbol, params, quant_flag, inputs_ext, graph={}):
    logger = logging.getLogger('log.calib.rewrite.op')
    logger.setLevel(quant_flag.log_level)

    inputs_shape = {k:v['shape'] for k, v in inputs_ext.items()}
    shapes, _, _ = symbol.infer_shape(**inputs_shape)
    args = symbol.list_arguments()
    infer_shapes = {args[i]:shapes[i] for i in range(len(args))}

    sym, params = topo_visit(symbol, params, get_op=get_mxnet_op,
            logger=logger, inputs_ext=inputs_ext,
            callback=_matrix_decomposition, infer_shapes=infer_shapes)

    return sym, params
