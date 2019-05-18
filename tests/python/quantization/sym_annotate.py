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
out_key = 'out_bit'
disable_requant_ops = [
    'Activation', 'relu',
    'Pooling',
    'slice', 'slice_like', 'slice_axis',
    'clip', 'negative',
    'repeat', 'tile', 'expand_dims',
    'Reshape', 'transpose', 'Flatten',
    '_contrib_box_nms',
]

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

def _infer_dynamic_precs(sym, params, graph, inputs_ext, infer_shapes, precs):
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

    # new_childs = []
    for i, c in enumerate(childs):
        c_tb = cprecs[i][name]
        c_bit = cprecs[i][out_key] if out_key in cprecs[i] else cprecs[i][name]
        if c_tb >= c_bit:
            cprecs[i][name] = c_bit
        # if c_tb < c_bit:
        #     tmp = _annotate(c, graph, precs, c_bit, c_tb, logger)
        #     precs[tmp.attr('name')][name] = c_tb
        #     new_childs.append(tmp)
        # else:
        #     cprecs[i][name] = c_bit
        #     new_childs.append(c)
    # node = get_mxnet_op(op_name)(*new_childs, **attr, name=name)

    cbits = [prec[name] for prec in cprecs]
    if op_name in ['sigmoid', 'exp']:
        precs[name][out_key] = 16
    elif op_name in ['Convolution', 'FullyConnected']:
        W_shape = infer_shapes[childs[1].attr('name')]
        sum_len = np.product(W_shape[1:])
        sum_bit = math.ceil(math.log2(sum_len))
        out_bit = cbits[0] + cbits[1] + sum_bit
        if eval(attr['no_bias']) == False:
            cprecs[2][name] = out_bit
            out_bit += 1
        precs[name][out_key] = out_bit
    elif op_name in ['broadcast_add', 'broadcast_sub', 'elemwise_add',
            'elemwise_sub']:
        A, B = childs[0], childs[1]
        if A.attr('op_name') == 'null':
            cbits[0] = cprecs[0][name] = cprecs[1][name]
        elif B.attr('op_name') == 'null':
            cbits[1] = cprecs[1][name] = cprecs[0][name]
        in_bit = max(cbits[0], cbits[1])
        precs[name][out_key] = in_bit + 1
    elif op_name in ['broadcast_mul']:
        precs[name][out_key] = cbits[0] + cbits[1]
    elif op_name in ['Concat']:
        precs[name][out_key] = max(cbits)
    elif op_name in ['sum']:
        axis = eval(attr['axis'])
        dshape = infer_shapes[childs[0].attr('name')]
        sum_len = np.product([dshape[i] for i in axis])
        sum_bit = math.ceil(math.log2(sum_len))
        precs[name][out_key] = cbits[0] + sum_bit

    assert precs[name][out_key] <= PLACE_HOLDER, \
        "%s name=%-40s out of PLACE_HOLDER %s" % (op_name, name, PLACE_HOLDER)
    return sym, params

def _annotate(sym, graph, precs, out_bit, out_tb, logger):
    name, op_name = sym.attr('name'), sym.attr('op_name')
    logger.info("Requantize layer %-20s name=%-40s out of bit %s vs. %s",
        op_name, name, out_tb, out_bit)
    tmp_name = name + '_requant_' + str(out_tb)
    if tmp_name not in graph:
        graph[tmp_name] = mx.sym.Custom(sym, in_prec=out_bit,
                out_prec=out_tb, name=tmp_name, op_type='cvm_annotate')
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
            tmp = _annotate(c, graph, precs, c_bit, c_tb, logger)
            precs[tmp.attr('name')][name] = c_tb
        new_childs.append(tmp)
    node = get_mxnet_op(op_name)(*new_childs, **attr, name=name)

    if 'out_tb' in precs[name]:
        out_tb, out_bit = precs[name]['out_tb'], precs[name][out_key]
        if out_tb < out_bit:
            node = _annotate(node, graph, precs, out_bit, out_tb, logger)
            precs[node.attr('name')][name] = out_tb
    return node, params

def _get_thresholds(out):
    return (out.min().asscalar(), out.max().asscalar())
def _sym_calibrate_th_dict(symbol, params, inputs_ext,
        calib_len=8, ctx=mx.cpu()):
    logger = logging.getLogger("log.sim.calibrate")
    th_dict = {}
    def _run_layer(calib_sym):
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

        print([c.attr('name') for c in calib_sym])
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
        if _is_annotate_op(sym) or \
            op_name in ['broadcast_add', 'broadcast_sub',
                'elemwise_add', 'elemwise_sub']:
            calib_sym.extend(childs)

        if len(calib_sym) > calib_len:
            # remove duplicate
            calib_sym = {s.attr('name'):s for s in calib_sym if s.attr('name') not in graph}
            graph.extend(calib_sym.keys())
            calib_sym = [c for c in calib_sym.values() if c.attr('op_name') != 'null']
            _run_layer(calib_sym)
            calib_sym = []
    _run_layer(calib_sym)
    return th_dict

def _collect_scales(symbol, params, inputs_ext, precs, th_dict):
    def _get_scale(amin, amax, prec):
        alpha = max(abs(amin), abs(amax))
        tb_max = 2 ** (prec - 1) - 1
        return tb_max / alpha
    scales = {}
    args = symbol.list_inputs()
    for k, v in params.items():
        assert k in args, "%s has not in symbol %s"%(k, args)
        assert len(precs[k].keys()) == 1
        n, prec = list(precs[k].items())[0]
        amin, amax = _get_thresholds(params[k])
        scales[k] = _get_scale(amin, amax, prec)

    for k, v in th_dict.items():
        scales[k] = _get_scale(*v, precs[k][out_key])
    return scales


def sym_annotate(symbol, params, inputs_ext, in_bit=8, out_bit=8):
    logger = logging.getLogger('log.infer.precision')
    precs = {}
    topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_infer_fixed_precs, precs=precs)

    for k in inputs_ext:
        precs[k][out_key] = in_bit
        for n, v in precs[k].items():
            assert v >= in_bit, "input %s out of bit %s vs. %s" \
                    % (k, v, in_bit)
            precs[k][n] = in_bit

    infer_shapes = spass.sym_infer_shape(symbol, params, inputs_ext)
    topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_infer_dynamic_precs,
            infer_shapes=infer_shapes, precs=precs)

    for sym in symbol:
        precs[sym.attr('name')]['out_tb'] = out_bit
    symbol, params = topo_visit(symbol, params, inputs_ext,
            get_op=get_mxnet_op, logger=logger,
            callback=_sym_annotate, precs=precs)

    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs = sym_iter(sym.get_children())
        if op_name == 'null':
            continue
        logger.debug("%-20s name=%-40s out_prec=%s in_precs=%s",
                op_name, name, precs[name][out_key],
                [precs[c.attr('name')][name] for c in childs])

    return symbol, params, precs

def sym_simulate(symbol, params, inputs_ext, precs, ctx):
    for k, v in inputs_ext.items():
        assert 'data' in v, "inputs %s has not supply attribute data: %s"%(k, v)

    infer_shapes = spass.sym_infer_shape(symbol, params, inputs_ext)
    print (infer_shapes)
    th_dict = _sym_calibrate_th_dict(symbol, params, inputs_ext, ctx=ctx)
    scales = _collect_parameters_scale(symbol, params, inputs_ext, precs)
    print (th_dict, scales)
    exit()

    return symbol, params














