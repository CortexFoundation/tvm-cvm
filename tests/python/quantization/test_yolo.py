import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
from mxnet import gluon

import tvm
from tvm.contrib import graph_runtime
import nnvm

import sym_calib as calib
import sym_quant as cvmq
import utils
import gluon_zoo as zoo
import sym_pass as spass
import sym_utils as sutils
import sim_quant_helper as sim
import dataset

import logging
import numpy as np

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    if with_ext:
        return "./data/yolo3%s%s.json"%(version, suffix), \
            "./data/yolo3%s%s.params"%(version, suffix), \
            "./data/yolo3%s%s.ext"%(version, suffix)
    else:
        return "./data/yolo3%s%s.json"%(version, suffix), \
            "./data/yolo3%s%s.params"%(version, suffix)

def validate(net, val_data, eval_metric, iter_num, logger=logging):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    # net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    for idx, batch in enumerate(val_data):
        if idx >= iter_num:
            break
        data, label = batch[0], batch[1]
        acc = validate_data(net, data, label, eval_metric)
        logger.info('Validation: {:5.2%}'.format(acc))

def validate_data(net, data, label, eval_metric):
    det_ids, det_scores, det_bboxes = [], [], []
    gt_ids, gt_bboxes, gt_difficults = [], [], []

    # get prediction results
    x, y = data, label
    ids, scores, bboxes = net(x)
    det_ids.append(ids)
    det_scores.append(scores)
    # clip to image size
    det_bboxes.append(bboxes.clip(0, x.shape[2]))
    # split ground truths
    gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
    gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
    gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

    # update metric
    eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    map_name, mean_ap = eval_metric.get()
    acc = {k:v for k,v in zip(map_name, mean_ap)}['mAP']
    return acc

def split_model(symbol, params, inputs_ext, logger=logging):
    infer_shapes = spass.sym_infer_shape(symbol, params, inputs_ext)
    keys = [
        # 'yolov30_yolodetectionblockv30_leakyrelu5_fwd',
        # 'yolov30_yolodetectionblockv31_leakyrelu5_fwd',
        # 'yolov30_yolodetectionblockv32_leakyrelu5_fwd',
        'yolov30_yolooutputv30_conv0_fwd',
        'yolov30_yolooutputv31_conv0_fwd',
        'yolov30_yolooutputv32_conv0_fwd',
    ]
    bases = [s for s in sutils.topo_sort(symbol) if s.attr('name') in keys]
    base = mx.sym.Group(bases)
    base_params = {k:params[k] for k in base.list_inputs() if k in params}
    base_inputs_ext = inputs_ext

    graph = {}
    inputs = {k:v for k,v in inputs_ext.items()}
    for sym in sutils.topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sutils.sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [graph[c.attr('name')] for c in childs]
            node = sutils.get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in keys:
            node = mx.sym.var(name)
            inputs[name] = {'shape': infer_shapes[name]}
        graph[name] = node
    nodes = [graph[sym.attr('name')] for sym in symbol]
    top = nodes[0] if len(nodes) == 1 else mx.sym.Group(nodes)
    top_params = {k:params[k] for k in top.list_inputs() if k in params}
    top_inputs_ext = {k:v for k,v in inputs.items() if k not in inputs_ext}

    return base, base_params, base_inputs_ext, top, top_params, top_inputs_ext

def test_sym_pass(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.sym.pass")

    ctx = mx.gpu(2)
    input_size = 416
    h, w = input_size, input_size
    inputs_ext = {
        'data': {
            'shape': (batch_size, 3, h, w),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    val_data = dataset.load_voc(batch_size, input_size)
    val_data_iter = iter(val_data)
    def data_iter_func():
        data, label = next(val_data_iter)
        return data, label
    data, _ = data_iter_func()

    sym_file, param_file = load_fname("_darknet53_voc")
    sym, params = mx.sym.load(sym_file), nd.load(param_file)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    pre_sym, pre_param = load_fname("_darknet53_voc", "prepare")
    open(pre_sym, "w").write(sym.tojson())

    # graph = mx.gluon.nn.SymbolBlock(sym, inputs)
    # utils.load_parameters(graph, params, ctx=ctx)
    # def net(data):
    #    return graph(data.as_in_context(ctx))
    # validate(net, val_data, dataset.load_voc_metric(), iter_num)
    # exit()

    base, base_params, base_inputs_ext, top, top_params, top_inputs_ext \
            = split_model(sym, params, inputs_ext, logger)
    dump_sym, dump_params = load_fname("_darknet53_voc", "base")
    open(dump_sym, "w").write(base.tojson())
    dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "top", True)
    open(dump_sym, "w").write(top.tojson())
    nd.save(dump_params, top_params)
    sim.save_ext(dump_ext, top_inputs_ext)

    top_inputs = [mx.sym.var(n) for n in top_inputs_ext]
    top, top_params = cvmq.sym_simulate(top, top_params, top_inputs_ext, data)
    top_graph = mx.gluon.nn.SymbolBlock(top, top_inputs)
    utils.load_parameters(top_graph, top_params, ctx=ctx)

    base_ctx = mx.gpu(3)
    base_inputs = [mx.sym.var(n) for n in base_inputs_ext]
    print ("op_name: ", sutils.sym_collect_attr(base))
    base_graph = mx.gluon.nn.SymbolBlock(base, base_inputs)
    utils.load_parameters(base_graph, base_params, ctx=base_ctx)
    base_metric = dataset.load_voc_metric()
    base_metric.reset()
    def base_func(data, label):
        def net(data):
            tmp = base_graph(data.as_in_context(base_ctx))
            tmp = [t.as_in_context(ctx) for t in tmp]
            return top_graph(*tmp)
        return validate_data(net, data, label, base_metric)

    # quantize top graph
    top_data = base_graph(data.as_in_context(base_ctx))
    for idx, c in enumerate(base_graph(mx.sym.Group(base_inputs))):
        top_inputs_ext[c.attr('name')]['data'] = top_data[idx]
    qsym, qparams, precs, out_scales = calib.sym_simulate(top, top_params,
            top_inputs_ext, None, ctx)
    top_qgraph = mx.gluon.nn.SymbolBlock(qsym, top_inputs)
    utils.load_parameters(top_qgraph, qparams, ctx=ctx)
    top_qmetric = dataset.load_voc_metric()
    top_qmetric.reset()
    def top_quantize(data, label):
        def net(data):
            tmp = base_graph(data.as_in_context(base_ctx))
            tmp = [t.as_in_context(ctx) for t in tmp]
            out = top_qgraph(*tmp)
            out = [t / out_scales[i] for i,t in enumerate(out)]
            return out
        return validate_data(net, data, label, top_qmetric)

    # quantize base graph
    # qsym, qparams, precs, out_scales = calib.sym_simulate(base, base_params,
    #         base_inputs_ext, data, ctx)
    # dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "simulate", True)
    # sim.save_ext(dump_ext, base_inputs_ext, precs, out_scales)
    # nd.save(dump_params, qparams)
    # open(dump_sym, "w").write(qsym.tojson())

    # qsym, qparams = calib.sym_realize(qsym, qparams,
    #        base_inputs_ext, precs, "cvm")
    # dump_sym, dump_params = load_fname("_darknet53_voc", "quantize")
    # nd.save(dump_params, qparams)
    # open(dump_sym, "w").write(qsym.tojson())

    # qctx = mx.gpu(4)
    # _, _, dump_ext = load_fname("_darknet53_voc", "simulate", True)
    # base_inputs_ext, precs, out_scales = sim.load_ext(dump_ext)
    # dump_sym, dump_params = load_fname("_darknet53_voc", "quantize")
    # qsym, qparams = mx.sym.load(dump_sym), nd.load(dump_params)
    # qgraph = mx.gluon.nn.SymbolBlock(qsym, base_inputs)
    # utils.load_parameters(qgraph, qparams, ctx=qctx)
    # qmetric = dataset.load_voc_metric()
    # qmetric.reset()
    # def cvm_quantize(data, label):
    #     def net(data):
    #         data = sim.load_real_data(data, 'data', base_inputs_ext)
    #         # data = sim.load_sim_data(data, 'data', base_inputs_ext)
    #         tmp = list(qgraph(data.as_in_context(qctx)))
    #         tmp = [t.as_in_context(ctx) / out_scales[i] for i,t in enumerate(tmp)]
    #         return top_graph(*tmp)
    #     return validate_data(net, data, label, qmetric)


    utils.multi_validate(base_func, data_iter_func, top_quantize, # cvm_quantize,
            iter_num=iter_num, logger=logger)

if __name__ == '__main__':
    utils.log_init()

    # zoo.save_model('yolo3_darknet53_voc')

    test_sym_pass(16, 10)
