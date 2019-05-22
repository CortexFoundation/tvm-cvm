import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
from mxnet import gluon

import tvm
from tvm.contrib import graph_runtime
import nnvm

import sym_calib as calib
import utils
import gluon_zoo as zoo
import sym_pass as spass
import sym_utils as sutils
import sym_annotate as anno
import sim_quant_helper as sim
import dataset

import logging
import numpy as np

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    prefix = "./data/yolo3%s%s"%(version, suffix)
    return utils.extend_fname(prefix, with_ext)

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

def split_model(symbol, params, inputs_ext, keys, logger=logging):
    infer_shapes = spass.sym_infer_shape(symbol, params, inputs_ext)
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

    base_ctx = mx.gpu(7)
    ctx = mx.gpu(7)
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

    net1 = mx.gluon.nn.SymbolBlock(sym, inputs)
    utils.load_parameters(net1, params, ctx=ctx)
    metric = dataset.load_voc_metric()
    metric.reset()
    def yolov3(data, label):
       def net(data):
           return net1(data.as_in_context(ctx))
       acc = validate_data(net, data, label, metric)
       return "{:6.2%}".format(acc)

    keys = [
        'yolov30_yolooutputv30_conv0_fwd',
        'yolov30_yolooutputv31_conv0_fwd',
        'yolov30_yolooutputv32_conv0_fwd',
    ]
    base, base_params, base_inputs_ext, top, top_params, top_inputs_ext \
            = split_model(sym, params, inputs_ext, keys, logger)
    dump_sym, dump_params = load_fname("_darknet53_voc", "base")
    open(dump_sym, "w").write(base.tojson())
    dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "top", True)
    open(dump_sym, "w").write(top.tojson())
    nd.save(dump_params, top_params)
    sim.save_ext(dump_ext, top_inputs_ext)

    base_inputs = [mx.sym.var(n) for n in base_inputs_ext]
    base_graph = mx.gluon.nn.SymbolBlock(base, base_inputs)
    utils.load_parameters(base_graph, base_params, ctx=base_ctx)

    top_inputs = [mx.sym.var(n) for n in top_inputs_ext]
    top_graph = mx.gluon.nn.SymbolBlock(top, top_inputs)
    utils.load_parameters(top_graph, top_params, ctx=ctx)

    # base_metric = dataset.load_voc_metric()
    # base_metric.reset()
    # def yolov3(data, label):
    #     def net(data):
    #         tmp = base_graph(data.as_in_context(base_ctx))
    #         tmp = [t.as_in_context(ctx) for t in tmp]
    #         return top_graph(*tmp)
    #     acc = validate_data(net, data, label, base_metric)
    #     return "{:6.2%}".format(acc)

    # quantize base graph
    # in_bit, out_bit = 8, 8
    # base_inputs_ext['data']['data'] = data
    # qsym, qparams, type_ext = anno.mixed_precision(base, base_params,
    #         base_inputs_ext, in_bit=in_bit, out_bit=out_bit,
    #         calib_group=12, ctx=[mx.gpu(7)])
    # dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "base.quantize", True)
    # open(dump_sym, "w").write(qsym.tojson())
    # sim.save_ext(dump_ext, base_inputs_ext, type_ext)
    # nd.save(dump_params, qparams)

    dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "base.quantize", True)
    net2_inputs_ext, btype_ext = sim.load_ext(dump_ext)
    base_oscales = [
        btype_ext['yolov30_yolooutputv30_conv0_fwd'],
        btype_ext['yolov30_yolooutputv31_conv0_fwd'],
        btype_ext['yolov30_yolooutputv32_conv0_fwd'],
    ]
    net2_inputs = [mx.sym.var(n) for n in net2_inputs_ext]
    net2 = utils.load_model(dump_sym, dump_params, net2_inputs, ctx=ctx)
    base_metric = dataset.load_voc_metric()
    base_metric.reset()
    def base_quantize(data, label):
        def net(data):
            data = sim.load_real_data(data, 'data', net2_inputs_ext)
            tmp = list(net2(data.as_in_context(ctx)))
            tmp = [t / base_oscales[i] for i,t in enumerate(tmp)]
            return top_graph(*tmp)
        acc = validate_data(net, data, label, base_metric)
        return "{:6.2%}".format(acc)

    # quantize top graph
    # top_data = base_graph(data.as_in_context(base_ctx))
    # top_sym = base_graph(mx.sym.Group(base_inputs))
    # top_names = [c.attr('name') for c in top_sym]
    # for idx, n in enumerate(top_names):
    #     top_inputs_ext[n]['data'] = top_data[idx]
    #     print (n, top_data[idx].abs().max().asscalar())
    # in_bit, out_bit = 8, 30
    # outputs_ext = {
    #     'yolov30_yolooutputv30_expand_dims0': { 'thresholds': (0, 1), 'type': 'score' },
    #     'yolov30_yolooutputv31_expand_dims0': { 'thresholds': (0, 1), 'type': 'score' },
    #     'yolov30_yolooutputv32_expand_dims0': { 'thresholds': (0, 1), 'type': 'score' },
    #     'yolov30_yolooutputv30_tile0': { 'thresholds': (0, 416), 'type': 'bbox' },
    #     'yolov30_yolooutputv31_tile0': { 'thresholds': (0, 416), 'type': 'bbox' },
    #     'yolov30_yolooutputv32_tile0': { 'thresholds': (0, 416), 'type': 'bbox' },
    #     'yolov30_yolooutputv30_broadcast_add1': { 'fixed': True, 'type': 'ids' },
    #     'yolov30_yolooutputv31_broadcast_add1': { 'fixed': True, 'type': 'ids' },
    #     'yolov30_yolooutputv32_broadcast_add1': { 'fixed': True, 'type': 'ids' },
    # }
    # qsym, qparams, type_ext = anno.mixed_precision(top, top_params,
    #         top_inputs_ext, in_bit=in_bit, out_bit=out_bit,
    #         out_ext=outputs_ext, calib_group=12, ctx=[mx.gpu(7)])
    # dump_sym, dump_params, dump_ext = load_fname("_darknet53_voc", "top.quantize", True)
    # open(dump_sym, "w").write(qsym.tojson())
    # sim.save_ext(dump_ext, top_inputs_ext, type_ext)
    # nd.save(dump_params, qparams)

    sym_file, param_file, ext_file = load_fname("_darknet53_voc", "top.quantize", True)
    net3_inputs_ext, type_ext = sim.load_ext(ext_file)
    out_scales = [type_ext['ids'], type_ext['score'], type_ext['bbox']]
    top_sym = base_graph(mx.sym.Group(base_inputs))
    top_names = [c.attr('name') for c in top_sym]
    net3_inputs = [mx.sym.var(n) for n in net3_inputs_ext]
    net3 = utils.load_model(sym_file, param_file, net3_inputs, ctx=ctx)
    top_qmetric = dataset.load_voc_metric()
    top_qmetric.reset()
    def top_quantize(data, label):
        def net(data):
            # data = sim.load_real_data(data, 'data', net3_inputs_ext)
            # out = net3(data.as_in_context(ctx))
            # out = [(t / out_scales[i]) for i,t in enumerate(out)]
            # return out

            tmp = base_graph(data.as_in_context(base_ctx))
            tmp = [t.as_in_context(ctx) for t in tmp]
            tmp = [sim.load_real_data(tmp[i], n, top_inputs_ext) for i,n in enumerate(top_names)]
            out = net3(*tmp)
            out = [(t / out_scales[i]) for i,t in enumerate(out)]
            return out
        acc = validate_data(net, data, label, top_qmetric)
        return "{:6.2%}".format(acc)



    utils.multi_validate(yolov3, data_iter_func, base_quantize, # cvm_quantize,
            iter_num=iter_num, logger=logger)

if __name__ == '__main__':
    utils.log_init()

    # zoo.save_model('yolo3_darknet53_voc')

    test_sym_pass(16, 10)
