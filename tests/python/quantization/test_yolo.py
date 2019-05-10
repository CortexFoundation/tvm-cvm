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
import sim_quant_helper as sim
import dataset

import logging
import numpy as np

def load_fname(version, suffix=None):
    suffix = "."+suffix if suffix is not None else ""
    return "./data/yolo3%s%s.json"%(version, suffix), \
        "./data/yolo3%s%s.params"%(version, suffix)

def validate(net, val_data, ctx, eval_metric, iter_num, logger=logging):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    # net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    for idx, batch in enumerate(val_data):
        if idx >= iter_num:
            break
        data, label = batch[0], batch[1]
        acc = validate_data(net, data, label, ctx, eval_metric)
        logger.info('Validation: {:5.2%}'.format(acc))

def validate_data(net, data, label, ctx, eval_metric):
    det_ids, det_scores, det_bboxes = [], [], []
    gt_ids, gt_bboxes, gt_difficults = [], [], []

    # get prediction results
    x, y = data, label
    ids, scores, bboxes = net(x.as_in_context(ctx))
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
    bases = []
    def _split(sym, params, graph, inputs):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        node = sym
        if name in ['yolov30_yolodetectionblockv30_leakyrelu5_fwd',
                'yolov30_yolodetectionblockv31_leakyrelu5_fwd',
                'yolov30_yolodetectionblockv32_leakyrelu5_fwd']:
            bases.append(sym)

            node = mx.sym.var(name, shape=infer_shapes[name])
            inputs[name] = {'shape': infer_shapes[name]}
        return node, params

    inputs = {k:v for k,v in inputs_ext.items()}
    top, _ = sutils.topo_visit(symbol, params, inputs,
            get_op=sutils.get_mxnet_op, logger=logger,
            callback=_split)
    top_params = {k:params[k] for k in top.list_inputs() if k in params}
    top_inputs_ext = {k:v for k,v in inputs.items() if k not in inputs_ext}

    base = mx.sym.Group(bases)
    base_params = {k:params[k] for k in base.list_inputs() if k in params}
    base_inputs_ext = inputs_ext

    dump_sym, dump_params = load_fname("_darknet53_voc", "base")
    open(dump_sym, "w").write(base.tojson())
    dump_sym, dump_params = load_fname("_darknet53_voc", "top")
    open(dump_sym, "w").write(top.tojson())

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

    val_data, val_metric = dataset.load_voc(batch_size, input_size)

    sym_file, param_file = load_fname("_darknet53_voc")
    sym, params = mx.sym.load(sym_file), nd.load(param_file)

    base, base_params, base_inputs_ext, top, top_params, top_inputs_ext \
            = split_model(sym, params, inputs_ext, logger)

    base_inputs = [mx.sym.var(n) for n in base_inputs_ext]
    print ("op_name: ", sutils.sym_collect_attr(base))
    base, base_params = spass.sym_quant_prepare(base, base_params, base_inputs_ext)
    base_graph = mx.gluon.nn.SymbolBlock(base, base_inputs)
    utils.load_parameters(base_graph, base_params, ctx=ctx)

    top_inputs = [mx.sym.var(n) for n in top_inputs_ext]
    top_graph = mx.gluon.nn.SymbolBlock(top, top_inputs)
    utils.load_parameters(top_graph, top_params, ctx=ctx)

    def net(data):
        tmp = base_graph(data)
        res = top_graph(*tmp)
        # print (msg)
        return top_graph(*tmp)

    # net1 = utils.load_model(sym_file, param_file, inputs, ctx)
    validate(net, val_data, ctx, val_metric, iter_num)


if __name__ == '__main__':
    utils.log_init()

    # zoo.save_model('yolo3_darknet53_voc')

    test_sym_pass(16, 10)
