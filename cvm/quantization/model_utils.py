import sys
import logging
from os import path
import numpy as np

import mxnet as mx
from mxnet import gluon, ndarray as nd

import tfm_pass as tpass
import dataset as ds
from transformer import Model, MRT # , init, compile_to_cvm
import sim_quant_helper as sim
import utils
import sutils

def get_mergefunc_yolo(oscales):
    def mergefunc(node, params, graph):
        name, op_name = node.attr('name'), node.attr('op_name')
        childs, attr = sutils.sym_iter(
            node.get_children()), node.list_attr()
        if op_name == '_contrib_box_nms':
            valid_thresh = sutils.get_attr(attr, 'valid_thresh', 0)
            attr['valid_thresh'] = int(valid_thresh * oscales[3])
            node = sutils.get_mxnet_op(op_name)(
                *childs, **attr, name=name)
        return node
    return mergefunc

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
    gt_difficults.append(y.slice_axis(
        axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

    # update metric
    eval_metric.update(det_bboxes, det_ids, det_scores,
                       gt_bboxes, gt_ids, gt_difficults)
    map_name, mean_ap = eval_metric.get()
    acc = {k:v for k, v in zip(map_name, mean_ap)}['mAP']
    return acc

def get_evalfunc_yolo(ctx, **kwargs):
    metric = ds.load_voc_metric()
    metric.reset()

    inputs_qext = kwargs.get('is_quantize', None)
    if inputs_qext is not None:
        qctx = kwargs.get('qctx', ctx)
        model_graph = kwargs.get('model').to_graph(ctx=qctx)
        oscales = kwargs.get('oscales')
        def evalfunc(data, label):
            def net(data):
                data = sim.load_real_data(data, 'data', inputs_qext)
                outs = model_graph(data.as_in_context(qctx))
                outs = [o.as_in_context(ctx) / oscales[i] \
                        for i, o in enumerate(outs)]
                return outs
            acc = validate_data(net, data, label, metric)
            return "{:6.2%}".format(acc)
    else:
        top_graph = kwargs.get('top').to_graph(ctx=ctx)
        base_graph = kwargs.get('base').to_graph(ctx=ctx)
        def evalfunc(data, label):
            def net(data):
                tmp = base_graph(data.as_in_context(ctx))
                outs = top_graph(*tmp)
                return outs
            acc = validate_data(net, data, label, metric)
            return "{:6.2%}".format(acc)
    return evalfunc

def load_model(model, ctx, inputs_qext=None):
    net = model.to_graph(ctx=ctx)

    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    def model_func(data, label):
        data = sim.load_real_data(data, 'data', inputs_qext) \
               if inputs_qext else data
        data = gluon.utils.split_and_load(data, ctx_list=ctx,
                                          batch_axis=0, even_split=False)
        res = [net.forward(d) for d in data]
        res = nd.concatenate(res)
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)
    return model_func

def validate_model(sym_path, prm_path, ctx, num_channel=3,
                   input_size=224, batch_size=16, iter_num=10,
                   ds_name='imagenet', from_scratch=0, lambd=None,
                   dump_model=False, input_shape=None):
    from gluon_zoo import save_model

    flag = [False]*from_scratch + [True]*(2-from_scratch)
    model_name, _ = path.splitext(path.basename(sym_path))
    model_dir = path.dirname(sym_path)
    input_shape = input_shape if input_shape else \
                  (batch_size, num_channel, input_size, input_size)
    logger = logging.getLogger("log.validate.%s"%model_name)

    if not path.exists(sym_path) or not path.exists(prm_path):
        save_model(model_name)
    model = Model.load(sym_path, prm_path)
    model.prepare(input_shape)
    # model = init(model, input_shape)

    print(tpass.collect_op_names(model.symbol, model.params))

    data_iter_func = ds.data_iter(ds_name, batch_size, input_size=input_size)
    data, _ = data_iter_func()

    # prepare
    mrt = model.get_mrt()
    # mrt = MRT(model)

    # calibrate
    mrt.set_data(data)
    prefix = path.join(model_dir, model_name+'.mrt.dict')
    _, _, dump_ext = utils.extend_fname(prefix, True)
    if flag[0]:
        th_dict = mrt.calibrate(lambd=lambd)
        sim.save_ext(dump_ext, th_dict)
    else:
        (th_dict,) = sim.load_ext(dump_ext)
        mrt.set_th_dict(th_dict)

    mrt.set_input_prec(8)
    mrt.set_output_prec(8)

    if flag[1]:
        mrt.quantize()
        mrt.save(model_name+".mrt.quantize", datadir=model_dir)
    else:
        mrt = MRT.load(model_name+".mrt.quantize", datadir=model_dir)

    # dump model
    if dump_model:
        datadir = "/data/ryt"
        model_name = model_name + "_tfm"
        dump_shape = (1, num_channel, input_size, input_size)
        mrt.current_model.to_cvm(
            model_name, datadir=datadir, input_shape=input_shape)
        data = data[0].reshape(dump_shape)
        data = sim.load_real_data(
            data.astype("float64"), 'data', mrt.get_inputs_ext())
        np.save(datadir+"/"+model_name+"/data.npy", data.astype('int8').asnumpy())
        sys.exit(0)

    # validate
    org_model = load_model(Model.load(sym_path, prm_path), ctx)
    cvm_quantize = load_model(
        mrt.current_model, ctx,
        inputs_qext=mrt.get_inputs_ext())

    utils.multi_validate(org_model, data_iter_func, cvm_quantize,
                         iter_num=iter_num,
                         logger=logging.getLogger('mrt.validate'))
    logger.info("test %s finished.", model_name)
