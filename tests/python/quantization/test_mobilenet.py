import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
from mxnet import gluon

import tvm
from tvm.contrib import graph_runtime
import nnvm

import sym_calib as calib
import utils
import mrt as _mrt
import dataset as ds
import gluon_zoo as zoo
import sym_pass as spass
import sym_utils as sutils
import sim_quant_helper as sim
import cvm_op as cvm

import logging
import numpy as np

version = '1_0'
# version = '_v2_1_0'
def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    prefix = "./data/mobilenet%s%s"%(version, suffix)
    return utils.extend_fname(prefix, with_ext)

def test_sym_nnvm():
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
    (inputs_ext,) = sim.load_ext(dump_ext)
    for k, v in inputs_ext.items():
        v['shape'] = (1, *v['shape'][1:])

    dump_sym, dump_params = load_fname(version, "nnvm.compile")
    spass.mxnet_to_nnvm(sym, params, inputs_ext, dump_sym, dump_params)

def test_mx_quantize(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.mx.quantize")

    ctx = [mx.gpu(int(i)) for i in "1,2,3,4,5,6,7".split(',') if i.strip()]
    inputs_ext = { 'data': {
        'shape': (batch_size, 3, 224, 224),
    }}
    inputs = [mx.sym.var(n) for n in inputs_ext]

    data_iter = ds.load_imagenet_rec(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data, _ = data_iter_func()

    version = "1_0"
    net1 = utils.load_model(*load_fname(version), inputs, ctx=ctx)
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    def mobilenet(data, label):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net1.forward(d) for d in data]
        res = nd.concatenate(res)
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    calib_ctx = mx.gpu(1)
    sym_fname, param_fname = load_fname(version)
    sym, params = mx.sym.load(sym_fname), nd.load(param_fname)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    if True:
        mrt = _mrt.MRT(sym, params, inputs_ext)
        mrt.set_data('data', data)
        mrt.calibrate()
        mrt.set_output_prec(8)
        qsym, qparams, inputs_ext = mrt.quantize()
    else:
        inputs_ext['data']['data'] = data
        th_dict = calib.sym_calibrate(sym, params, inputs_ext, ctx=calib_ctx)
        qsym, qparams, precs, _ = calib.sym_simulate(sym, params, inputs_ext, th_dict)
        qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext, precs)
    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    sim.save_ext(dump_ext, inputs_ext)
    nd.save(dump_params, qparams)
    open(dump_sym, "w").write(qsym.tojson())

    dump_sym, dump_params = load_fname(version, "nnvm.compile")
    spass.mxnet_to_nnvm(qsym, qparams, inputs_ext, dump_sym, dump_params)

    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    (inputs_ext,) = sim.load_ext(dump_ext)
    net2 = utils.load_model(dump_sym, dump_params, inputs, ctx=ctx)
    qacc_top1 = mx.metric.Accuracy()
    qacc_top5 = mx.metric.TopKAccuracy(5)
    qacc_top1.reset()
    qacc_top5.reset()
    def cvm_quantize(data, label):
        data = sim.load_real_data(data, 'data', inputs_ext)
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net2.forward(d) for d in data]
        res = nd.concatenate(res)
        qacc_top1.update(label, res)
        _, top1 = qacc_top1.get()
        qacc_top5.update(label, res)
        _, top5 = qacc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    utils.multi_validate(mobilenet, data_iter_func,
            cvm_quantize,
            iter_num=iter_num, logger=logger)
    # utils.multi_eval_accuracy(mobilenet, data_iter_func,
    #         cvm_quantize,
    #         iter_num=iter_num, logger=logger)

if __name__ == '__main__':
    utils.log_init()

    # zoo.save_mobilenet_v2_1_0()
    # zoo.save_mobilenet1_0()
    # zoo.save_model('mobilenet1.0', 1000)
    # zoo.save_model('mobilenet1.0_int8', 1000)

    test_mx_quantize(16, 1000)
    # test_sym_nnvm()
    # test_sym_nnvm(16, 10)
    # test_performance(16, 10)
