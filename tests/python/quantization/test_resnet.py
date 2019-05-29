import mxnet as mx
from mxnet import gluon
from mxnet import symbol
from mxnet.contrib import quantization as qm
from mxnet.gluon.model_zoo import vision
from mxnet import ndarray as nd
from mxnet.gluon import HybridBlock, nn
from mxnet.contrib import quantization as mquant

import tvm
from tvm.contrib import graph_runtime
import nnvm

import numpy as np
import logging
import os

from quant_op import *
from quant_utils import *
import utils
import dataset as ds
import sym_utils as sutils
import sym_pass as spass
import sym_annotate as anno
import sym_calib as calib
import sim_quant_helper as sim
import gluon_zoo as zoo

# import resnet18 as resnet
# import resnet152 as resnet
import resnet50 as resnet

from sym_pass import *

def get_dump_fname(suffix="quant"):
    return '%s.%s'%(resnet.SYMBOL_FILE, suffix), \
        '%s.%s'%(resnet.PARAMS_FILE, suffix)

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    fname = "./data/resnet%s%s"%(version, suffix)
    return utils.extend_fname(fname, with_ext)

def test_sym_nnvm(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    target = "cuda"
    tvm_ctx = tvm.context(target, 1)
    mx_ctx = mx.gpu(2)
    inputs_ext = { 'data': {
            'shape': (batch_size, 3, 224, 224),
    } }
    inputs = [mx.sym.var(name) for name in inputs_ext]
    inputs_shape = {k:v['shape'] for k,v in inputs_ext.items()}

    data_iter = load_dataset(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data_iter_func()

    version = "18_v2"
    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    (inputs_ext,) = sim.load_ext(dump_ext)
    sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
    graph = nn.SymbolBlock(sym, inputs)
    load_parameters(graph, params, ctx=mx_ctx)
    # sim.load_ins_ext(params, inputs_ext)
    def graph_func(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        np.save("/tmp/resnet18/data.npy", data.asnumpy().astype('int8'))
        res = graph.forward(data.as_in_context(mx_ctx))
        np.save("/tmp/resnet18/result.npy", res.asnumpy().astype('int8'))
        return res

    nnvm_sym, _ = nnvm.frontend.from_mxnet(sym)
    nnvm_sym, real_params = nnvm_realize(nnvm_sym, params, inputs_ext)

    nnvm_graph = nnvm.graph.create(nnvm_sym)
    save_symbol_file, _ = get_dump_fname("nnvm.realize")
    with open(save_symbol_file, "w") as fout:
       fout.write(nnvm_graph.json())

    use_dtype = "int32"
    for key, value in list(real_params.items()):
        real_params[key] = tvm.nd.array(value.asnumpy().astype(use_dtype), tvm_ctx)

    with nnvm.compiler.build_config(opt_level=0, runtime="cvm"):
        deploy_graph, lib, real_params = nnvm.compiler.build(
            nnvm_sym, target=target, shape=inputs_shape,
            params=real_params, dtype=use_dtype)

    real_params = spass.tvm_params_reduce(nnvm_sym, real_params, inputs_ext, tvm_ctx)

    dump_sym, dump_params = load_fname(version, "nnvm.compile", False)
    with open(dump_sym, "w") as fout:
        fout.write(deploy_graph.json())
    with open(dump_params, "wb") as fout:
        param_bytes = nnvm.compiler.save_param_dict(real_params)
        fout.write(param_bytes)

    # module = graph_runtime.create(deploy_graph, lib, tvm_ctx)
    # module.load_params(param_bytes)
    # def nnvm_real(data):
    #     data = sim.load_real_data(data, 'data', inputs_ext)
    #     data = tvm.nd.array(data.asnumpy(), tvm_ctx)
    #     module.run(data=data.asnumpy())
    #     res = nd.array(module.get_output(0).asnumpy())
    #     return res

    multi_eval_accuracy(graph_func, data_iter_func, # nnvm_real,
            iter_num=iter_num, logger=logger)

def test_sym_pass(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.sym.pass")

    calib_ctx = mx.gpu(2)
    ctx = [mx.gpu(int(i)) for i in "1,2,3,4,5,6,7".split(',') if i.strip()]
    inputs_ext = { 'data': {
            'shape': (batch_size, 3, 224, 224),
    } }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")
    data_iter = ds.load_imagenet_rec(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    for i in range(10):
        if i == 3:
            break
        data, _ = data_iter_func()
    data_iter.reset()

    version = "18_v2"
    net1 = utils.load_model(*load_fname(version), inputs, ctx=ctx)
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    def resnet(data, label):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net1.forward(d) for d in data]
        res = nd.concatenate(res)
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    sym_fname, param_fname = load_fname(version)
    sym, params = mx.sym.load(sym_fname), nd.load(param_fname)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    if False:
        inputs_ext['data']['data'] = data
        in_bit, out_bit = 8, 8
        qsym, qparams, _ = anno.mixed_precision(sym, params, inputs_ext,
                in_bit=in_bit, out_bit=out_bit, ctx=[calib_ctx])
    else:
        qsym, qparams, precs, _ = calib.sym_simulate(sym, params, inputs_ext, data, calib_ctx)
        #dump_sym, dump_params = load_fname(version, "sym.simulate")
        #open(dump_sym, "w").write(qsym.tojson())
        qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext, precs, "cvm")
    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    sim.save_ext(dump_ext, inputs_ext)
    nd.save(dump_params, qparams)
    open(dump_sym, "w").write(qsym.tojson())

    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    (inputs_ext,) = sim.load_ext(dump_ext)
    net3 = utils.load_model(dump_sym, dump_params, inputs, ctx=ctx)
    qacc_top1 = mx.metric.Accuracy()
    qacc_top5 = mx.metric.TopKAccuracy(5)
    qacc_top1.reset()
    qacc_top5.reset()
    def cvm_quantize(data, label):
        data = sim.load_real_data(data, 'data', inputs_ext)
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net3.forward(d) for d in data]
        res = nd.concatenate(res)
        qacc_top1.update(label, res)
        _, top1 = qacc_top1.get()
        qacc_top5.update(label, res)
        _, top5 = qacc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    utils.multi_validate(resnet, data_iter_func,
            cvm_quantize,
            iter_num=iter_num, logger=logger)
    # multi_eval_accuracy(graph_func, data_iter_func,
    #         gluon_cv, cvm_quantize,
    #         iter_num=iter_num, logger=logger)

def test_performance(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.tvm.performance")

    target = "cuda"
    tvm_ctx = tvm.context(target, 1)
    cvm_ctx = tvm.context(target, 2)
    opt = 0
    inputs_ext = { 'data': {
            'shape': (batch_size, 3, 224, 224),
        }, }
    inputs = [mx.sym.var(name) for name in inputs_ext]
    inputs_shape = {k:v['shape'] for k,v in inputs_ext.items()}

    data_iter = utils.load_dataset(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data_iter_func()

    sym_fname, param_fname = load_fname("mxg")
    mx_sym, mx_params = mx.sym.load(sym_fname), nd.load(param_fname)
    nnvm_sym, nnvm_params = nnvm.frontend.from_mxnet(mx_sym, mx_params)
    with nnvm.compiler.build_config(opt_level=opt, runtime="tvm"):
        graph, lib, nnvm_params = nnvm.compiler.build(
            nnvm_sym, target=target, shape=inputs_shape,
            params=nnvm_params)
    net1 = graph_runtime.create(graph, lib, tvm_ctx)
    net1.load_params(nnvm.compiler.save_param_dict(nnvm_params))
    def graph_func(data):
        net1.run(data=data.asnumpy())
        return nd.array(net1.get_output(0).asnumpy())

    sym_fname, param_fname = load_fname("mxg", "sym.quantize")
    mx_sym, params = mx.sym.load(sym_fname), nd.load(param_fname)
    sim.load_ins_ext(params, inputs_ext)
    nnvm_sym, _ = nnvm.frontend.from_mxnet(mx_sym)
    nnvm_sym, real_params = spass.nnvm_realize(nnvm_sym, params, inputs_ext)
    use_dtype = "int32"
    for key, value in list(real_params.items()):
        real_params[key] = tvm.nd.array(value.asnumpy().astype(use_dtype), cvm_ctx)
    with nnvm.compiler.build_config(opt_level=opt, runtime="tvm"):
        graph, lib, real_params = nnvm.compiler.build(
            nnvm_sym, target=target, shape=inputs_shape,
            params=real_params, dtype=use_dtype)
    net2 = graph_runtime.create(graph, lib, cvm_ctx)
    net2.load_params(nnvm.compiler.save_param_dict(real_params))
    def quantize(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        net2.run(data=data.asnumpy())
        return nd.array(net2.get_output(0).asnumpy())

    utils.eval_time_accuracy(graph_func, data_iter_func, quantize,
            iter_num=iter_num, logger=logger)

def save_data():
    batch_size = 1024
    data_iter = load_dataset(batch_size)
    calib_data = data_iter.next()
    x, _ = quant_helper(calib_data.data[0])
    np.save('/tmp/imagenet.x', x.asnumpy())
    np.save('/tmp/imagenet.y', calib_data.label[0].asnumpy())

if __name__ == "__main__":
    utils.log_init()

    # resnet.save_graph(mx.gpu())
    # zoo.save_model('resnet50_v1', 1000)
    # zoo.save_model('resnet18_v1')
    # zoo.save_model('resnet50_v1d_0.86')
    # zoo.save_model('resnet18_v1b_0.89')

    # zoo.save_model('resnet18_v2')
    # exit(-1)

    # save_data()

    # test_nnvm_load(batch_size=16, iter_num=10)
    test_sym_pass(batch_size=32, iter_num=5)
    test_sym_nnvm(batch_size=1, iter_num=0)
    # test_performance(16, 10)


