import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
from mxnet import gluon

import tvm
from tvm.contrib import graph_runtime
import nnvm

import sym_calib as calib
import utils
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
identity = 'mobilenet' + version
prefix = './data/'
symbol_file, params_file = prefix+identity+'.json', prefix+identity+'.params'
def get_dump_fname(suffix="quant"):
    return './data/%s.%s.json'%(identity, suffix), \
        './data/%s.%s.params'%(identity, suffix)

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    prefix = "./data/mobilenet%s%s"%(version, suffix)
    return utils.extend_fname(prefix, with_ext)

def test_sym_nnvm(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    target = "cuda"
    tvm_ctx = tvm.context(target, 1)
    mx_ctx = mx.gpu(2)
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

    sym_fname, param_fname = load_fname("1_0", "sym.quantize")
    mx_sym, params = mx.sym.load(sym_fname), nd.load(param_fname)
    sim.load_ins_ext(params, inputs_ext)

    graph = nn.SymbolBlock(mx_sym, inputs)
    utils.load_parameters(graph, params, ctx=mx_ctx)
    def graph_func(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        return graph.forward(data.as_in_context(mx_ctx))

    print (sutils.sym_collect_attr(mx_sym))
    nnvm_sym, _ = nnvm.frontend.from_mxnet(mx_sym)
    nnvm_sym, real_params = spass.nnvm_realize(nnvm_sym, params, inputs_ext)
    use_dtype = "int32"
    for key, value in list(real_params.items()):
        real_params[key] = tvm.nd.array(value.asnumpy().astype(use_dtype), tvm_ctx)
    with nnvm.compiler.build_config(opt_level=0): #, add_pass=["PrecomputePrune"]):
        deploy_graph, lib, real_params = nnvm.compiler.build(
            nnvm_sym, target=target, shape=inputs_shape,
            params=real_params, dtype=use_dtype, runtime="tvm")

    dump_sym, dump_params = load_fname("1_0", "tvm.compile")
    dump_lib = "./data/mobilnet.tvm.so"
    open(dump_sym, "w").write(deploy_graph.json())
    print (inputs_ext)
    param_bytes = nnvm.compiler.save_param_dict(real_params)
    open(dump_params, "wb").write(param_bytes)
    lib.export_library(dump_lib)

    module = graph_runtime.create(deploy_graph, lib, tvm_ctx)
    module.load_params(param_bytes)
    def nnvm_int8(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        data = tvm.nd.array(data.asnumpy(), tvm_ctx)
        module.run(data=data.asnumpy())
        return nd.array(module.get_output(0).asnumpy())

    utils.multi_eval_accuracy(graph_func, data_iter_func, nnvm_int8,
            iter_num=iter_num, logger=logger)

def test_sym_pass(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.sym.pass")

    ctx = mx.gpu(2)
    inputs_ext = { 'data': {
            'shape': (batch_size, 3, 224, 224),
        }, }
    inputs = [mx.sym.var(name) for name in inputs_ext]
    logger.info("load dataset, symbol and parameters")
    data_iter = utils.load_dataset(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data, _ = data_iter_func()

    sym, params = mx.sym.load(symbol_file), nd.load(params_file)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    dump_sym, dump_params = get_dump_fname('sym.sim.prepare')
    nd.save(dump_params, params)
    with open(dump_sym, 'w') as fout:
       fout.write(sym.tojson())
    graph_comp = nn.SymbolBlock(sym, inputs)
    utils.load_parameters(graph_comp, params, ctx=ctx)
    def graph_func(data):
        return graph_comp.forward(data.as_in_context(ctx))

    qsym, qparams= calib.sym_simulate(sym,
            params, inputs_ext, data, ctx)
    qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext)
    sim.save_ins_ext(qparams, inputs_ext)
    dump_sym, dump_params = get_dump_fname('sym.sim.pass')
    nd.save(dump_params, qparams)
    with open(dump_sym, 'w') as fout:
       fout.write(qsym.tojson())
    qsym, qparams = mx.sym.load(dump_sym), nd.load(dump_params)
    sim.load_ins_ext(qparams, inputs_ext)
    qgraph = nn.SymbolBlock(qsym, inputs)
    utils.load_parameters(qgraph, qparams, ctx=ctx)
    def simulate(data):
        # data = sim.load_sim_data(data, 'data', inputs_ext)
        data = sim.load_real_data(data, 'data', inputs_ext)
        return qgraph.forward(data.as_in_context(ctx))

    utils.multi_eval_accuracy(graph_func, data_iter_func, simulate,
            iter_num=iter_num, logger=logger)

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
    qsym, qparams, precs, _ = calib.sym_simulate(sym, params, inputs_ext,
            data, calib_ctx)
    qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext,
            precs, "tvm")
    dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    sim.save_ext(dump_ext, inputs_ext)
    nd.save(dump_params, qparams)
    open(dump_sym, "w").write(qsym.tojson())

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

def test_performance(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.tvm.performance")

    target = "llvm"
    tvm_ctx = tvm.context(target, 1)
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

    sym_fname, param_fname = load_fname("1_0")
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

    sym_fname, param_fname = load_fname("1_0", "sym.quantize")
    mx_sym, params = mx.sym.load(sym_fname), nd.load(param_fname)
    sim.load_ins_ext(params, inputs_ext)
    nnvm_sym, _ = nnvm.frontend.from_mxnet(mx_sym)
    nnvm_sym, params = spass.nnvm_realize(nnvm_sym, params, inputs_ext)
    use_dtype = "int32"
    for key, value in list(params.items()):
        params[key] = tvm.nd.array(value.asnumpy().astype(use_dtype), tvm_ctx)
    with nnvm.compiler.build_config(opt_level=opt, runtime="tvm"):
        graph, lib, real_params = nnvm.compiler.build(
            nnvm_sym, target=target, shape=inputs_shape,
            params=params, dtype=use_dtype)
    net2 = graph_runtime.create(graph, lib, tvm_ctx)
    net2.load_params(nnvm.compiler.save_param_dict(real_params))
    def quantize(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        net2.run(data=data.asnumpy())
        return nd.array(net2.get_output(0).asnumpy())

    utils.eval_time_accuracy(graph_func, data_iter_func, quantize,
            iter_num=iter_num, logger=logger)

if __name__ == '__main__':
    utils.log_init()

    # zoo.save_mobilenet_v2_1_0()
    # zoo.save_mobilenet1_0()
    # zoo.save_model('mobilenet1.0', 1000)
    # zoo.save_model('mobilenet1.0_int8', 1000)

    # test_sym_pass(16, 10)
    test_mx_quantize(160, 10)
    # test_sym_nnvm(16, 10)
    # test_performance(16, 10)
