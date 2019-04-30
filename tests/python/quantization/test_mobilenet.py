
import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn

import tvm
from tvm.contrib import graph_runtime
import nnvm

import sym_calib as calib
import utils
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

def load_fname(version, suffix=None):
    suffix = "."+suffix if suffix is not None else ""
    return "./data/mobilenet%s%s.json"%(version, suffix), \
        "./data/mobilenet%s%s.params"%(version, suffix)

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

    ctx = mx.cpu()
    inputs_ext = { 'data': {
        'shape': (batch_size, 3, 224, 224),
    }}
    inputs = [mx.sym.var(n) for n in inputs_ext]

    data_iter = utils.load_dataset(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data, _ = data_iter_func()

    net1 = utils.load_model(*load_fname("1_0"), inputs, ctx=ctx)
    def graph_func(data):
        return net1.forward(data.as_in_context(ctx))

    net2 = utils.load_model(*load_fname("1.0"), inputs, ctx=ctx)
    def gluon_cv(data):
        return net2.forward(data.as_in_context(ctx))

    # net3 = utils.load_model(*load_fname("1.0_int8"), inputs, ctx=ctx)
    # def gluon_cv_quantize(data):
    #     return net3.forward(data.as_in_context(ctx))

    origin = "1_0"
    sym_fname, param_fname = load_fname(origin)
    sym, params = mx.sym.load(sym_fname), nd.load(param_fname)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    dump_sym, dump_params = load_fname(origin, suffix="sym.sim.prepare")
    nd.save(dump_params, params)
    open(dump_sym, "w").write(sym.tojson())

    qsym, qparams = calib.sym_simulate(sym, params, inputs_ext, data, ctx)
    qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext, "tvm")
    sim.save_ins_ext(qparams, inputs_ext)
    dump_sym, dump_params = load_fname(origin, "sym.quantize")
    nd.save(dump_params, qparams)
    open(dump_sym, "w").write(qsym.tojson())

    cvm_ctx = mx.gpu()
    dump_sym, dump_params = load_fname(origin, "sym.quantize")
    sim.load_ins_ext(nd.load(dump_params), inputs_ext)
    net4 = utils.load_model(dump_sym, dump_params, inputs, ctx=cvm_ctx)
    def cvm_quantize(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        return net4.forward(data.as_in_context(cvm_ctx))

    utils.multi_eval_accuracy(graph_func, data_iter_func,
            gluon_cv, cvm_quantize,
            iter_num=iter_num, logger=logger)

if __name__ == '__main__':
    utils.log_init()

    # zoo.save_mobilenet_v2_1_0()
    # zoo.save_mobilenet1_0()
    # zoo.save_model('mobilenet1.0', 1000)
    # zoo.save_model('mobilenet1.0_int8', 1000)

    # test_sym_pass(16, 10)
    # test_mx_quantize(16, 10)
    test_sym_nnvm(16, 10)
    # test_mx_quantize(16, 10)
