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

import logging
import numpy as np

def get_dump_fname(suffix="quant"):
    return './data/inception_v3.%s.json'%suffix, \
        './data/inception_v3.%s.params'%suffix

def test_sym_nnvm(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    target = "cuda"
    tvm_ctx = tvm.context(target, 1)
    mx_ctx = mx.gpu(2)
    inputs_ext = {
        'data': {
            'shape': (batch_size, 3, 224, 224),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]
    inputs_shape = {k:v['shape'] for k,v in inputs_ext.items()}

    data_iter = utils.load_dataset(batch_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data, _ = data_iter_func()

    load_symbol_fname, load_params_fname = get_dump_fname("sym.sim.simulate")
    sym, params = mx.sym.load(load_symbol_fname), nd.load(load_params_fname)
    sim.load_ins_ext(params, inputs_ext)
    graph = nn.SymbolBlock(sym, inputs)
    utils.load_parameters(graph, params, ctx=mx_ctx)
    def graph_func(data):
        data = sim.load_real_data(data, 'data', inputs_ext)
        np.save("/tmp/inception_v3/data.npy", data.asnumpy().astype('int8'))
        res = graph.forward(data.as_in_context(mx_ctx))
        np.save("/tmp/inception_v3/result.npy", res.asnumpy().astype('int8'))
        return res

    nnvm_sym, _ = nnvm.frontend.from_mxnet(sym)
    nnvm_sym, real_params = spass.nnvm_realize(nnvm_sym, params, inputs_ext)

    nnvm_graph = nnvm.graph.create(nnvm_sym)
    save_symbol_file, _ = get_dump_fname("nnvm.realize")
    with open(save_symbol_file, "w") as fout:
      fout.write(nnvm_graph.json())

    use_dtype = "int32"
    for key, value in list(real_params.items()):
       real_params[key] = tvm.nd.array(value.asnumpy().astype(use_dtype), tvm_ctx)

    with nnvm.compiler.build_config(opt_level=0): #, add_pass=["PrecomputePrune"]):
       deploy_graph, lib, real_params = nnvm.compiler.build(
           nnvm_sym, target=target, shape=inputs_shape,
           params=real_params, dtype=use_dtype)

    dump_symbol, dump_params = '/tmp/inception_v3/symbol.json', '/tmp/inception_v3/params'
    with open(dump_symbol, "w") as fout:
       fout.write(deploy_graph.json())
    print (real_params['A1_pool0_conv_weight'])
    with open(dump_params, "wb") as fout:
       param_bytes = nnvm.compiler.save_param_dict(real_params)
       fout.write(param_bytes)

    # module = graph_runtime.create(deploy_graph, lib, tvm_ctx)
    # module.load_params(param_bytes)
    # def nnvm_real(data):
    #     data = sim.load_real_data(data, 'data', inputs_ext)
    #     data = tvm.nd.array(data.asnumpy(), tvm_ctx)
    #     module.run(data=data.asnumpy())
    #     return nd.array(module.get_output(0).asnumpy())

    # utils.multi_eval_accuracy(graph_func, data_iter_func, # nnvm_real,
    #         iter_num=iter_num, logger=logger)
    res = graph_func(data)
    print (res.asnumpy().flatten()[:10])

def test_sym_pass(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.sym.pass")

    ctx = mx.gpu(2)
    inputs_ext = {
        'data': {
            'shape': (batch_size, 3, 299, 299),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")
    data_iter = utils.load_dataset(batch_size, (3, 299, 299))
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    data, _ = data_iter_func()
    print (data.shape)

    symbol_file, params_file = "./data/inception_v3.json", "./data/inception_v3.params"
    sym, params = mx.sym.load(symbol_file), nd.load(params_file)
    graph_comp = nn.SymbolBlock(sym, inputs)
    utils.load_parameters(graph_comp, params, ctx=ctx)
    def graph_func(data):
        return graph_comp.forward(data.as_in_context(ctx))


    utils.multi_eval_accuracy(graph_func, data_iter_func,
            iter_num=iter_num, logger=logger)
    exit()

    ops = sutils.sym_collect_attr(sym)
    print (ops)

    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    dump_sym, dump_params = get_dump_fname('sym.sim.prepare')
    nd.save(dump_params, params)
    with open(dump_sym, 'w') as fout:
       fout.write(sym.tojson())
    graph = nn.SymbolBlock(sym, inputs)
    utils.load_parameters(graph, params, ctx=ctx)
    def prepare(data):
        return graph.forward(data.as_in_context(ctx))

    ops = sutils.sym_collect_attr(sym)
    print (ops)

    qsym, qparams = calib.sym_simulate(sym, params, inputs_ext, data, ctx)
    qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext)

    sim.save_ins_ext(qparams, inputs_ext)
    dump_sym, dump_params = get_dump_fname('sym.sim.simulate')
    nd.save(dump_params, qparams)
    with open(dump_sym, 'w') as fout:
      fout.write(qsym.tojson())
    qsym, qparams = mx.sym.load(dump_sym), nd.load(dump_params)
    sim.load_ins_ext(qparams, inputs_ext)
    qgraph = nn.SymbolBlock(qsym, inputs)
    utils.load_parameters(qgraph, qparams, ctx=ctx)
    def simulate(data):
        #  data = sim.load_sim_data(data, 'data', inputs_ext)
        data = sim.load_real_data(data, 'data', inputs_ext)
        return qgraph.forward(data.as_in_context(ctx))

    utils.multi_eval_accuracy(graph_func, data_iter_func, prepare, simulate,
            iter_num=iter_num, logger=logger)

def test_mxnet_sym(batch_size=10):
    ctx = mx.gpu(3)
    inputs_ext = {
        'data': {
            'shape': (batch_size, 3, 224, 224),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]
    dtype = 'float64'

    dump_sym, dump_params = get_dump_fname('sym.sim.simulate')
    sym, params = mx.sym.load(dump_sym), nd.load(dump_params)
    args = sym.list_inputs()
    for k, v in params.items():
        if k not in args:
            print ("key: %s not exists in graph"%k)
        else:
            msg = "key:%s value:%s"%(k, v)
            flat = v.asnumpy().flatten()
            assert all(flat >= sutils.INT32_MIN) and all(flat <= sutils.INT32_MAX), msg
            assert all(flat.astype('int32').astype('float32') == flat), msg
            params[k] = v.astype('int32')

    if True:
        mx_sym, mx_params = sym, params
        mx_graph = nn.SymbolBlock(mx_sym, inputs)
        utils.load_parameters(mx_graph, mx_params, ctx=ctx, dtype=dtype)

        data = np.load('/tmp/inception_v3/data.npy')
        data = nd.array(data, ctx=ctx, dtype=dtype)
        res = mx_graph.forward(nd.array(data, ctx=ctx, dtype=dtype))
        print (res.asnumpy().flatten()[:10])
        exit()

    data = np.load('/tmp/inception_v3/data.npy')
    data = nd.array(data, dtype=dtype)

    qsym, qparams = sym, params
    print (sutils.sym_collect_attr(qsym))
    graph = nn.SymbolBlock(qsym, inputs)
    # graph.load_parameters(dump_params, ignore_extra=True, ctx=ctx)
    utils.load_parameters(graph, qparams, ctx=ctx, dtype=dtype)
    res = graph.forward(data.as_in_context(ctx))
    np.save("/tmp/inception_v3/test_result1.npy", res.asnumpy().astype('int32'))
    print (res.asnumpy().flatten()[:10])

if __name__ == '__main__':
    utils.log_init()

    # zoo.save_inception_v3()
    test_sym_pass(16, 100000)
    # test_sym_nnvm(1, 1)
    # test_mxnet_sym(1)
