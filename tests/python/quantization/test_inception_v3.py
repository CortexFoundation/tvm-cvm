import mxnet as mx
from mxnet import ndarray as nd
from mxnet import gluon
from mxnet.gluon import nn

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

import logging
import numpy as np

def get_dump_fname(suffix="quant"):
    return './data/inception_v3.%s.json'%suffix, \
        './data/inception_v3.%s.params'%suffix

def load_fname(version, suffix=None, with_ext=False):
    suffix = "."+suffix if suffix is not None else ""
    fname = "./data/inception%s%s"%(version, suffix)
    return utils.extend_fname(fname, with_ext)

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

    with nnvm.compiler.build_config(opt_level=0, runtime="cvm"): #, add_pass=["PrecomputePrune"]):
       deploy_graph, lib, real_params = nnvm.compiler.build(
           nnvm_sym, target=target, shape=inputs_shape,
           params=real_params, dtype=use_dtype)

    real_params = spass.tvm_params_reduce(nnvm_sym, real_params, inputs_ext, tvm_ctx)

    dump_sym, dump_params = load_fname("", "nnvm.compile", False)
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
    #     return nd.array(module.get_output(0).asnumpy())

    # utils.multi_eval_accuracy(graph_func, data_iter_func, # nnvm_real,
    #         iter_num=iter_num, logger=logger)
    res = graph_func(data)
    print (res.asnumpy().flatten()[:10])

def test_sym_pass(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.sym.pass")

    calib_ctx = mx.gpu(2)
    ctx = [mx.gpu(int(i)) for i in "1,2,3,4,5,6,7".split(',') if i.strip()]
    input_size = 299
    version = "v3"
    h, w = input_size, input_size
    inputs_ext = {
        'data': {
            'shape': (batch_size, 3, h, w),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")
    data_iter = ds.load_imagenet_rec(batch_size, input_size)
    # data_iter = utils.load_dataset(batch_size, input_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]

    net1 = utils.load_model(*load_fname(version), inputs, ctx=ctx)
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    def inception_v3(data, label):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net1.forward(d) for d in data]
        res = nd.concatenate(res)
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    # sym_file, param_file = load_fname(version)
    # sym, params = mx.sym.load(sym_file), nd.load(param_file)
    # sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)
    # dump_sym, _ = load_fname(version, 'sym.prepare')
    # open(dump_sym, "w").write(sym.tojson())

    # data, _ = data_iter_func()
    # qsym, qparams, precs, _ = calib.sym_simulate(sym, params, inputs_ext, data, calib_ctx)
    # dump_sym, dump_params, dump_ext = load_fname(version, "sym.simulate", True)
    # sim.save_ext(dump_ext, inputs_ext, precs)
    # nd.save(dump_params, qparams)
    # open(dump_sym, "w").write(qsym.tojson())
    # qsym, qparams = calib.sym_realize(qsym, qparams, inputs_ext, precs)
    # dump_sym, dump_params, dump_ext = load_fname(version, "sym.quantize", True)
    # sim.save_ext(dump_ext, inputs_ext)
    # nd.save(dump_params, qparams)
    # open(dump_sym, "w").write(qsym.tojson())

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

    utils.multi_validate(inception_v3, data_iter_func,
            cvm_quantize,
            iter_num=iter_num, logger=logger)
    # utils.multi_eval_accuracy(graph_func, data_iter_func,
    #         cvm_quantize,
    #         iter_num=iter_num, logger=logger)

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

def validate(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.mx.quantize")

    ctx = [mx.gpu(int(i)) for i in "1,2,3,4,5,6,7".split(',') if i.strip()]
    input_size = 299
    h, w = input_size, input_size
    inputs_ext = { 'data': {
        'shape': (batch_size, 3, h, w),
    }}
    inputs = [mx.sym.var(n) for n in inputs_ext]

    data_iter = ds.load_imagenet_rec(batch_size, input_size)
    # data_iter = utils.load_dataset(batch_size, input_size)
    def data_iter_func():
        data = data_iter.next()
        return data.data[0], data.label[0]
    #  data, _ = data_iter_func()

    net1 = utils.load_model(*load_fname("_v3"), inputs, ctx=ctx)
    def graph_func(data):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net1.forward(d) for d in data]
        return nd.concatenate(res)

    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    net2 = utils.load_model(*load_fname("v3"), inputs, ctx=ctx)
    def gluon_cv(data, label):
        data = gluon.utils.split_and_load(data, ctx_list=ctx, batch_axis=0, even_split=False)
        res = [net2.forward(d) for d in data]
        res = nd.concatenate(res)
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)

    utils.multi_validate(gluon_cv, data_iter_func,
           iter_num=iter_num, logger=logger)
    # utils.multi_eval_accuracy(graph_func, data_iter_func,
    #        gluon_cv,
    #        iter_num=iter_num, logger=logger)

if __name__ == '__main__':
    utils.log_init()

    # zoo.save_inception_v3()
    # zoo.save_model('inceptionv3', 1000)

    # test_sym_pass(600, 100000)
    test_sym_nnvm(1, 0)
    # test_mxnet_sym(1)
    # validate(700, 100000)
