import mxnet as mx
from mxnet import gluon
from mxnet import symbol
from mxnet.contrib import quantization as qm
from mxnet.gluon.model_zoo import vision
from mxnet import ndarray as nd
from mxnet.gluon import HybridBlock, nn
from mxnet.contrib import quantization as mquant

import tvm
from tvm.contrib import graph_runtime, util
import nnvm

import numpy as np
import logging
import os

from quant_op import *
from quant_utils import *
from utils import *
import quant_pass as qpass

# import resnet18 as resnet
# import resnet152 as resnet
import resnet50 as resnet

from sym_pass import *

def get_dump_fname(suffix="quant"):
    return '%s.%s'%(resnet.SYMBOL_FILE, suffix), \
        '%s.%s'%(resnet.PARAMS_FILE, suffix)

def mxnet_realize(quant_flag):
    logger = logging.getLogger("log.quant.main.mxnet")

    load_symbol_file, load_params_file = get_dump_fname("gluon.quant")

    inputs = mx.sym.var('data')
    ctx = mx.gpu(1)

    mxnet_symbol = mx.sym.load(load_symbol_file)
    params = nd.load(load_params_file)

    #  sym, params = quant_realize(mxnet_symbol, params, {}, quant_flag)

    save_symbol_file, save_params_file = get_dump_fname("post.quant")
    nd.save(save_params_file, params)
    print (params.keys())
    with open(save_symbol_file, 'w') as fout:
        fout.write(sym.tojson())

def gluon_quant_resnet(quant_flag, batch_size=10,
        iter_num=10, need_requant=False):
    logger = logging.getLogger("log.quant.main.gluon")
    logger.info("=== Model Quantazation ===")

    pass_name = "gluon.quant"
    quant_symbol_file, quant_params_file = get_dump_fname(pass_name)

    if not os.path.exists(resnet.SYMBOL_FILE):
        logger.info("save resnet symbol&params")
        resnet.save_graph(mx.gpu())

    inputs = mx.sym.var('data')
    ctx = mx.gpu(1)

    logger.info("load dataset")
    data_iter = load_dataset(batch_size)
    calib_data = data_iter.next()

    logger.info("quantization model")
    tmp_params_file = quant_params_file + ".tmp"
    if (not need_requant) and os.path.exists(tmp_params_file):
        logger.debug("load quant params")
        qparams = nd.load(tmp_params_file)
    else:
        qparams = qpass.fuse_bn_parameters(nd.load(resnet.PARAMS_FILE), quant_flag)
        name_scope = "calib_"
        scope_graph = nn.HybridSequential(prefix=name_scope)
        with scope_graph.name_scope():
            graph = resnet.load_quant_graph(QuantFlag(is_fuse_bn=True,
                        calib_mode=CalibMode.NONE))
            scope_graph.add(graph)
        qparams = qpass.calibrate_parameters(scope_graph, qparams, ctx,
                calib_data, quant_flag, name_scope=name_scope)
        nd.save(tmp_params_file, qparams)

    graph = resnet.load_quant_graph(quant_flag)
    sym, qparams = graph(inputs), load_parameters(graph, qparams, ctx=ctx)
    sym, qparams = fold_cond(sym, qparams, {}, quant_flag)

    nd.save(quant_params_file, qparams)
    with open(quant_symbol_file, 'w') as fout:
        fout.write(sym.tojson())

    logger.info("load quant/original model")
    qsym_block = nn.SymbolBlock(sym, [inputs])
    qsym_block.load_parameters(quant_params_file, ctx=ctx, ignore_extra=True)

    sym_block = resnet.load_graph(ctx)

    logger.info("calculate model accuracy")
    qacc, acc, diff, total = 0, 0, 0, 0
    for i in range(iter_num):
        image_data = calib_data.data[0]
        qimage_data, _ = quant_helper(image_data)

        res = sym_block.forward(image_data.as_in_context(ctx))

        if quant_flag.calib_mode == CalibMode.NONE:
            qimage_data = image_data
        qres = qsym_block.forward(qimage_data.as_in_context(ctx))

        assert res.shape == qres.shape
        for idx in range(res.shape[0]):
            res_label = res[idx].asnumpy().argmax()
            qres_label = qres[idx].asnumpy().argmax()
            image_label = calib_data.label[0][idx].asnumpy()

            diff += 0 if res_label == qres_label else 1
            acc += 1 if res_label == image_label else 0
            qacc += 1 if qres_label == image_label else 0
            total += 1

        try:
            calib_data = data_iter.next()
        except:
            exit()

        logger.info("Iteration: %5d | Accuracy: %.2f%% | Quant Acc: %.2f%%" +
                " | Difference: %.2f%% | Total Sample: %5d",
                i, 100.*acc/total, 100.*qacc/total, 100.*diff/total, total)

def test_quant_model(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.mxnet")
    logger.info("=== Log Test Mxnet ===")

    load_symbol_file, load_params_file = get_dump_fname("post.quant")

    ctx = mx.gpu(1)
    inputs = mx.sym.var("data")

    sym = mx.sym.load(load_symbol_file)

    data_iter = load_dataset(batch_size)
    calib_data = data_iter.next()

    graph = nn.SymbolBlock(sym, [inputs])
    # print ('graph params:', sorted(list(graph.collect_params().keys())))
    # print ('params:', sorted(list(params.keys())))
    # params_dict = load_parameters(graph, params, ctx=ctx)

    graph.load_parameters(load_params_file, ctx=ctx)

    qacc, total = 0, 0
    for i in range(iter_num):
        qimage_data, _ = quant_helper(calib_data.data[0])

        # params['data'] = qimage_data
        # graph = sym.bind(ctx, params)
        qres = graph.forward(qimage_data.as_in_context(ctx))

        for idx in range(qres.shape[0]):
            qres_label = qres[idx].asnumpy().argmax()
            image_label = calib_data.label[0][idx].asnumpy()

            qacc += 1 if qres_label == image_label else 0
            total += 1

        try:
            calib_data = data_iter.next()
        except:
            exit()

        logger.info("Iteration: %5d | Quant Acc: %.2f%% | Total Sample: %5d",
                i, 100.*qacc/total, total)

def test_nnvm_load(batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.nnvm")
    logger.info("=== Log Test NNVM ===")

    load_symbol_fname, load_params_fname = get_dump_fname("gluon.quant")

    in_shape = (batch_size, 3, 224, 224)
    data_iter = load_dataset(batch_size)
    calib_data = data_iter.next()

    params = nd.load(load_params_fname)

    sym = mx.sym.load(load_symbol_fname)
    nnvm_sym, _ = nnvm.frontend.from_mxnet(sym)

    nnvm_sym, params = quant_realize(nnvm_sym, params, {}, quant_flag)

    nnvm_graph = nnvm.graph.create(nnvm_sym)
    save_symbol_file, _ = get_dump_fname("nnvm.realize")
    with open(save_symbol_file, "w") as fout:
       fout.write(nnvm_graph.ir())

    use_dtype = "int32"
    for key, value in list(params.items()):
        params[key] = tvm.nd.array(value.asnumpy().astype(use_dtype))
    with nnvm.compiler.build_config(opt_level=0): #, add_pass=["PrecomputePrune"]):
        deploy_graph, lib, params = nnvm.compiler.build(
            nnvm_graph, target="cuda", shape={"data": in_shape},
            params=params, dtype=use_dtype)

        with open("deploy.log", "w") as fout:
            fout.write(deploy_graph.ir())

    module = graph_runtime.create(deploy_graph, lib, tvm.gpu(1))
    param_bytes = nnvm.compiler.save_param_dict(params)
    module.load_params(param_bytes)
    out_shape = (1000,)
    qacc, total = 0, 0
    for i in range(iter_num):
        qimage_data, _ = quant_helper(calib_data.data[0])

        module.run(data=qimage_data.asnumpy())
        qres = module.get_output(0).asnumpy()

        for idx in range(qres.shape[0]):
            qres_label = qres[idx].argmax()
            image_label = calib_data.label[0][idx].asnumpy()

            qacc += 1 if qres_label == image_label else 0
            total += 1

        try:
            calib_data = data_iter.next()
        except:
            exit()

        logger.info("Iteration: %5d | Quant Acc: %.2f%% | Total Sample: %5d",
                i, 100.*qacc/total, total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
    formatter = ColoredFormatter(
            fmt="[ %(asctime)s %(name)s.%(levelname)s ] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")

    allows=["log.quant", "log.calib", "log.main", "log.test"]
    disables = ["log.quant.op.requant.helper", "autotvm"]

    log_filter = FilterList(
                allows=allows, disables=disables,
                # keywords=["layer=pool", "calib_pool"],
                log_level=logging.INFO,
                default=False)
    for handler in logging.root.handlers:
        handler.addFilter(log_filter)
        handler.setFormatter(formatter)

    quant_flag = QuantFlag(is_fuse_bn=True, calib_mode=CalibMode.NAIVE,
            log_level=logging.DEBUG,
            disabled_layers=["relu", "pool0", "activation"])

    #  gluon_quant_resnet(quant_flag, batch_size=10, iter_num=10,
            #  need_requant=False)

    # mxnet_realize(quant_flag)
    test_nnvm_load(batch_size=10, iter_num=10)


