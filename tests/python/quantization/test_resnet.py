import mxnet as mx
from mxnet import gluon
from mxnet import symbol
from mxnet.contrib import quantization as qm
from mxnet.gluon.model_zoo import vision
from mxnet import ndarray as nd
from mxnet.gluon import HybridBlock, nn
from mxnet.contrib import quantization as mquant

import nnvm
import logging
import os

from quant_op import *
from quant_utils import *
from utils import *
import quant_pass as qpass

# import resnet18 as resnet
# import resnet152 as resnet
import resnet50 as resnet

quant_symbol_file = resnet.SYMBOL_FILE + ".quant"
quant_params_file = resnet.PARAMS_FILE + ".quant"

def load_mxnet_resnet(quant_flag, batch_size=10,
        iter_num=10, need_requant=False):
    logger = logging.getLogger("log.main")

    if not os.path.exists(resnet.SYMBOL_FILE):
        logger.info("save resnet symbol&params")
        resnet.save_graph(mx.gpu())

    inputs = mx.sym.var('data')
    ctx = mx.gpu(1)

    logger.info("load dataset")
    data_iter = load_dataset(batch_size)
    calib_data = data_iter.next()

    logger.info("quantization model")
    if (not need_requant) and os.path.exists(quant_params_file):
        logger.debug("load quant params")
        qparams = nd.load(quant_params_file)
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
        nd.save(quant_params_file, qparams)

    logger.info("load quant/original model")
    qsym_block = resnet.load_quant_graph(quant_flag)
    load_parameters(qsym_block, qparams, prefix="", ctx=ctx)

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

    qsym_block.save_params(quant_params_file)

def test_quant_model(quant_flag):
    graph = resnet.load_quant_graph(quant_flag)
    sym = graph(mx.sym.var('data'))
    with open(quant_symbol_file, 'w') as fout:
        fout.write(sym.tojson())

    ctx = mx.gpu(0)
    inputs = mx.sym.var("data")

    sym = mx.sym.load(quant_symbol_file)

    data_iter = load_dataset(10)
    calib_data = data_iter.next()

    graph = nn.SymbolBlock(sym, [inputs])
    # print ('graph params:', sorted(list(graph.collect_params().keys())))
    # print ('params:', sorted(list(params.keys())))
    # params_dict = load_parameters(graph, params, ctx=ctx)

    graph.load_parameters(quant_params_file, ctx=ctx)

    logger = logging.getLogger("log.main")
    qacc, total = 0, 0
    for i in range(10):
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET)
    formatter = ColoredFormatter(
            fmt="[ %(asctime)s %(name)s.%(levelname)s ] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")

    log_filter = FilterList(
                allows=["log.quant.op.pool", "log.calib", "log.main"],
                disables=["log.quant.op.requant.helper"],
                # keywords=["layer=pool", "calib_pool"],
                log_level=logging.INFO)
    for handler in logging.root.handlers:
        handler.addFilter(log_filter)
        handler.setFormatter(formatter)

    quant_flag = QuantFlag(is_fuse_bn=True, calib_mode=CalibMode.NAIVE,
            log_level=logging.DEBUG,
            disabled_layers=["relu", "pool0", "activation"])

    # load_mxnet_resnet(quant_flag, batch_size=10, iter_num=1,
            # need_requant=False)

    test_quant_model(quant_flag)


