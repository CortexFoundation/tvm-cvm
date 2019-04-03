import mxnet as mx
from mxnet.gluon import nn
from mxnet import ndarray as nd
import logging

from quant_utils import *
from sym_pass import *
from utils import *

def data_xform(data):
    """Move channel axis to the beginning, cast to float32, and normalize to [0, 1]."""
    return nd.moveaxis(data, 2, 0).astype('float32') / 255

def load_dataset(batch_size=10):
    data = mx.gluon.data.vision.MNIST(train=False).transform_first(data_xform)
    loader = mx.gluon.data.DataLoader(data, shuffle=False, batch_size=batch_size)

    return iter(loader)

def test_load_simplenet(quant_flag, batch_size=10, iter_num=10):
    logger = logging.getLogger("log.test.main")

    ctx = mx.gpu(2)
    inputs_ext = {
        'data': {
            'shape': (batch_size, 1, 28, 28),
        }
    }
    inputs = [mx.sym.var(name) for name in inputs_ext]

    logger.info("load dataset, symbol and parameters")
    data_iter = load_dataset(batch_size)
    data, label = next(data_iter)

    symbol_file, params_file = "./data/simplenet.json", "./data/simplenet.params"
    sym = mx.sym.load(symbol_file)
    params = nd.load(params_file)

    logger.info("matrix decomposition")
    qsym, qparams = mx_sym_rewrite(sym, params, quant_flag, inputs_ext=inputs_ext)

    logger.info("load model&quant_model")
    mx_graph = nn.SymbolBlock(sym, inputs)
    load_parameters(mx_graph, nd.load(params_file), ctx=ctx)

    qgraph = nn.SymbolBlock(qsym, inputs)
    load_parameters(qgraph, qparams, ctx=ctx)

    logger.info("calculate model accuracy")
    qacc, acc, diff, total = 0, 0, 0, 0
    for i in range(iter_num):
        quant_data, _ = quant_helper(data)

        res = mx_graph.forward(data.as_in_context(ctx))

        if quant_flag.calib_mode == CalibMode.NONE:
            quant_data = data
        qres = qgraph.forward(data.as_in_context(ctx))

        for idx in range(res.shape[0]):
            res_label = res[idx].asnumpy().argmax()
            qres_label = qres[idx].asnumpy().argmax()
            data_label = label[idx].asnumpy()

            diff += 0 if res_label == qres_label else 1
            acc += 1 if res_label == data_label else 0
            qacc += 1 if qres_label == data_label else 0
            total += 1

        try:
            data, label = next(data_iter)
        except:
            exit()

        logger.info("Iteration: %5d | Accuracy: %.2f%% | Quant Acc: %.2f%%" +
                " | Difference: %.2f%% | Total Sample: %5d",
                i, 100.*acc/total, 100.*qacc/total, 100.*diff/total, total)


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
            log_level=logging.DEBUG, use_scalar=False,
            disabled_layers=["relu", "pool0", "activation"])

    test_load_simplenet(quant_flag, batch_size=16, iter_num=10)
