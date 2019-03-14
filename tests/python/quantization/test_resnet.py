import logging

import mxnet as mx
from mxnet import gluon
from mxnet import symbol
from mxnet.contrib import quantization as qm
from mxnet.gluon.model_zoo import vision
from mxnet import ndarray as nd
from mxnet.gluon import HybridBlock, nn

import nnvm

from quant_op import *
from quant_utils import *
from utils import *
import quant_pass as qpass
import resnet18 as resnet
# import resnet152 as resnet

logging.addLevelName( logging.WARNING, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.WARNING))
logging.addLevelName( logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

def load_mxnet_resnet():
    # resnet.save_graph(mx.gpu())
    # exit()

    inputs = mx.sym.var('data')
    ctx = mx.gpu(1)

    print ("load dataset")
    data_iter = load_dataset()
    calib_data = data_iter.next()

    print ("quantization model")
    qparams = qpass.fuse_bn_parameters(nd.load(resnet.PARAMS_FILE))

    quant_flag = QuantFlag(is_fuse_bn=True, calib_mode=CalibMode.NONE)
    graph = resnet.load_quant_graph(quant_flag)

    quant_flag = QuantFlag(is_fuse_bn=True, calib_mode=CalibMode.NAIVE,
            log_level=logging.DEBUG, disabled_layers=["relu", "activation"])
    qparams = qpass.calibrate_parameters(graph, qparams, ctx, calib_data, quant_flag)

    print ("load quant/original model")

    qsym_block = nn.HybridSequential(prefix="calib_")
    with qsym_block.name_scope():
        qsym_block.add(resnet.load_quant_graph(quant_flag))
    load_parameters(qsym_block, qparams, prefix="calib_", ctx=ctx)

    sym_block = resnet.load_graph(ctx)
    # print (qsym_block.collect_params().keys())

    print ("calculate model accuracy")
    qacc, acc, diff, total = 0, 0, 0, 0
    for i in range(1):
        image_data = calib_data.data[0]
        qimage_data, _ = quant_helper(image_data)

        res = sym_block.forward(image_data.as_in_context(ctx))
        # comp_res, _ = quant_helper(res)
        # print (res.max().asnumpy(), comp_res[0].max().asnumpy(), comp_res[0].min().asnumpy())

        qres = qsym_block.forward(qimage_data.as_in_context(ctx))
        # print (qres[0].max(), qres[0].min())

        assert res.shape == qres.shape
        for idx in range(res.shape[0]):
            res_label = res[idx].asnumpy().argmax()
            qres_label = qres[idx].asnumpy().argmax()
            image_label = calib_data.label[0][idx].asnumpy()

            # print ("result", res_label, qres_label, image_label)

            diff += 0 if res_label == qres_label else 1
            acc += 1 if res_label == image_label else 0
            qacc += 1 if qres_label == image_label else 0
            total += 1

        calib_data = data_iter.next()

        print ("Accurracy: ", 1. * acc / total,
                "Quant acc: ", 1. * qacc / total,
                "Different: ", 1. * diff / total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
            format="[ %(asctime)s %(name)s.%(levelname)s ] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("log")

    log_filter = FilterList(
                allows=["log.quant.op", "log.calib.requant"],
                disables=["log.quant.op.requant.helper"])
    for handler in logging.root.handlers:
            handler.addFilter(log_filter)

    load_mxnet_resnet()
