import logging
import os

import mxnet as mx
from mxnet import gluon
from mxnet import symbol
from mxnet.contrib import quantization as qm
from mxnet.gluon.model_zoo import vision
from mxnet import ndarray as nd
from mxnet.gluon import HybridBlock, nn
from mxnet.contrib import quantization as mquant

import nnvm

from quant_op import *
from quant_utils import *
from utils import *
import quant_pass as qpass
import resnet18 as resnet
# import mx_quant as mquant
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
    # out = graph(inputs)
    # sym, mx_params = mx_quant(out, qparams, data_iter, ctx)
    # print (out.list_outputs(), sym.list_outputs())

    quant_flag = QuantFlag(is_fuse_bn=True, calib_mode=CalibMode.NAIVE,
            log_level=logging.DEBUG, disabled_layers=["relu", "activation"])

    qparams = qpass.fuse_bn_parameters(nd.load(resnet.PARAMS_FILE), quant_flag)

    graph = resnet.load_quant_graph(QuantFlag(is_fuse_bn=True,
                calib_mode=CalibMode.NONE))

    qparams = qpass.calibrate_parameters(graph, qparams, ctx, calib_data, quant_flag)

    print ("load quant/original model")

    qsym_block = nn.HybridSequential(prefix="calib_")
    with qsym_block.name_scope():
        qsym_block.add(resnet.load_quant_graph(quant_flag))
    load_parameters(qsym_block, qparams, prefix="calib_", ctx=ctx)
    # qsym_block = nn.SymbolBlock(sym, [inputs, mx.sym.var("softmax_label")])
    # load_parameters(qsym_block, mx_params, ctx=ctx)
    # mod = mx.mod.Module(sym, context=ctx)
    # batch_size = 10
    # input_shape = (batch_size, 3, 224, 224)
    # output_shape = (batch_size,)
    # mod.bind(for_training=False, data_shapes=[('data', input_shape)],
            # label_shapes=[('softmax_label', output_shape)])
    # mod.set_params(mx_params, {})

    sym_block = resnet.load_graph(ctx)
    # print (qsym_block.collect_params().keys())

    print ("calculate model accuracy")
    qacc, acc, diff, total = 0, 0, 0, 0
    for i in range(100):
        image_data = calib_data.data[0]
        qimage_data, _ = quant_helper(image_data)

        res = sym_block.forward(image_data.as_in_context(ctx))
        # comp_res, _ = quant_helper(res)
        # print (res.max().asnumpy(), comp_res[0].max().asnumpy(), comp_res[0].min().asnumpy())

        if quant_flag.calib_mode == CalibMode.NONE:
            qimage_data = image_data
        qres = qsym_block.forward(qimage_data.as_in_context(ctx))
        # qres = mod.predict(image_data.as_in_context(ctx))
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

def mx_quant(sym, params, data_iter, ctx):
    symbol_file = "./data/mxquant-symbol.json"
    params_file = "./data/mxquant-0000.params"

    if os.path.exists(symbol_file):
        print ("Load exist mxnet quant model")
        sym = mx.sym.load(symbol_file)
        params = nd.load(params_file)

        return sym, params
    else:
        sym = mx.sym.SoftmaxOutput(sym, name='softmax')

        arg_params, aux_params = {}, {}
        for key, value in params.items():
            if "running" in key:
                aux_params[key] = value
            else:
                arg_params[key] = value
        print(params.keys())
        # print ("ddddd", sym.get_internals())

        sym, arg_params, aux_params = mquant.quantize_model(sym,
                arg_params, aux_params, ctx=ctx,
                excluded_sym_names=["conv0_fwd"],
                calib_data=data_iter,
                num_calib_examples=100)

        with open(symbol_file, "w") as fout:
            fout.write(sym.tojson())
        nd.save(params_file, {**arg_params, **aux_params})

        return sym, {**arg_params, **aux_params}

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
            format="[ %(asctime)s %(name)s.%(levelname)s ] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("log")

    log_filter = FilterList(
                allows=["log.quant.op.dddd", "log.calib"],
                disables=["log.quant.op.requant.helper"],
                log_level=logging.INFO)
    for handler in logging.root.handlers:
            handler.addFilter(log_filter)

    load_mxnet_resnet()
    # mx_quant()

