import json
import sys
from os import path
import logging

import mxnet as mx
from mxnet import ndarray as nd

import utils
import dataset as ds
import sym_pass as spass
import sym_utils as sutils
import sim_quant_helper as sim
import mrt as _mrt

utils.log_init()
logger = logging.getLogger("log.main")

Usage = r"""
Usage: python main.py json.cfg

json.cfg    configuration for MRT quantization
    symbol: the symbol file path.
    params: the params file path.

    cvm_symbol:     quantized model name dumped.
    cvm_params:     quantized model name dumped.
    cvm_ext:        quantized model scale name dumped.

    input_shape:    axis stands for batch_size set -1, such as (-1, 3, 224, 224).
    dataset:        model train and test dataset.
    quantization:   MRT quantization flags.
        batch_size  batch size loaded from dataset.
        pure_int8   whether use int8 for internal input and output.
        device      device information for MRT contexts, available options is
                    "cpu" or "gpu", and the second number stands for device id.
                    eg: ["cpu"], ["gpu", 0].
        calibrate_num       calibration iterator number.
        output_precision    set model output precision, by default is non set(-1).

        fixed       set fixed symbol without scale.
        thresholds  set symbol output range manually instead of calibrate.

        split_names     split model into two part for yolo.
        name_maps       model output name maps to split_names for scales.
        valid_thresh    the operator box_nms scale corresponding with split_names.
    cvm:            transform to cvm supported model flags
        batch_size      batch size compiled to cvm model,
                        by default same as quantization batch size.
        save_ext        whether save quantized information 
"""

def CHECK(cond, err="", lgr=logger):
    if not cond:
        lgr.info(Usage)
        lgr.error(err)
        exit()

if __name__ == "__main__":
    CHECK(len(sys.argv) == 2, "args number must be 2")
    cfgPath = sys.argv[1]
    CHECK(path.exists(cfgPath) and path.isfile(cfgPath), "file path error")

    baseDir = path.abspath(path.dirname(cfgPath))
    logger.info("Load config file: %s", cfgPath)
    with open(cfgPath, "r") as fin:
        cfg = eval(fin.read())

    CHECK("symbol" in cfg and "params" in cfg, "config error: model path")
    CHECK("input_shape" in cfg, "config error: input_shape")
    CHECK("quantization" in cfg, "config error: quantization")
    CHECK("dataset" in cfg, "config error: dataset")
    CHECK("cvm" in cfg, "config error: cvm")
    cvm_sym = cfg.get("cvm_symbol", "./cvm.symbol")
    cvm_prm = cfg.get("cvm_params", "./cvm.params")
    cvm_ext = cfg.get("cvm_ext", "./cvm.ext")

    sym_file, prm_file = cfg["symbol"], cfg["params"]
    if not path.isabs(sym_file):
        sym_file = path.abspath(path.join(baseDir, sym_file))
    if not path.isabs(prm_file):
        prm_file = path.abspath(path.join(baseDir, prm_file))

    quant_flag = cfg["quantization"]
    batch_size = int(quant_flag.get("batch_size", 1))
    pure_int8 = bool(quant_flag.get("pure_int8", False))
    calibrate_num = int(quant_flag.get("calibrate_num", 1))
    CHECK(batch_size > 0, "config error: batch_size")

    device = quant_flag.get("device", ["cpu"])
    ctx = mx.gpu(device[1]) if device[0] == "gpu" else mx.cpu()

    input_shape = cfg["input_shape"]
    shp = tuple(batch_size if s == -1 else s for s in input_shape)
    inputs_ext = { "data": {
        "shape": shp,
    } }

    dataset = cfg["dataset"]
    if dataset == "imagenet":
        data_iter = ds.load_imagenet_rec(batch_size, shp[2])
        def data_iter_func():
            data = data_iter.next()
            return data.data[0], data.label[0]
    elif dataset == "voc":
        val_data = ds.load_voc(batch_size, shp[2])
        val_data_iter = iter(val_data)
        def data_iter_func():
            data, label = next(val_data_iter)
            return data, label
    elif dataset == "trec":
        data_iter = ds.load_trec(batch_size)
        def data_iter_func():
            return next(data_iter)
    else:
         CHECK(False, "config error: dataset")

    inputs = [mx.sym.var("data")]
    sym, params = mx.sym.load(sym_file), nd.load(prm_file)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)

    keys = quant_flag.get("split_names", [])
    if len(keys) > 0:
        sym, params, inputs_ext, sym2, prm2, ins_ext2 \
            = _mrt.split_model(sym, params, inputs_ext, keys)

        CHECK("name_maps" in quant_flag, "config error: name_maps")
        name_maps = quant_flag.get("name_maps")

        CHECK("valid_thresh" in quant_flag, "config error: valid_thresh")
        valid_name = quant_flag.get("valid_thresh")

    thresholds = quant_flag.get("thresholds", {})
    fixed = quant_flag.get("fixed", [])
    oprec = quant_flag.get("output_precision", -1)

    mrt = _mrt.MRT(sym, params, inputs_ext)     # initialize
    for i in range(calibrate_num):
        data, _ = data_iter_func()
        mrt.set_data('data', data)              # set input data
        mrt.calibrate(ctx=ctx)                  # calibration
    for k, v in thresholds.items():
        mrt.set_threshold(k, v)
    for k in fixed:
        mrt.set_fixed(k)
    if oprec > 0:
        mrt.set_output_prec(oprec)
    if pure_int8:
        mrt.set_pure_int8()
    qsym, qparams, inputs_ext = mrt.quantize()  # quantization

    oscales = mrt.get_output_scales()
    if len(keys) > 0:
        oscales_dict = dict(zip([c.attr('name') for c in sym], oscales))
        oscales = [oscales_dict[name_maps[c.attr('name')]] for c in sym2]
        def box_nms(node, params, graph):
            name, op_name = node.attr('name'), node.attr('op_name')
            childs, attr = sutils.sym_iter(node.get_children()), node.list_attr()
            if op_name == '_contrib_box_nms':
                valid_thresh = sutils.get_attr(attr, 'valid_thresh', 0)
                attr['valid_thresh'] = int(valid_thresh * oscales_dict[valid_name])
                node = sutils.get_mxnet_op(op_name)(*childs, **attr, name=name)
            return node
        maps = mrt.get_maps()
        qsym, qparams = _mrt.merge_model(qsym, qparams, sym2, prm2, maps, box_nms)

    cvm_flag = cfg["cvm"]
    cvm_batch_size = cvm_flag.get("batch_size", batch_size)

    shp = tuple(cvm_batch_size if s == -1 else s for s in input_shape)
    inputs_ext["data"]["shape"] = shp
    nnvm_sym, nnvm_params = spass.mxnet_to_nnvm(qsym, qparams, inputs_ext)

    spass.cvm_build(nnvm_sym, nnvm_params, inputs_ext, cvm_sym, cvm_prm)

    if cvm_flag.get("save_ext", False):
        sim.save_ext(cvm_ext, inputs_ext, oscales)


