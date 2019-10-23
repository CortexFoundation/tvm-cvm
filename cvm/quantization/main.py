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

NoneCfg = object()
class Config(object):
    def __init__(self, name="", parent=None):
        self.name = name
        self.cfg = {}
        self.parent = parent

    def __getitem__(self, key):
        return self.cfg[key]

    def default(self, key, default):
        self.cfg[key] = default
        return self

    def set_default(self, key, opt):
        assert type(self.cfg[key]) == type(opt), \
                "%s [%s] is invalid format, Expected %s vs. %s" \
                % (self, key, type(self.cfg[key]).__name__, type(opt).__name__)
        self.cfg[key] = opt

    def declare(self, key, dtype):
        assert isinstance(dtype, type)
        self.cfg[key] = dtype
        return self

    def set_declare(self, key, opt):
        assert isinstance(opt, self.cfg[key]), \
                "%s [%s] is invalid format, Expected %s vs. %s" \
                % (self, key, self.cfg[key].__name__, type(opt).__name__)
        self.cfg[key] = opt

    def config(self, key):
        self.cfg[key] = Config(key, self)
        return self.cfg[key]

    def set_config(self, key, cfg):
        self.cfg[key].parse(cfg)

    def __str__(self):
        if self.parent:
            return str(self.parent) + " > " + self.name
        return self.name

    def parse(self, cfg):
        for k, v in self.cfg.items():
            if isinstance(v, type):
                assert k in cfg, "%s [%s] is not set" % (self, k)
                self.set_declare(k, cfg[k])
            elif isinstance(v, Config):
                assert k in cfg, "%s [%s] is not set" % (self, k)
                self.set_config(k, cfg[k])
            elif k in cfg:
                self.set_default(k, cfg[k])

        for k in cfg:
            assert k in self.cfg, "%s [%s] is not supported" % (self, k)

config = (Config()
    .declare("symbol", str)
    .declare("params", str)
    .declare("input_shape", tuple)
    .declare("dataset", str)
)

(config.config("quantization")
    .declare("batch_size", int)
    .default("pure_int8", False)
    .default("calibrate_num", 1)
    .default("device", "cpu:0")
    .declare("output_precision", int)

    .default("fixed", [])
    .default("thresholds", {})

    .default("split_names", [])
    .default("name_maps", {})
    .default("attr_scales", {})

    .default("log", False)
)

(config.config("cvm")
    .default("batch_size", -1)
    .default("save_ext", False)

    .default("dir", "./")
)

(config.config("accuracy")
    .default("iter_num", 0)
)

if __name__ == "__main__":
    utils.log_init()
    logger = logging.getLogger("log.main")

    assert len(sys.argv) == 2
    cfgPath = sys.argv[1]
    baseDir = path.abspath(path.dirname(cfgPath))
    logger.info("Load config file: %s", cfgPath)
    with open(cfgPath, "r") as fin:
        lines = [l.strip() for l in fin.readlines()]
        lines = [l for l in lines if not l.startswith("#")]
        lines = [l for l in lines if not l == ""]
        cfg = eval(" ".join(lines))

    config.parse(cfg)
    sym_file, prm_file = config["symbol"], config["params"]
    if not path.isabs(sym_file):
        sym_file = path.abspath(path.join(baseDir, sym_file))
    if not path.isabs(prm_file):
        prm_file = path.abspath(path.join(baseDir, prm_file))

    qconf = config["quantization"]
    batch_size = qconf["batch_size"]
    pure_int8 = qconf["pure_int8"]
    calibrate_num = qconf["calibrate_num"]

    device = qconf["device"].split(":")
    ctx = mx.gpu(int(device[1])) if device[0] == "gpu" else mx.cpu()

    input_shape = config["input_shape"]
    shp = tuple(batch_size if s == -1 else s for s in input_shape)
    inputs_ext = { "data": {
        "shape": shp,
    } }

    dataset = config["dataset"]
    if dataset == "imagenet":
        data_iter = ds.load_imagenet_rec(batch_size, shp[2])
        def data_iter_func():
            data = data_iter.next()
            return data.data[0], data.label[0]
    elif dataset == "voc":
        val_data = ds.load_voc(batch_size, shp[2])
        data_iter = iter(val_data)
        def data_iter_func():
            return next(data_iter)
    elif dataset == "trec":
        data_iter = ds.load_trec(batch_size)
        def data_iter_func():
            return next(data_iter)
    elif dataset == "mnist":
        val_loader = ds.load_mnist(batch_size)
        data_iter = iter(val_loader)
        def data_iter_func():
            return next(data_iter)
    elif dataset == "quickdraw":
        val_data = ds.load_quickdraw10(batch_size)
        data_iter = iter(val_data)
        def data_iter_func():
            return next(data_iter)
    else:
        assert False, "dataset:%s is not supported" % (dataset)

    inputs = [mx.sym.var("data")]
    sym, params = mx.sym.load(sym_file), nd.load(prm_file)
    sym, params = spass.sym_quant_prepare(sym, params, inputs_ext)

    debug = qconf["log"]
    if debug:
        with open(baseDir + "/mrt.prepare.json", "w") as fout:
            fout.write(sym.tojson())

    keys = qconf["split_names"]
    if len(keys) > 0:
        sym, params, inputs_ext, sym2, prm2, ins_ext2 \
            = _mrt.split_model(sym, params, inputs_ext, keys)
        name_maps = qconf["name_maps"]

    if debug:
        with open(baseDir + "/mxnet.split.json", "w") as fout:
            fout.write(sym.tojson())

    thresholds = qconf["thresholds"]
    fixed = qconf["fixed"]
    oprec = qconf["output_precision"]

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

    if debug:
        sim.save_ext(baseDir + "/mrt.quantize.ext", inputs_ext, oscales)
        with open(baseDir + "/mrt.quantize.json", "w") as fout:
            fout.write(qsym.tojson())
        nd.save(baseDir + "/mrt.quantize.params", qparams)

    if len(keys) > 0:
        oscales_dict = dict(zip([c.attr('name') for c in sym], oscales))
        oscales = [oscales_dict[name_maps[c.attr('name')]] for c in sym2]

        attr_scales = qconf["attr_scales"]
        def op_scales(node, params, graph):
            name, op_name = node.attr('name'), node.attr('op_name')
            childs, attr = sutils.sym_iter(node.get_children()), node.list_attr()
            if name in attr_scales:
                scales = attr_scales[name]
            elif op_name in attr_scales:
                scales = attr_scales[op_name]
            else:
                return node

            for k, v in scales.items():
                assert k in attr, "attribute %s not in %s(%s) with %s" \
                    % (k, op_name, name, attr.keys())
                attr[k] = int(float(attr[k]) * oscales_dict[v])
                node = sutils.get_mxnet_op(op_name)(*childs, **attr, name=name)
            return node
        maps = mrt.get_maps()
        qsym, qparams = _mrt.merge_model(qsym, qparams, sym2, prm2, maps, op_scales)

    cvm_flag = cfg["cvm"]
    cvm_batch_size = cvm_flag.get("batch_size", batch_size)

    shp = tuple(cvm_batch_size if s == -1 else s for s in input_shape)
    inputs_ext["data"]["shape"] = shp
    nnvm_sym, nnvm_params = spass.mxnet_to_nnvm(qsym, qparams, inputs_ext)

    cvm_dir = config["cvm"]["dir"]
    spass.cvm_build(nnvm_sym, nnvm_params, inputs_ext,
            path.join(cvm_dir, "cvm.symbol"),
            path.join(cvm_dir, "cvm.params"))

    if config["cvm"]["save_ext"]:
        sim.save_ext(path.join(cvm_dir, "cvm.ext"), inputs_ext, oscales)

