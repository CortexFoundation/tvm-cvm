""" MRT Interface API

    Refractor of source code, using the registry pattern.
    Rules of coding with pylint.
    Collection of hyper-parameters controller.
    Simplification of public API.
"""

import logging
from os import path
import sys
import numpy as np

import mxnet as mx
from mxnet import gluon, ndarray as nd
import nnvm

# import as registry pattern
import tfm_ops  # pylint: disable=unused-import
import cvm_op   # pylint: disable=unused-import

from tfm_pass import OUT_KEY, convert_params_dtype
from tfm_pass import \
    attach_input_shape, graph_validate, infer_shape, \
    fuse_multiple_outputs, fuse_constant, \
    calculate_ops, collect_op_names, fuse_transpose, \
    rewrite, sym_calibrate, quantize, compile

import sym_utils as sutils
from sym_utils import topo_sort, sym_iter, get_mxnet_op
import dataset as ds
import utils
import sim_quant_helper as sim

__all__ = ["init", "MRT", "compile_to_cvm",
           "load_model", "validate_model",
           "split_model", "merge_model",
           # transformer helper pass
           "convert_params_dtype"]

def init(symbol, params, input_shape=None):
    logger = logging.getLogger("mrt.prepare")
    logger.info("Graph initialize and reduce...")

    _sym, _prm = symbol, params
    _prm = convert_params_dtype(_prm)
    if input_shape is not None:
        _sym, _prm = attach_input_shape(_sym, _prm,
                                        {'data': input_shape})
    _sym, _prm = graph_validate(_sym, _prm)

    _sym, _prm = fuse_multiple_outputs(_sym, _prm)
    orig_ops = calculate_ops(_sym, _prm)
    _sym, _prm = fuse_constant(_sym, _prm)
    _sym, _prm = fuse_transpose(_sym, _prm)
    _sym, _prm = rewrite(_sym, _prm)
    _sym, _prm = fuse_constant(_sym, _prm)

    logger.info("Original ops[%s] reduced into %s",
                orig_ops, calculate_ops(_sym, _prm))
    return _sym, _prm


class Model:
    def __init__(self, symbol, params, dtype="float64"):
        self.symbol = symbol
        self.params = convert_params_dtype(params, dest_dtype=dtype)

    def input_names(self):
        return [s.attr('name') for s in self.symbol \
            if sutils.is_inputs(s, self.params)]

    def output_names(self):
        return [s.attr('name') for s in self.symbol]

    def names(self):
        return self.output_names()

    def to_graph(self, dtype="float32", ctx=mx.cpu()):
        graph = gluon.nn.SymbolBlock(self.symbol, \
            [mx.sym.var(n) for n in self.input_names])
        utils.load_parameters(graph, convert_params_dtype(
            self.params,
            dest_dtype=dtypw), ctx=ctx)
        return graph

    def save(self, symbol_file, params_file):
        with open(symbol_file, 'w') as fout:
            fout.write(self.symbol.tojson())
        nd.save(params_file, self.params)

    @staticmethod
    def load(symbol_file, params_file):
        symbol = mx.sym.load(symbol_file)
        params = nd.load(params_file)
        return Model(symbol, params)

class MRT:
    """ An MRT quantization class contained many helper functions.

    Quantization Procedures:
    ========================
        1. prepare: initial of model graph, such as fuse_constant,
            rewrite, validate, ...etc;
        2. calibration: caculate the internal thresholds of layers;
        3. quantization: quantize the floating parameters into INT(p)
            precision with scales, using the floading data simulate
            the realized environment of interger dataflow;
    """
    def __init__(self, symbol, params, input_prec=8):
        self._lgr = logging.getLogger("mrt")

        self.csym, self.cprm = symbol, params
        self.current_model = Model(symbol, params)

        self.old_names = [s.attr('name') for s in symbol]
        self._data = None
        self.th_dict = {}

        self._op_default_input_precs()
        self.precs = {s.attr('name'):{} for s in topo_sort(self.csym)}
        if 'data' not in self.precs:
            raise RuntimeError("please invoke `init` function first")
        self.precs['data'][OUT_KEY] = input_prec
        self.scales = {}

    def set_data(self, data):
        self._data = data

    def calibrate(self, ctx=mx.cpu(), lambd=None, old_ths=None):
        self.th_dict = sym_calibrate(self.csym, self.cprm, self._data,
                                     ctx=ctx, lambd=lambd, old_ths=old_ths)
        return self.th_dict

    def set_threshold(self, name, threshold):
        self.th_dict[name] = threshold

    def set_th_dict(self, th_dict):
        self.th_dict = th_dict

    def _op_default_input_precs(self):
        op_precs = self.op_input_precs = {}
        for name in ['Convolution', 'FullyConnected',
                     'sigmoid', 'exp', 'softmax']:
            op_precs[name] = 8
        op_precs['sum'] = 8
        for name in ['broadcast_add', 'broadcast_sub',
                     'elemwise_add', 'elemwise_sub', 'slice_like']:
            op_precs[name] = 16
        op_precs['broadcast_mul'] = 16
        op_precs['Concat'] = 16
        op_precs['Embedding'] = 16
        op_precs['slice_like'] = 30

    def set_input_prec(self, prec):
        self.precs['data'][OUT_KEY] = prec

    def set_output_prec(self, prec):
        for sym in self.csym:
            name = sym.attr('name')
            self.precs[name][name] = prec

    def quantize(self):
        if self.th_dict is None:
            self._lgr.error("Please calibrate thresholds first.")
            assert False

        self.csym, self.cprm = \
            quantize(self.csym, self.cprm, self.th_dict,
                     self.precs, self.scales, self.op_input_precs)
        return self.csym, self.cprm, self.get_inputs_ext()

    def get_output_scales(self):
        oscales = []
        for sym in self.csym:
            name = sym.attr('name')
            oscales.append(self.scales[name])
        return oscales

    def get_maps(self):
        return dict(zip([c.attr('name') for c in self.csym], self.old_names))

    def get_inputs_ext(self):
        inputs_ext = {'data': {
            'scale': self.scales['data'],
            'target_bit': self.precs['data'][OUT_KEY]}}
        return inputs_ext

    def save(self, model_name, datadir="./data"):
        # pylint: disable=unbalanced-tuple-unpacking
        sym_file, params_file, ext_file = \
            utils.extend_fname(path.join(datadir, model_name), True)
        with open(sym_file, 'w') as fout:
            fout.write(self.csym.tojson())
        nd.save(params_file, self.cprm)
        sim.save_ext(ext_file,
                     self.old_names, self.th_dict,
                     self.precs, self.scales)

    @staticmethod
    def load(model_name, datadir="./data"):
        # pylint: disable=unbalanced-tuple-unpacking
        sym_file, params_file, ext_file = \
            utils.extend_fname(path.join(datadir, model_name), True)
        sym, params = mx.sym.load(sym_file), nd.load(params_file)
        sim.load_ext(ext_file)
        old_names, th_dict, precs, scales = sim.load_ext(ext_file)
        mrt = MRT(sym, params)
        mrt.old_names = old_names
        mrt.set_th_dict(th_dict)
        mrt.precs = precs
        mrt.scales = scales
        return mrt

def load_model(model_name, sym_path, prm_path, ctx, inputs_qext=None):
    inputs = [mx.sym.var('data')]
    sym, params = mx.sym.load(sym_path), nd.load(prm_path)
    net = gluon.nn.SymbolBlock(sym, inputs)
    nparams = params if inputs_qext else \
            convert_params_dtype(params, src_dtypes="float64",
                                 dest_dtype="float32")
    utils.load_parameters(net, nparams, ctx=ctx)
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    def model_func(data, label):
        data = sim.load_real_data(data.astype("float64"), 'data', inputs_qext) \
               if inputs_qext else data
        data = gluon.utils.split_and_load(data, ctx_list=ctx,
                                          batch_axis=0, even_split=False)
        res = [net.forward(d) for d in data]
        res = nd.concatenate(res)
        acc_top1.update(label, res)
        _, top1 = acc_top1.get()
        acc_top5.update(label, res)
        _, top5 = acc_top5.get()
        return "top1={:6.2%} top5={:6.2%}".format(top1, top5)
    return model_func

def validate_model(sym_path, prm_path, ctx, num_channel=3,
                   input_size=224, batch_size=16, iter_num=10,
                   ds_name='imagenet', from_scratch=0, lambd=None,
                   dump_model=False):
    from gluon_zoo import save_model

    flag = [False]*from_scratch + [True]*(2-from_scratch)
    model_name, _ = path.splitext(path.basename(sym_path))
    model_dir = path.dirname(sym_path)
    input_shape = (batch_size, num_channel, input_size, input_size)
    logger = logging.getLogger("log.validate.%s"%model_name)

    if not path.exists(sym_path) or not path.exists(prm_path):
        save_model(model_name)
    sym, params = mx.sym.load(sym_path), mx.nd.load(prm_path)

    print(collect_op_names(sym, params))

    data_iter_func = ds.data_iter(ds_name, batch_size, input_size=input_size)
    data, _ = data_iter_func()

    # prepare
    mrt = MRT(sym, params, input_shape)
    mrt.set_data(data)

    # calibrate
    prefix = path.join(model_dir, model_name+'.mrt.dict')
    _, _, dump_ext = utils.extend_fname(prefix, True)
    if flag[0]:
        th_dict = mrt.calibrate(lambd=lambd)
        sim.save_ext(dump_ext, th_dict)
    else:
        (th_dict,) = sim.load_ext(dump_ext)
        mrt.set_th_dict(th_dict)

    mrt.set_input_prec(8)
    mrt.set_output_prec(8)

    # quantize, get: qsym, qprm, inputs_qext
    qsym, qprm, inputs_qext = None, None, None
    prefix = path.join(model_dir, model_name+'.mrt.quantize')
    qsym_path, qprm_path, qext_path = utils.extend_fname(prefix, True)
    if flag[1]:
        qsym, qprm, inputs_qext = mrt.quantize()
        open(path.expanduser(qsym_path), 'w').write(qsym.tojson())
        nd.save(qprm_path, qprm)
        sim.save_ext(qext_path, inputs_qext)
    else:
        qsym, qprm = mx.sym.load(qsym_path), nd.load(qprm_path)
        (inputs_qext, ) = sim.load_ext(qext_path)

    # dump model
    if dump_model:
        datadir = "/data/ryt"
        model_name = model_name + "_tfm"
        dump_shape = (1, num_channel, input_size, input_size)
        compile_to_cvm(qsym, qprm, model_name, datadir=datadir,
                       input_shape=dump_shape)
        data = data[0].reshape(dump_shape)
        data = sim.load_real_data(data.astype("float64"), 'data', inputs_qext)
        np.save(datadir+"/"+model_name+"/data.npy", data.astype('int8').asnumpy())
        sys.exit(0)

    # validate
    org_model = load_model(model_name, sym_path, prm_path, ctx)
    cvm_quantize = load_model(model_name, qsym_path, qprm_path, ctx, \
            inputs_qext=inputs_qext)

    utils.multi_validate(org_model, data_iter_func, cvm_quantize,
                         iter_num=iter_num,
                         logger=logging.getLogger('mrt.validate'))
    logger.info("test %s finished.", model_name)

def compile_to_cvm(symbol, params, model_name,
                   datadir="/data/std_out",
                   input_shape=None, target="cuda"):
    import os
    import tvm

    logger = logging.getLogger("mrt.compile")

    datadir = path.join(datadir, model_name)
    os.makedirs(datadir, exist_ok=True)

    # transform from mxnet symbol to cvm
    logger.info("Transform Mxnet symbol into CVM")
    nnvm_sym, _ = compile(symbol, params)
    dtype, nnvm_params = "int32", {}
    tvm_ctx = tvm.context(target, 0)
    for sym in topo_sort(symbol):
        if sutils.is_params(sym, params):
            key, value = sym.attr('name'), params[sym.attr('name')]
            flat = value.asnumpy()
            assert np.abs(flat).max() <= sutils.INT32_MAX, \
                "key: {}\nvalue: {}".format(key, value)
            assert (flat.astype(dtype).astype("float64") == flat).all(), \
                "key: {}\nvalue: {}".format(key, value)
            nnvm_params[key] = tvm.nd.array(flat.astype(dtype), tvm_ctx)

    # compile to JSON&Bytes format
    # graph = nnvm.graph.create(nnvm_sym)
    # open("/tmp/tmp.nnvm.json", "w").write(graph.json())
    logger.info("Compile into CVM graph")
    if input_shape is None:
        for sym in topo_sort(symbol):
            if sutils.is_inputs(sym, params):
                _, oshp, _ = sym.infer_shape()
                assert len(oshp) == 1
                input_shape = oshp[0]
    input_shapes = {'data': input_shape}
    with nnvm.compiler.build_config(opt_level=0):
        deploy_graph, _, nnvm_params = nnvm.compiler.build(
            nnvm_sym, target=target, shape=input_shapes,
            params=nnvm_params, dtype=dtype)

    # tvm parameters reduce
    logger.info("Parameters precision reduce")
    for sym in topo_sort(nnvm_sym):
        if sutils.is_params(sym, nnvm_params):
            name, attr = sym.attr('name'), sym.list_attr()
            precision = sutils.get_attr(attr, "precision")
            dtype = "int32" if precision > 8 else "int8"
            nnvm_params[name] = tvm.nd.array(
                params[name].asnumpy().astype(dtype), tvm_ctx)

    # dump
    logger.info("CVM Json&Params dump")
    open(path.join(datadir, "symbol"), "w").write(deploy_graph.json())
    param_bytes = nnvm.compiler.save_param_dict(nnvm_params)
    open(path.join(datadir, "params"), "wb").write(param_bytes)
    return deploy_graph, nnvm_params
