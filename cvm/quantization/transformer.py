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

def split_model(symbol, params, keys):
    infer_shapes = infer_shape(symbol, params)
    bases = [s for s in topo_sort(symbol) if s.attr('name') in keys]
    base = mx.sym.Group(bases)
    base_params = {k:params[k] for k in base.list_inputs() if k in params}

    graph = {}
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [sutils.get_node(c, graph) for c in childs]
            node = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in keys:
            node = mx.sym.var(name,
                              shape=infer_shapes[name][get_entry_id(sym)])
        graph[name] = node
    nodes = [sutils.get_node(c, graph) for c in symbol]
    top = nodes[0] if len(nodes) == 1 else mx.sym.Group(nodes)
    top_params = {k:params[k] for k in top.list_inputs() if k in params}

    return base, base_params, top, top_params

def merge_model(base, base_params, top, top_params,
                base_maps=None, callback=None):
    logger = logging.getLogger("mrt.model.merge")
    logger.info("Merge model with map: %s", base_maps)

    base_maps = {} if base_maps is None else base_maps
    graph = {base_maps.get(c.attr('name'), c.attr('name')):c for c in base}
    for sym in topo_sort(top):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [sutils.get_node(c, graph) for c in childs]
            node = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in graph:
            node = graph[name]
        if callback is not None:
            node = callback(node, top_params, graph)
        graph[name] = node
    nodes = [sutils.get_node(s, graph) for s in top]
    symbol = nodes[0] if len(nodes) == 1 else mx.sym.Group(nodes)
    params = base_params
    params.update(top_params)
    return symbol, params


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
