from mxnet import gluon

import tfm_ops
from tfm_pass import *
from gluon_zoo import *

import sym_utils as sutils
import cvm_op

import logging
from os import path

#TODO(wlt): control available api for MRT

def init(symbol, params, input_shape=None):
    if input_shape is not None:
        symbol, params = attach_input_shape(symbol, params,
            {'data': input_shape})
    sym, params = graph_validate(symbol, params)
    infer_shape(sym, params) # check infer_shape is correct.
    sym, params = validate(sym, params)
    return sym, params


class MRT(object):
    def __init__(self, symbol, params, input_shape, input_prec=8):
        self._sym = symbol
        self._prm = params
        self._ishp = input_shape

        self._data = None
        self._fixed = set()

        self.precs = {}
        self._update_precs()
        self.precs['data'][OUT_KEY] = input_prec
        self.th_dict = {}
        self.scales = {}

        self._qsym = None
        self._qprm = None
        self._qext = None

        self.op_input_precs = self._op_default_input_precs()

    def prepare(self):
        self._lgr = logging.getLogger('mrt')
        self._lgr.info("Graph initialize and reduce...")

        _sym, _prm = init(self._sym, self._prm, self._ishp)
        orig_ops = calculate_ops(_sym, _prm)
        _sym, _prm = fuse_constant(_sym, _prm)
        _sym, _prm = fuse_transpose(_sym, _prm)
        self._sym, self._prm = rewrite(_sym, _prm)
        # TODO: some model may need another fuse_transpose after rewrite
        self._update_precs()
        self._lgr.info("Original ops[%s] reduced into %s",
                orig_ops, calculate_ops(_sym, _prm))

    def calibrate(self, ctx=mx.cpu(), lambd=None, old_ths=None):
        self.th_dict = sym_calibrate(self._sym, self._prm, self._data,
                ctx=ctx, lambd=lambd, old_ths=old_ths)

    def quantize(self, no_realize=False):
        if self.th_dict is None:
            self._lgr.error("Please calibrate thresholds first.")
            assert False

        self._qsym, self._qprm = quantize(self._sym, self._prm,
                self.th_dict, self.precs, self.scales, self.op_input_precs)
        self._get_ext()
        return self._qsym, self._qprm, self._qext

    def dump(self, fname="test_dump", directory="~/tvm-cvm/data/"):
        assert self._qsym is not None
        import os
        directory = path.expanduser(directory)
        prefix = path.join(directory, fname)
        qsym_path, qprm_path, qext_path = utils.extend_fname(prefix, True)

        with open(path.expanduser(qsym_path), 'w') as f:
            f.write(self._qsym.tojson())
        nd.save(qprm_path, self._qprm)
        sim.save_ext(qext_path, self._qext)
        return qsym_path, qprm_path, qext_path

    def set_data(self, data):
        self._data = data

    def set_threshold(self, name, threshold):
        self.th_dict[name] = threshold

    def set_th_dict(self, th_dict):
        self.th_dict = th_dict

    def set_input_prec(self, prec):
        self.precs['data'][OUT_KEY] = prec

    def set_output_prec(self, prec):
        for sym in self._sym:
            name = sym.attr('name')
            self.precs[name][name] = prec

    def set_fixed(self, fixes):
        if isinstance(fixes, list):
            self._fixed.update(fixes)
        else:
            self._fixed.add(fixes)

    def set_threshold(self, name, threshold):
        self.th_dict[name] = threshold

    def _op_default_input_precs(self):
        op_precs = {}
        for n in ['Convolution', 'FullyConnected', 'sigmoid', 'exp', 'softmax']:
            op_precs[n] = 8
        op_precs['sum'] = 8
        for n in ['broadcast_add', 'broadcast_sub', 'elemwise_add', 'elemwise_sub']:
            op_precs[n] = 16
        op_precs['broadcast_mul'] = 16
        op_precs['Concat'] = 16
        op_precs['Embedding'] = 16
        op_precs['slice_like'] = 30
        op_precs['_arange'] = 30
        return op_precs

    def _update_precs(self):
        for sym in topo_sort(self._sym):
            name = sym.attr('name')
            if name not in self.precs:
                self.precs[name] = {}

    def _get_ext(self):
        self._qext = { 'data': {
            'shape': self._ishp,
            'scale': self.scales['data'],
            'target_bit': self.precs['data'][OUT_KEY], } }

def load_model(model_name, sym_path, prm_path, ctx, inputs_qext=None):
    inputs = [mx.sym.var('data')]
    net = utils.load_model(sym_path, prm_path, inputs, ctx=ctx)
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    acc_top1.reset()
    acc_top5.reset()
    def model_func(data, label):
        data = sim.load_real_data(data, 'data', inputs_qext) \
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

def validate_model(sym_path, prm_path, ctx, input_size=224,
        batch_size=16, iter_num=10, ds_name='imagenet', qctx=None):
    model_name, _ = path.splitext(path.basename(sym_path))
    logger = logging.getLogger("log.validate.%s"%model_name)
    if not path.exists(sym_path) or not path.exists(prm_path):
        save_model(model_name)
    sym, params = mx.sym.load(sym_path), mx.nd.load(prm_path)

    print (collect_op_names(sym, params))
    print ("Registered Graph Pass")
    for k, v in pass_info().items():
        print ("%20s" % k, v)

    data_iter_func = ds.data_iter(ds_name, batch_size, input_size=input_size)
    data, _ = data_iter_func()
    mrt = MRT(sym, params, (batch_size, 3, input_size, input_size))
    mrt.prepare()
    mrt.set_data(data)
    mrt.calibrate()
    mrt.set_input_prec(8)
    mrt.set_output_prec(8)
    mrt.quantize()
    qsym_path, qprm_path, qext_path = mrt.dump()
    (inputs_qext, ) = sim.load_ext(qext_path)

    inputs = [mx.sym.var('data')]
    inputs_ext = { 'data': {
        'shape': (batch_size, 3, input_size, input_size), } }

    org_model, cvm_quantize = None, None
    if model_name in ['yolo3_darknet53_voc']:
        org_model, cvm_quantize = load_model_yolo(model_name, sym_path, prm_path,
                ctx, qctx, inputs_ext, inputs_qext, mrt._sym, mrt._prm)
    else:
        org_model = load_model(model_name, sym_path, prm_path, ctx)
        cvm_quantize = load_model(model_name, qsym_path, qprm_path, ctx, \
                inputs_qext=inputs_qext)

    utils.multi_validate(org_model, data_iter_func, cvm_quantize,
            iter_num=iter_num, logger=logging.getLogger('mrt.validate'))
    logger.info("test %s finished."%model_name)

def split_model(symbol, params, input_shapes, keys):
    symbol, params = attach_input_shape(symbol, params, input_shapes)
    infer_shapes = infer_shape(symbol, params)
    bases = [s for s in topo_sort(symbol) if s.attr('name') in keys]
    base = mx.sym.Group(bases)
    base_params = {k:params[k] for k in base.list_inputs() if k in params}

    graph = {}
    inputs = {k: {'shape': v} for k, v in input_shapes.items()}
    for sym in topo_sort(symbol):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [graph[c.attr('name')] for c in childs]
            node = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in keys:
            node = mx.sym.var(name)
            inputs[name] = {'shape': infer_shapes[name]}
        graph[name] = node
    nodes = [graph[sym.attr('name')] for sym in symbol]
    top = nodes[0] if len(nodes) == 1 else mx.sym.Group(nodes)
    top_params = {k:params[k] for k in top.list_inputs() if k in params}
    top_inputs_ext = {k:v for k,v in inputs.items() if k not in ['data']}

    return base, base_params, top, top_params, top_inputs_ext

def merge_model(base, base_params, top, top_params, base_maps, callback=None):
    graph = {base_maps[c.attr('name')]:c for c in base}
    for sym in topo_sort(top):
        name, op_name = sym.attr('name'), sym.attr('op_name')
        childs, attr = sym_iter(sym.get_children()), sym.list_attr()
        node = sym
        if childs is not None:
            childs = [graph[c.attr('name')] for c in childs]
            node = get_mxnet_op(op_name)(*childs, **attr, name=name)
        if name in graph:
            node = graph[name]
        if callback is not None:
            node = callback(node, top_params, graph)
        graph[name] = node
    symbols = [graph[s.attr('name')] for s in top]
    symbol = symbols[0] if len(symbols) == 1 else mx.sym.Group(symbols)
    params = base_params
    params.update(top_params)
    params = {k:params[k] for k in symbol.list_inputs() if k in params}
    return symbol, params
