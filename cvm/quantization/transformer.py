from mxnet import gluon
import tvm
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
        self.csym, self.cprm = symbol, params
        self._ishp = input_shape

        self._data = None
        self._fixed = set()

        self.precs = {}
        self.precs['data'] = { OUT_KEY: input_prec }
        self.th_dict = {}
        self.scales = {}

        self._qext = None

        self.op_input_precs = self._op_default_input_precs()
        self.csym, self.cprm = init(self.csym, self.cprm, self._ishp)
        self.csym, self.cprm = self._prepare()
        self._update_precs()

        self.rsym, self.rprm = self.csym, self.cprm

    def compile(self, model_name):
        logger = logging.getLogger('mrt.compile')
        datadir = "/data/std_out/" + model_name
        sym, params = prepare_for_compile(self._qsym, self._qprm)
        nnvm_sym, _ = compile(sym, params)
        args = nnvm_sym.list_input_names()
        real_params = {}
        use_dtype = "int32"
        tvm_ctx = tvm.context("llvm", 0)
        for key, value in params.items():
            if key not in args:
                continue

            msg = "key:%s value:%s"%(key, value)
            flat = value.asnumpy().flatten()
            assert all(flat >= INT32_MIN) and all(flat <= INT32_MAX), msg
            assert all(flat.astype('int32').astype('float32') == flat), msg
            real_params[key] = tvm.nd.array(value.astype(use_dtype).asnumpy(), tvm_ctx)
        return self.cvm_build(nnvm_sym, real_params,
                datadir+"/symbol", datadir+"/params")

    def cvm_build(self, nnvm_sym, nnvm_params, dump_sym, dump_params,
            target="cuda", logger=logging, dtype="int32"):
        tvm_ctx = tvm.context(target, 0)
        inputs_shape = {'data': self._ishp}
        with nnvm.compiler.build_config(opt_level=0):
            deploy_graph, lib, real_params = nnvm.compiler.build(
                nnvm_sym, target=target, shape=inputs_shape,
                params=nnvm_params, dtype=dtype)
        real_params = self.tvm_params_reduce(
                    nnvm_sym, real_params, tvm_ctx)
        open(dump_sym, "w").write(deploy_graph.json())
        param_bytes = nnvm.compiler.save_param_dict(real_params)
        open(dump_params, "wb").write(param_bytes)
        return deploy_graph, real_params

    def tvm_params_reduce(self, symbol, params, ctx):
        for sym in topo_sort(symbol):
            name, attr = sym.attr('name'), sym.list_attr()
            if is_params(sym, params):
                precision = get_attr(attr, "precision")
                val = params[name].asnumpy()
                if precision > 8:
                    params[name] = tvm.nd.array(val.astype('int32'), ctx)
                else:
                    params[name] = tvm.nd.array(val.astype('int8'), ctx)
        return params

    def _prepare(self):
        self._lgr = logging.getLogger('mrt')
        self._lgr.info("Graph initialize and reduce...")

        _sym, _prm = self.csym, self.cprm
        _sym, _prm = fuse_multiple_outputs(_sym, _prm)
        orig_ops = calculate_ops(_sym, _prm)
        _sym, _prm = fuse_constant(_sym, _prm)
        _sym, _prm = fuse_transpose(_sym, _prm)
        _sym, _prm = rewrite(_sym, _prm)
        _sym, _prm = fuse_constant(_sym, _prm)

        self._lgr.info("Original ops[%s] reduced into %s",
                orig_ops, calculate_ops(_sym, _prm))
        return _sym, _prm

    def calibrate(self, ctx=mx.cpu(), lambd=None, old_ths=None):
        self.th_dict = sym_calibrate(self.csym, self.cprm, self._data,
                ctx=ctx, lambd=lambd, old_ths=old_ths)
        return self.th_dict

    def quantize(self, no_realize=False):
        if self.th_dict is None:
            self._lgr.error("Please calibrate thresholds first.")
            assert False

        self._check_fixed()
        self.csym, self.cprm = quantize(self.csym, self.cprm,
                self.th_dict, self.precs, self.scales, self.op_input_precs)
        self._get_ext()
        return self.csym, self.cprm, self._qext

    def set_data(self, data):
        self._data = data

    def set_threshold(self, name, threshold):
        self.th_dict[name] = threshold

    def set_th_dict(self, th_dict):
        self.th_dict = th_dict

    def set_input_prec(self, prec):
        self.precs['data'][OUT_KEY] = prec

    def set_output_prec(self, prec):
        for sym in self.csym:
            name = sym.attr('name')
            self.precs[name][name] = prec

    def set_fixed(self, fixes):
        if isinstance(fixes, list):
            self._fixed.update(fixes)
        else:
            self._fixed.add(fixes)

    def set_threshold(self, name, threshold):
        self.th_dict[name] = threshold

    def get_output_scales(self):
        oscales = []
        for s in self.csym:
            name = s.attr('name')
            if name in self.scales:
                oscales.append(self.scales[name])
            else:
                oscales.append(1)
        return oscales

    def get_maps(self):
        return dict(zip([c.attr('name') for c in self.csym],
                    [c.attr('name') for c in self.rsym]))

    def _op_default_input_precs(self):
        op_precs = {}
        for n in ['Convolution', 'FullyConnected', 'sigmoid', 'exp', 'softmax']:
            op_precs[n] = 8
        op_precs['sum'] = 8
        for n in ['broadcast_add', 'broadcast_sub', 'elemwise_add', 'elemwise_sub', 'slice_like']:
            op_precs[n] = 16
        op_precs['broadcast_mul'] = 16
        op_precs['Concat'] = 16
        op_precs['Embedding'] = 16
        op_precs['slice_like'] = 30
        return op_precs

    def _update_precs(self):
        for sym in topo_sort(self.csym):
            name = sym.attr('name')
            if name not in self.precs:
                self.precs[name] = {}

    def _get_ext(self):
        self._qext = { 'data': {
            'shape': self._ishp,
            'scale': self.scales['data'],
            'target_bit': self.precs['data'][OUT_KEY], } }

    def _check_fixed(self):
        for sym in topo_sort(self.csym):
            name, op_name = sym.attr('name'), sym.attr('op_name')
            if name not in self._fixed:
                continue
            assert op_name == 'null', (op_name, name)
            if is_params(sym, self.cprm):
                bit = get_bit(self.cprm[name])
                self.precs[name] = { OUT_KEY: bit, }
            else:
                bit = self.precs[name][OUT_KEY]
                self.precs[name] = { OUT_KEY: bit, }
            self.th_dict[name] = get_range(bit)

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

def validate_model(sym_path, prm_path, ctx, num_channel=3, input_size=224,
        batch_size=16, iter_num=10, ds_name='imagenet', from_scratch=0, lambd=None):
    flag = [False]*from_scratch + [True]*(2-from_scratch)
    model_name, _ = path.splitext(path.basename(sym_path))
    model_dir = path.dirname(sym_path)
    input_shape = (batch_size, num_channel, input_size, input_size)
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

    inputs = [mx.sym.var('data')]
    inputs_ext = { 'data': { 'shape': input_shape } }

    # validate
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

def merge_model(base, base_params, top, top_params,
        base_maps={}, callback=None):
    graph = {base_maps.get(c.attr('name'), c.attr('name')):c \
        for c in base }
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
