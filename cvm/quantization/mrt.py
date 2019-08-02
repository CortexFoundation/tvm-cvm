import mxnet as mx
from mxnet import ndarray as nd

import numpy as np
import math

from sym_utils import *
from utils import *
import sym_pass as spass
import sim_quant_helper as sim

disable_requant_ops = [
    'Activation', 'relu',
    'Pooling',
    'slice', 'slice_like', 'slice_axis',
    'clip', 'negative',
    'repeat', 'tile', 'expand_dims', 'squeeze',
    'Reshape', 'transpose', 'Flatten',
    'max', 'upsampling',
]

def RPassThrough(prec, oprec):
    l = max(prec.l, oprec.l)
    p = prec.p if prec.l > oprec.l else oprec.p
    return PREC(p, l)
def RInjective(prec, oprec):
    l = max(prec.l, oprec.l)
    if prec.l > oprec.l:
        assert prec.p <= oprec.p
        p = prec.p
    else:
        p = oprec.p
    return PREC(p, l)
LPassThrough = 'level_pass_through'
LInjective = 'level_injective'
LMaxInput = 'level_max_input'
LMax, LMin = 'level_maximum', 'level_minimum'
LSelectIndexOne = 'level_select_index_one'
L0, L1, L2, L3, L4, L5, L6 = 0, 25, 50, 75, 100, 150, 200
LFIX = 1000
class PREC():
    def __init__(self, *args):
        if (len(args) == 0):
            self.p, self.l = -1, L0
        elif isinstance(args[0], PREC):
            self.p, self.l = args[0].p, args[0].l
        elif isinstance(args[0], int):
            self.p = args[0]
            self.l = L0 if len(args) == 1 else args[1]
        else:
            assert False, "args: %s"%(args)
    def __lt__(self, other):
        return self.p < other.p
    def __le__(self, other):
        return self.p <= other.p
    def __eq__(self, other):
        return self.p == other.p
    def __gt__(self, other):
        return self.p > other.p
    def __ge__(self, other):
        return self.p >= other.p
    def __repr__(self):
        return "<%d, %d>"%(self.p, self.l)

out_key = 'out_key'
target_key = 'target_key'
def is_inputs(sym, params, inputs_ext):
    return (sym.attr('name') in inputs_ext)
def is_params(sym, params, inputs_ext):
    return (sym.attr('name') in params)
def is_var(sym, params, inputs_ext):
    return (sym.attr('op_name') == 'null')
def is_op(sym, params, inputs_ext):
    return (sym.attr('op_name') != 'null')

def scale(threshold, precision):
    assert threshold >= 0
    if threshold == 0:
        return 1
    alpha = (2 ** (precision - 1)) - 1
    return alpha / threshold

def _mrt_sim_quantize(sym, sb, params, graph, prec):
    name = "%s_%d_%d" % (sym.attr('name'), sb, prec)
    if name in graph:
        return graph[name]

    return mx.sym.Custom(sym, sb=sb, prec=prec,
                name=name, op_type='mrt_sim_quant')

MAX_BIT = 32
id_counts = {}
def _uniq_name(name):
    if name in id_counts:
        id_counts[name] += 1
    else:
        id_counts[name] = 0
    return "%s_%d" % (name, id_counts[name])
def _get_prec(precs, name, prec=None):
    if name not in precs:
        assert prec is not None
        precs[name] = PREC(prec)
    return precs[name]
def _get_range(prec):
    return (2 ** (prec - 1)) - 1
def _get_bit(opt):
    if isinstance(opt, nd.NDArray):
        opt = opt.abs().max().asscalar()
    if opt == 0:
        return 1
    return math.ceil(math.log2(opt)) + 1


def _simulate(sym, params, graph, inputs_ext, self):
    logger = logging.getLogger('log.mrt.simulate')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    infer_shapes, th_dict = self.shpes, self.th_dict
    precs, scales = self.precs, self.scales
    rtypes, ltypes = self._rtypes, self._ltypes
    op_input_precs = self._op_input_precs

    cns = [c.attr('name') for c in childs] if childs else []
    def _requant_parameter(pname, def_prec, oscale=None):
        P_name = _uniq_name(pname)
        P_prec = _get_prec(precs[pname], name, def_prec)
        xs = oscale if oscale else scale(th_dict[pname], P_prec.p)
        params[P_name] = params[pname] * xs
        P_attr = { 'precision': str(P_prec.p) }
        graph[P_name] = mx.sym.var(P_name,
                shape=params[P_name].shape, attr=P_attr)
        logger.debug(
            "Parameter th_dict=%-12.8f name=%-40s requantize with scale=%-16.8f to prec=%s",
                th_dict[pname], pname, xs, P_prec)
        return graph[P_name], P_prec, xs
    def _requant_operator(X, def_prec, oscale=None):
        xopn, xn = X.attr('op_name'), X.attr('name')
        X_name = _uniq_name(xn)
        oprec = _get_prec(precs[xn], name, def_prec)
        exactly = True if oscale else False
        oscale = oscale if oscale else scale(th_dict[xn], oprec.p)
        iscale = scales[xn]
        iprec = precs[xn][out_key]
        oprec = rtypes[name](iprec, oprec)
        if exactly:
            in_prec = _get_bit(th_dict[xn] * iscale)
            out_prec = oprec.p
            sb = in_prec - out_prec if in_prec > out_prec else 0
            if sb > 1:
                iprec = PREC(iprec.p - sb)
                X = _mrt_sim_quantize(X, sb, params, graph, iprec.p)
                iscale = iscale / (2 ** sb)
                logger.debug(
                    "Operator  %-20s name=%-40s exactly quantize with sb=%s" +
                    " scale=%s, prec=%s",
                        xopn, xn, sb, iscale, iprec)

        if exactly or (iprec > oprec and iscale > oscale):
            rescale = oscale / iscale
            frac, exp = sim.cvm_float(rescale, MAX_BIT - iprec.p)
            sim_scale = frac * (2 ** exp)
            scale_err = abs((sim_scale - rescale) / rescale)
            if exactly and scale_err > 0.001:
                logger.warn(
                    "Operator  %-20s name=%-40s requantize to scale=%s " +
                    "with <%s, %d, %d>, error=%s",
                        xopn, xn, rescale, sim_scale, frac, exp, scale_err)
            oscale = iscale * frac * (2 ** exp)
            if frac > 1:
                X = _mrt_sim_quantize(X, 0, params, graph, iprec.p)
                var = mx_const(frac, graph, params)
                mul_name = _uniq_name("mrt_quantize_scale")
                X = mx.sym.broadcast_mul(X, var, name=mul_name)
            X = _mrt_sim_quantize(X, (-exp), params, graph, oprec.p)
            logger.debug(
                "Operator  %-20s name=%-40s requantize with scale=%-16.8f<%d, %d>" +
                " iprec=%s, iscale=%-10.5f, oprec=%s, oscale=%-10.5f",
                    xopn, xn, rescale, frac, exp, iprec, iscale,
                    oprec, oscale)
        # elif (iprec > oprec and iscale <= oscale):
        else:
            X = _mrt_sim_quantize(X, 0, params, graph, oprec.p)
            oscale = iscale
            logger.debug(
                "Operator  %-20s name=%-40s clip with iprec=%s, oprec=%s",
                    xopn, xn, iprec, oprec)
        # else:
            # oscale = iscale
        return X, oprec, oscale
    def _requant(X, def_prec, oscale=None):
        if is_params(X, params, inputs_ext):
            return _requant_parameter(X.attr('name'), def_prec, oscale)
        else:
            return _requant_operator(X, def_prec, oscale)

    # Update four attributes: th_dict, precs, scales, sym
    if is_inputs(sym, params, inputs_ext):
        prec = precs[name][out_key]
        scales[name] = scale(th_dict[name], prec.p)
        attr = { 'precision': str(prec.p) }
        sym = mx.sym.var(name, attr=attr)
        return sym, params
    elif is_params(sym, params, inputs_ext):
        return sym, params
    elif op_name in disable_requant_ops:
        # TODO: pass through thresholds
        # th_dict[name] = th_dict[cns[0]]
        precs[name][out_key] = PREC(precs[cns[0]][out_key])
        scales[name] = scales[cns[0]]
    elif op_name in ['sigmoid', 'exp']:
        iprec = op_input_precs[op_name]
        xs = scale(th_dict[cns[0]], iprec.p)
        X, xprec, xs = _requant_operator(childs[0], iprec, xs)
        alpha = _get_range(xprec.p)

        data = nd.arange(-alpha, alpha+1)
        out = get_nd_op(op_name)(data / xs)
        oprec = _get_prec(precs[name], out_key, PREC(16, L0))
        opt = out.abs().max().asscalar()
        # opt = th_dict[name]
        oscale = scales[name] = scale(opt, oprec.p)

        W_name = _uniq_name("cvm_lut_weight")
        weight = (out * oscale).round().reshape(2*alpha, 1)
        params[W_name] = weight
        wattr = { 'precision': str(oprec.p)}
        W = graph[W_name] = mx.sym.var(W_name, shape=weight.shape, attr=wattr)
        var = mx_const(alpha, graph, params)
        add_name = _uniq_name(op_name + "_offset")
        X = mx.sym.broadcast_add(X, var, name=add_name)
        sym = mx.sym.Custom(X, W, in_dim=2*alpha,
                name=name, op_type='cvm_lut')
        precs[name][out_key] = oprec
    elif op_name in ['Convolution', 'FullyConnected']:
        iprec = op_input_precs[op_name]
        X, xprec, xs = _requant_operator(childs[0], iprec)
        W, wprec, ws = _requant_parameter(cns[1], iprec)
        B, bprec = None, PREC()
        if not get_attr(attr, 'no_bias', False):
            bs = ws * xs
            bias_prec = PREC(_get_bit(th_dict[cns[2]] * bs))
            B, bprec, _ = _requant_parameter(cns[2], bias_prec, bs)
        oscale = scales[name] = ws * xs
        sym = get_mxnet_op(op_name)(X, W, B, **attr, name=name)
        precs[name][out_key] = PREC(_get_bit(th_dict[name] * oscale))
    elif op_name in ['broadcast_mul']:
        iprec = op_input_precs[op_name]
        X, xprec, xs = _requant(childs[0], iprec)
        B, bprec, bs = _requant(childs[1], iprec)
        oscale = scales[name] = xs * bs
        sym = get_mxnet_op(op_name)(X, B, **attr, name=name)
        precs[name][out_key] = PREC(_get_bit(th_dict[name] * oscale))
    elif op_name in ['sum']:
        iprec = op_input_precs[op_name]
        X, xprec, xs = _requant_operator(childs[0], iprec)
        oscale = scales[name] = xs
        sym = get_mxnet_op(op_name)(X, **attr, name=name)
        precs[name][out_key] = PREC(_get_bit(th_dict[name] * oscale))
    elif op_name in ['elemwise_add', 'elemwise_sub',
            'broadcast_add', 'broadcast_sub',
            'Concat']:
        iprec = op_input_precs[op_name]
        in_th = max([th_dict[n] for n in cns])
        oscale = scales[name] = scale(in_th, iprec.p)
        #  print ([th_dict[n] for n in cns], iprec, oscale)
        new_childs = []
        for c in childs:
            c, cprec, _ = _requant(c, iprec, oscale=oscale)
            new_childs.append(c)
        sym = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
        precs[name][out_key] = PREC(_get_bit(th_dict[name] * oscale))
    elif op_name in ['Embedding']:
        iprec = op_input_precs[op_name]
        X, xs = childs[0], scales[cns[0]]
        if xs != 1:
            X, xprec, _ = _requant_operator(childs[0], PREC(32), 1/xs)
        W, wprec, ws = _requant_parameter(cns[1], iprec)
        th_dict[name] = th_dict[cns[1]]
        oscale = scales[name] = ws
        sym = get_mxnet_op(op_name)(X, W, **attr, name=name)
        precs[name][out_key] = PREC(_get_bit(th_dict[name] * oscale))
    else:
        print (name, op_name, attr)
        assert False

    logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
           op_name, name, scales[name], cns)

    # Requantize output symbol
    if name in precs[name]:
        oprec = precs[name][name]
        os = scale(th_dict[name], oprec.p)
        sym, oprec, os = _requant_operator(sym, PREC(oprec), os)
        scales[name] = os

    oname = sym.attr('name')
    infer_shapes[oname] = infer_shapes[name]
    th_dict[oname] = th_dict[name]
    precs[oname] = precs[name]
    scales[oname] = scales[name]
    return sym, params

def _realize(sym, params, graph, inputs_ext):
    logger = logging.getLogger('log.mrt.realize')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()

    is_mrt_simq = lambda : op_name=='Custom' and \
        get_attr(attr, 'op_type', 'null')=='mrt_sim_quant'
    if is_params(sym, params, inputs_ext):
        if 'precision' in attr:
            prec = get_attr(attr, 'precision')
        else:
            prec = _get_bit(params[name])
            sym = mx.sym.var(name, attr={ 'precision': str(prec) },
                    shape=params[name].shape)
            logger.warn(
                "parameter %-40s independent from graph with precision %d",
                    name, prec)
        data = params[name]
        params[name] = sim.int_realize(data, prec, logger=logger)
        return sym, params
    elif not is_mrt_simq():
        return sym, params

    X = childs[0]
    sb, prec = get_attr(attr, 'sb'), get_attr(attr, 'prec')
    if sb == 0:
        sym = mx.sym.Custom(X, precision=prec,
                cvm_name=name,
                name=name, op_type='cvm_clip')
    elif sb < 0:
        sym = mx.sym.Custom(X, shift_bit=-sb, precision=prec,
                name=name, op_type='cvm_left_shift')
    else:
        sym = mx.sym.Custom(X, shift_bit=sb, precision=prec,
                cvm_name=name,
                name=name, op_type='cvm_right_shift')
    return sym, params

class MRT():
    def __init__(self, symbol, params, inputs_ext):
        self.sym = symbol
        self.prm = params
        self.ins_ext = inputs_ext

        self.precs = {}
        self.th_dict = None
        self.scales = {}

        self._fixed = set()
        self._datas = {}
        self._ltypes = {}
        self._rtypes = {}
        self._op_input_precs = self._op_default_input_precs()
        self._lgr = logging.getLogger('log.mrt')
        self._set_prerequisites()

    def set_input_prec(self, name, prec=8, level=L5):
        self.precs[name][out_key] = PREC(prec, level)

    def set_output_prec(self, prec, level=L5):
        """ Output precision used by point to self in network
        """
        for sym in self.sym:
            name = sym.attr('name')
            self.precs[name][name] = PREC(prec, level)

    def set_pure_int8(self):
        for k,v in self._op_input_precs.items():
            v.p = 8

    def set_fixed(self, fixes):
        if isinstance(fixes, list):
            self._fixed.update(fixes)
        else:
            self._fixed.add(fixes)

    def set_data(self, name, data):
        if name not in self.ins_ext:
            self._lgr.warn("name %s not in inputs_ext %s",
                    name, self.ins_ext.keys())
            return
        # TODO: multiple data calibration
        # if isinstance(data, nd.NDArray):
        #     data = [data]
        self._datas[name] = data

    def set_threshold(self, name, threshold):
        self.th_dict[name] = threshold
    def set_th_dict(self, th_dict):
        self.th_dict = th_dict

    def calibrate(self, ctx=mx.cpu()):
        for k in self.ins_ext:
            assert k in self._datas, "Input data `%s` not set"%k
        self._lgr.info("calibrate model outputs")
        self.th_dict = self._sym_calibrate(ctx=ctx)
        return self.th_dict

    def quantize(self, no_realize=False):
        if self.th_dict is None:
            self._lgr.error("Please calibrate thresholds first.")
            assert False

        self._check_fixed()
        print (sym_collect_attr(self.sym))

        qsym, qparams = self._simulate()
        if not no_realize:
            qsym, qparams = self._realize()
        qext = self._get_ext()
        return qsym, qparams, qext

    def get_output_scales(self):
        oscales = []
        for s in self.qsym:
            name = s.attr('name')
            if name in self.scales:
                oscales.append(self.scales[name])
            else:
                oscales.append(1)
        return oscales

    def _sym_calibrate(self, ctx):
        order, deps = topo_sort(self.sym, logger=self._lgr, with_deps=True)
        old_ths = self.th_dict if self.th_dict else {}
        self.th_dict, out_cache = {}, {}
        for sym in order:
            name, op_name = sym.attr('name'), sym.attr('op_name')
            attr, childs = sym.list_attr(), sym_iter(sym.get_children())
            if op_name == 'null':
                out = self._datas[name] if name in self.ins_ext \
                      else self.prm[name]
            elif childs is None:
                out = get_nd_op(op_name)(**attr)
            else:
                cinfos = [(c.attr('name'), get_entry_id(c)) for c in childs]
                nd_inputs = [out_cache[n[0]][n[1]] for n in cinfos]
                out = get_nd_op(op_name)(*nd_inputs, **attr)
                for n, _ in cinfos:
                    assert n in deps
                    deps[n].remove(name)
                    if len(deps[n]) == 0:
                        del out_cache[n]
            out = [out] if len(sym) == 1 else out
            out_cache[name] = [o.as_in_context(ctx) for o in out]
            opts = [float(o.abs().max().asscalar()) for o in out][0]
            # TODO: out may be multiple
            if name in old_ths:
                #  th_dict[name] = [max(old_ths[name][i], o) for i,o in enumerate(opts)]
                self.th_dict[name] = max(old_ths[name], opts)
            else:
                self.th_dict[name] = opts
                self._lgr.debug(
                    "collect symbol %-40s out_shape=%-20s th_dict: (%s)",
                        name, [o.shape for o in out], self.th_dict[name])

        out_cache.clear()
        return self.th_dict

    def _get_ext(self):
        self.qext = {}
        for k, v in self.ins_ext.items():
            self.qext[k] = {
                'shape': v['shape'],
                'scale': self.scales[k],
                'target_bit': self.precs[k][out_key].p, }
        return self.qext

    def _simulate(self):
        self.qsym, self.qprm = topo_visit(self.sym, self.prm, self.ins_ext,
                get_op=get_mxnet_op, logger=self._lgr,
                callback=_simulate, self=self)

        return self.qsym, self.qprm

    def _check_cvm_precs(self):
        infer_precs, graph = {}, {}


    def _realize(self):
        qsym, qparams = topo_visit(self.qsym, self.qprm, self.ins_ext,
                get_op=get_mxnet_op, logger=self._lgr,
                callback=_realize)

        def _check_int_params(params, arg):
           param = params[arg]
           amin, amax = param.min().asscalar(), param.max().asscalar()
           msg = "key:%s max_val:%s, min_val:%s"%(arg, amax, amin)
           assert amin >= INT32_MIN and amax <= INT32_MAX, msg
           flat = param.asnumpy().flatten()
           assert all(flat.astype('int32').astype(flat.dtype) == flat), msg
        qparams = examine_parameters(qsym, qparams, self.ins_ext,
              callback=_check_int_params)
        self.qsym, self.qprm = qsym, qparams
        return self.qsym, self.qprm

    def _set_prerequisites(self):
        for sym in topo_sort(self.sym):
            name, op_name = sym.attr('name'), sym.attr('op_name')
            self.precs[name] = {}

        for k in self.ins_ext:
            self.precs[k][out_key] = PREC(8, L0)

        self.shpes = spass.sym_infer_shape(self.sym, self.prm, self.ins_ext)
        self._set_default_types()

    def _op_default_input_precs(self):
        op_precs = {}
        for n in ['Convolution', 'FullyConnected', 'sigmoid', 'exp']:
            op_precs[n] = PREC(8, L5)
        op_precs['sum'] = PREC(8, L4)
        for n in ['broadcast_add', 'broadcast_sub', 'elemwise_add', 'elemwise_sub']:
            op_precs[n] = PREC(16, L4)
        op_precs['broadcast_mul'] = PREC(16, L4)
        op_precs['Concat'] = PREC(16, L4)
        op_precs['Embedding'] = PREC(16, L0)
        return op_precs

    def _check_fixed(self):
        for sym in topo_sort(self.sym):
            name, op_name = sym.attr('name'), sym.attr('op_name')
            if name not in self._fixed:
                continue
            assert op_name == 'null'
            if is_params(sym, self.prm, self.ins_ext):
                bit = _get_bit(self.prm[name])
                if out_key in self.precs[name]:
                    prec = self.precs[name][out_key]
                    assert prec >= PREC(bit)
                    self.precs[name][out_key] = PREC(bit, LFIX)
            else:
                bit = self.precs[name][out_key].p
            self.th_dict[name] = _get_range(bit)

    def _set_default_types(self):
        for sym in topo_sort(self.sym):
            name, op_name = sym.attr('name'), sym.attr('op_name')
            if op_name == 'null':
                continue
            elif name in self._rtypes:
                continue
            elif op_name in disable_requant_ops:
                self._rtypes[name] = RPassThrough
            elif op_name in ['broadcast_add', 'broadcast_sub',
                'broadcast_mul', 'elemwise_add', 'elemwise_sub',
                'Concat']:
                self._rtypes[name] = RPassThrough
            elif op_name in ['Convolution', 'FullyConnected']:
                self._rtypes[name] = RInjective
            elif op_name in ['sum']:
                self._rtypes[name] = RInjective
            elif op_name in ['sigmoid', 'exp']:
                self._rtypes[name] = RInjective
            elif op_name in ['Embedding']:
                self._rtypes[name] = RPassThrough
            else:
                assert False, "%s name=%s"%(op_name, name)

import os
from tvm.contrib import graph_runtime
import tvm
def std_dump(sym, params, inputs_ext, data, model_name,
        is_mxnet=True,
        batch=False, data_dtype="int8", max_num=20, dump_ops=[]):
    if not batch:
        for k, v in inputs_ext.items():
            v['shape'] = (1, *v['shape'][1:])
    datadir = "/data/std_out/" + model_name
    os.makedirs(datadir, exist_ok=True)
    if is_mxnet:
        data = sim.load_real_data(data, 'data', inputs_ext)
        inputs_ext['data']['data'] = data
        spass.sym_dump_layer_outputs(sym, params, inputs_ext, datadir,
                data_dtype=data_dtype, max_num=max_num,
                dump_ops=dump_ops, ctx=mx.gpu(0))
        sym, params = spass.mxnet_to_nnvm(sym, params, inputs_ext)
    else:
        tvm_graph, tvm_params, lib = spass.cvm_build(sym, params, inputs_ext,
                "/dev/null", "/dev/null", runtime="tvm",
                target="llvm", dtype="int32")
        model = graph_runtime.create(tvm_graph, lib, tvm.cpu())
        model.set_input(**params)
        model.set_input("data", data)
        model.run()
        np.save(datadir+"/data.npy", data.asnumpy().astype('int8'))
        for i in range(len(sym.list_output_names())):
            out = model.get_output(i).asnumpy()
            np.save("%s/result_%d.npy" % (datadir, i), out)

    return spass.cvm_build(sym, params, inputs_ext,
            datadir+"/symbol", datadir+"/params")

def split_model(symbol, params, inputs_ext, keys):
    infer_shapes = spass.sym_infer_shape(symbol, params, inputs_ext)
    bases = [s for s in topo_sort(symbol) if s.attr('name') in keys]
    base = mx.sym.Group(bases)
    base_params = {k:params[k] for k in base.list_inputs() if k in params}
    base_inputs_ext = inputs_ext

    graph = {}
    inputs = {k:v for k,v in inputs_ext.items()}
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
    top_inputs_ext = {k:v for k,v in inputs.items() if k not in inputs_ext}

    return base, base_params, base_inputs_ext, top, top_params, top_inputs_ext

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











