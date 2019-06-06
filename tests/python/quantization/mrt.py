import mxnet as mx
from mxnet import ndarray as nd

import numpy as np
import math

from sym_utils import *
from utils import *
import sym_pass as spass
import sim_quant_helper as sim

# TODO: op's requant_type: rPassThrough, rInjective

disable_requant_ops = [
    'Activation', 'relu',
    'Pooling',
    'slice', 'slice_like', 'slice_axis',
    'clip', 'negative',
    'repeat', 'tile', 'expand_dims', 'squeeze',
    'Reshape', 'transpose', 'Flatten',
    'max',
]

L0, L1, L2, L3, L4, L5, L6 = 0, 25, 50, 75, 100, 150, 200
rPassThrough = 'r_pass_through'
rInjective = 'r_injective'
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
        return self.p < other.p or self.l < other.l
    def __le__(self, other):
        return self.p <= other.p or self.l <= other.l
    def __gt__(self, other):
        return self.p > other.p or self.l > other.l
    def __ge__(self, other):
        return self.p >= other.p or self.l >= other.l
    def __eq__(self, other):
        if isinstance(other, PREC):
            return self.p == other.p
        assert isinstance(other, int)
        return self.p == other
    def __ne__(self, other):
        if isinstance(other, PREC):
            return self.p != other.p
        assert isinstance(other, int)
        return self.p != other
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
    name = sym.attr('name')
    requant_op = _uniq_name("mrt_quantize")
    return mx.sym.Custom(sym, sb=sb, prec=prec,
                name=requant_op, op_type='mrt_sim_quant')

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
    return math.ceil(math.log2(opt)) + 1

def _simulate(sym, params, graph, inputs_ext, self):
    """ MRT Quantization
    1. Request children's precision
    2. Requantize children to requested precision
    """
    logger = logging.getLogger('log.mrt.simulate')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    infer_shapes, th_dict = self.shpes, self.th_dict
    precs, scales = self.precs, self.scales

    def _requant_parameter(pname, arg):
        P_name = _uniq_name(pname)
        if isinstance(arg, PREC):
            def_prec = arg
            P_prec = _get_prec(precs[pname], name, def_prec)
            xs = scale(th_dict[pname], P_prec.p)
        else:
            xs = float(arg)
            def_prec = _get_bit(th_dict[pname] * xs)
            P_prec = _get_prec(precs[pname], name, def_prec)
        params[P_name] = params[pname] * xs
        P_attr = { 'precision': str(P_prec.p) }
        graph[P_name] = mx.sym.var(P_name,
                shape=params[P_name].shape, attr=P_attr)
        logger.debug(
            "Parameter %-40s th_dict=%-16.8f multiply scale=%-10.5f to prec=%s",
                pname, th_dict[pname], xs, P_prec)
        return graph[P_name], xs
    def _requant_operator(X, arg, exactly=False):
        xopn, xn = X.attr('op_name'), X.attr('name')
        X_name = _uniq_name(xn)
        if isinstance(arg, PREC):
            def_prec = arg
            oprec = _get_prec(precs[xn], name, def_prec).p
            oscale = scale(th_dict[xn], oprec)
        else:
            oscale = float(arg)
            def_prec = _get_bit(th_dict[xn] * oscale)
            oprec = _get_prec(precs[xn], name, def_prec).p
        iscale = scales[xn]
        iprec = precs[xn][out_key].p
        if exactly or (iprec > oprec and iscale > oscale):
            rescale = oscale / iscale
            frac, exp = sim.cvm_float(rescale, MAX_BIT - iprec)
            oscale = iscale * frac * (2 ** exp)
            if frac > 1:
                X = _mrt_sim_quantize(X, 0, params, graph, iprec)
                var = mx_const(frac, graph, params)
                X = mx.sym.broadcast_mul(X, var)
            X = _mrt_sim_quantize(X, (-exp), params, graph, oprec)
            logger.debug(
                "Operator  %-20s name=%-40s requantize with scale=%-16.8f<%d, %d>" +
                " iprec=%d, iscale=%-10.5f, oprec=%d, oscale=%-10.5f",
                    xopn, xn, rescale, frac, exp, iprec, iscale,
                    oprec, oscale)
        else:
            oscale = iscale
        return X, oscale
    def _requant(X, arg, exactly=False):
        if is_params(X, params, inputs_ext):
            return _requant_parameter(X.attr('name'), arg)
        else:
            return _requant_operator(X, arg, exactly)
    def _default_process_op(new_childs, oscale):
        orange = th_dict[name] * oscale
        def_prec = PREC(_get_bit(orange))
        oprec = _get_prec(precs[name], out_key, def_prec)
        return get_mxnet_op(op_name)(*new_childs, **attr, name=name)

    cns = [c.attr('name') for c in childs] if childs else []
    cprecs = [precs[n] for n in cns]
    cths = [th_dict[n] for n in cns]
    if is_inputs(sym, params, inputs_ext):
        prec = precs[name][out_key]
        scales[name] = scale(th_dict[name], prec.p)
        attr = { 'precision': str(prec.p) }
        sym = mx.sym.var(name, attr=attr)
        return sym, params
    elif is_params(sym, params, inputs_ext):
        # calculate by op
        return sym, params
    elif op_name in disable_requant_ops:
        # TODO: pass through thresholds
        # th_dict[name] = th_dict[cns[0]]
        precs[name][out_key] = PREC(precs[cns[0]][out_key])
        scales[name] = scales[cns[0]]
        # ip = precs[cns[0]][out_key]
        # oprec = _get_bit(scales[name] * th_dict[name])
        # precs[name][out_key] = PREC(oprec, ip.l)
        return sym, params
    elif op_name in ['sigmoid', 'exp']:
        X, xs = _requant_operator(childs[0], PREC(8, L5), True)
        iprec = _get_prec(cprecs[0], name).p
        alpha = _get_range(iprec)
        X = mx.sym.broadcast_add(X, alpha_sym)

        data = nd.arange(-alpha, alpha+1)
        out = get_nd_op(op_name)(data / scales[cns[0]])
        oprec = _get_prec(precs[name], out_key, PREC(16, L5)).p
        oscale = scale(th_dict[name], oprec)
        weight = (out * oscale).reshape(2*alpha, 1)
        W_name = _uniq_name("cvm_lut_weight")
        precs[W_name] = { out_key: oprec.p }
        params[W_name] = weight
        W = graph[W_name] = mx.sym.var(W_name, shape=weight.shape)
        var = mx_const(alpha, graph, params)
        sym = mx.sym.Custom(X, W, in_dim=2*alpha,
                name=name, op_type='cvm_lut')
    elif op_name in ['Convolution', 'FullyConnected']:
        X, xs = _requant_operator(childs[0], PREC(8, L5))
        W, ws = _requant_parameter(cns[1], PREC(8, L5))
        B = None
        if not get_attr(attr, 'no_bias', False):
            bs = ws * xs
            B, _ = _requant_parameter(cns[2], bs)
        oscale = scales[name] = ws * xs
        sym = get_mxnet_op(op_name)(X, W, B, **attr, name=name)
    elif op_name in ['broadcast_mul']:
        X, xs = _requant(childs[0], PREC(8, L4))
        B, bs = _requant(childs[1], PREC(8, L4))
        oscale = scales[name] = xs * bs
        sym = get_mxnet_op(op_name)(X, B, **attr, name=name)
    elif op_name in ['sum']:
        X, xs = _requant_operator(childs[0], PREC(8, L4))
        oscale = xs
        sym = get_mxnet_op(op_name)(X, **attr, name=name)
    elif op_name in ['elemwise_add', 'elemwise_sub',
            'broadcast_add', 'broadcast_sub',
            'Concat']:
        in_th = max([th_dict[n] for n in cns])
        in_prec = PREC(8, L4)
        oscale = scale(in_th, in_prec.p)
        new_childs = []
        for c in childs:
            c, _ = _requant(c, oscale, True)
            new_childs.append(c)
        sym = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
    elif op_name in ['Embedding']:
        X, xs = childs[0], scales[cns[0]]
        if xs != 1:
            X, _ = _requant_operator(childs[0], 1/xs, True)
        W, ws = _requant_parameter(cns[1], PREC(8, L4))
        oscale = scales[name] = ws
        new_childs = [X, W]
        th_dict[name] = th_dict[cns[1]]
        sym = get_mxnet_op(op_name)(X, W, **attr, name=name)
    else:
        print (name, op_name, attr)
        assert False

    orange = th_dict[name] * oscale
    def_prec = PREC(_get_bit(orange))
    oprec = _get_prec(precs[name], out_key, def_prec)

    if name in precs[name]:
        oprec = precs[name][name]
        sym, os = _requant_operator(sym, PREC(oprec))
        scales[sym.attr('name')] = os

    oname = sym.attr('name')
    infer_shapes[oname] = infer_shapes[name]
    th_dict[oname] = th_dict[name]
    precs[oname] = { out_key: PREC(oprec) }
    scales[oname] = oscale
    return sym, params

def _realize(sym, params, graph, inputs_ext):
    logger = logging.getLogger('log.mrt.realize')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()

    is_mrt_simq = lambda : op_name=='Custom' and \
        get_attr(attr, 'op_type', 'null')=='mrt_sim_quant'
    if is_params(sym, params, inputs_ext):
        prec = get_attr(attr, 'precision')
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

    def calibrate(self, ctx=mx.cpu()):
        for k in self.ins_ext:
            assert k in self._datas, "Input data `%s` not set"%k
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
            name = sym.attr('name')
            self.precs[name] = {}

        for k in self.ins_ext:
            self.precs[k][out_key] = PREC(8, L0)

        self.shpes = spass.sym_infer_shape(self.sym, self.prm, self.ins_ext)

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












