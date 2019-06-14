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
    logger = logging.getLogger('log.mrt.simulate')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()
    infer_shapes, th_dict = self.shpes, self.th_dict
    precs, scales = self.precs, self.scales
    rtypes, ltypes = self._rtypes, self._ltypes

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
        if out_key not in precs[xn]:
            print (xn, precs[xn])
        iprec = precs[xn][out_key]
        oprec = rtypes[name](iprec, oprec)
        if exactly or (iprec > oprec and iscale > oscale):
            rescale = oscale / iscale
            frac, exp = sim.cvm_float(rescale, MAX_BIT - iprec.p)
            oscale = iscale * frac * (2 ** exp)
            if frac > 1:
                X = _mrt_sim_quantize(X, 0, params, graph, iprec.p)
                var = mx_const(frac, graph, params)
                X = mx.sym.broadcast_mul(X, var)
            X = _mrt_sim_quantize(X, (-exp), params, graph, oprec.p)
            logger.debug(
                "Operator  %-20s name=%-40s requantize with scale=%-16.8f<%d, %d>" +
                " iprec=%s, iscale=%-10.5f, oprec=%s, oscale=%-10.5f",
                    xopn, xn, rescale, frac, exp, iprec, iscale,
                    oprec, oscale)
        elif (iprec > oprec and iscale <= oscale):
            X = _mrt_sim_quantize(X, 0, params, graph, oprec.p)
            oscale = iscale
            logger.debug(
                "Operator  %-20s name=%-40s clip with iprec=%s, oprec=%s",
                    xopn, xn, iprec, oprec)
        else:
            oscale = iscale
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
    # elif op_name in ['sigmoid', 'exp']:
    #     X, xprec, xs = _requant_operator(childs[0], PREC(8, L5), True)
    #     iprec = _get_prec(cprecs[0], name).p
    #     alpha = _get_range(iprec)
    #     X = mx.sym.broadcast_add(X, alpha_sym)

    #     data = nd.arange(-alpha, alpha+1)
    #     out = get_nd_op(op_name)(data / scales[cns[0]])
    #     oprec = _get_prec(precs[name], out_key, PREC(16, L5)).p
    #     oscale = scales[name] = scale(th_dict[name], oprec)
    #     weight = (out * oscale).reshape(2*alpha, 1)
    #     W_name = _uniq_name("cvm_lut_weight")
    #     precs[W_name] = { out_key: oprec.p }
    #     params[W_name] = weight
    #     W = graph[W_name] = mx.sym.var(W_name, shape=weight.shape)
    #     var = mx_const(alpha, graph, params)
    #     sym = mx.sym.Custom(X, W, in_dim=2*alpha,
    #             name=name, op_type='cvm_lut')
    #     precs[name][out_key] = PREC(_get_bit(th_dict[name] * oscale))
    elif op_name in ['Convolution', 'FullyConnected']:
        X, xprec, xs = _requant_operator(childs[0], PREC(8, L5))
        W, wprec, ws = _requant_parameter(cns[1], PREC(8, L5))
        B, bprec = None, PREC()
        if not get_attr(attr, 'no_bias', False):
            bs = ws * xs
            bias_prec = PREC(_get_bit(th_dict[cns[2]] * bs))
            B, bprec, _ = _requant_parameter(cns[2], bias_prec, bs)
        oscale = scales[name] = ws * xs
        sym = get_mxnet_op(op_name)(X, W, B, **attr, name=name)
        precs[name][out_key] = PREC(_get_bit(th_dict[name] * oscale))
    elif op_name in ['broadcast_mul']:
        X, xprec, xs = _requant(childs[0], PREC(8, L4))
        B, bprec, bs = _requant(childs[1], PREC(8, L4))
        oscale = scales[name] = xs * bs
        sym = get_mxnet_op(op_name)(X, B, **attr, name=name)
        precs[name][out_key] = PREC(_get_bit(th_dict[name] * oscale))
    elif op_name in ['sum']:
        X, xprec, xs = _requant_operator(childs[0], PREC(8, L4))
        oscale = scales[name] = xs
        sym = get_mxnet_op(op_name)(X, **attr, name=name)
        precs[name][out_key] = PREC(_get_bit(th_dict[name] * oscale))
    elif op_name in ['elemwise_add', 'elemwise_sub',
            'broadcast_add', 'broadcast_sub',
            'Concat']:
        in_th = max([th_dict[n] for n in cns])
        in_prec = PREC(8, L4)
        oscale = scales[name] = scale(in_th, in_prec.p)
        new_childs = []
        for c in childs:
            c, cprec, _ = _requant(c, in_prec, oscale=oscale)
            new_childs.append(c)
        sym = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
        precs[name][out_key] = PREC(_get_bit(th_dict[name] * oscale))
    elif op_name in ['Embedding']:
        X, xs = childs[0], scales[cns[0]]
        if xs != 1:
            X, xprec, _ = _requant_operator(childs[0], PREC(32), 1/xs)
        W, wprec, ws = _requant_parameter(cns[1], PREC(8, L4))
        th_dict[name] = th_dict[cns[1]]
        oscale = scales[name] = ws
        sym = get_mxnet_op(op_name)(X, W, **attr, name=name)
        precs[name][out_key] = PREC(_get_bit(th_dict[name] * oscale))
    else:
        print (name, op_name, attr)
        assert False

    # Requantize output symbol
    if name in precs[name]:
        oprec = precs[name][name]
        sym, oprec, os = _requant_operator(sym, PREC(oprec))
        scales[sym.attr('name')] = os

    oname = sym.attr('name')
    infer_shapes[oname] = infer_shapes[name]
    th_dict[oname] = th_dict[name]
    precs[oname] = precs[name]
    scales[oname] = scales[name]
    return sym, params

def _cvm_precision_check(sym, params, graph, precs,
        infer_shapes, infer_precs):
    logger = logging.getLogger('log.mrt.cvm.precision.check')
    name, op_name = sym.attr('name'), sym.attr('op_name')
    childs, attr = sym_iter(sym.get_children()), sym.list_attr()

    cns = [c.attr('name') for c in childs] if childs else []
    cprecs = [infer_precs[n] for n in cns]
    if op_name == 'null':
        oprec = get_attr(attr, 'precision')
    elif op_name in ['Convolution', 'FullyConnected']:
        X, W = childs[0], childs[1]
        if cprecs[0] > 8:
            assert precs[cns[0]][out_key].p <= 8
            X = _mrt_sim_quantize(X, 0, params)
        wshp = infer_shapes[cns[1]]
        sum_len = np.product(wshp[1:])
        sum_bit = math.ceil(math.log2(sum_len))
        oprec = cprecs[0] + cprecs[1] + sum_bit
        if not get_attr(attr, 'no_bias', False):
            oprec = max(oprec, cprecs[2]) + 1
    elif op_name in disable_requant_ops:
        oprec = cprecs[0]
    elif op_name in ['broadcast_mul']:
        oprec = cprecs[0] + cprecs[1]
    elif op_name in ['broadcast_add', 'broadcast_sub',
            'elemwise_add', 'elemwise_sub']:
        oprec = max(cprecs) + 1
    elif op_name in ['Concat']:
        oprec = max(cprecs)
    elif op_name in ['sum']:
        axis = get_attr(attr, 'axis', None)
        shp = infer_shapes[cns[0]]
        sum_axis = [shp[i] for i in axis] if axis else shp
        sum_len = np.product(sum_axis)
        sum_bit = math.ceil(math.log2(sum_len))
        oprec = cprecs[0] + sum_bit
    elif op_name == 'Custom':
        op_type = get_attr(attr, 'op_type', 'null')
        assert op_type in ['cvm_clip', 'cvm_left_shift', 'cvm_right_shift',
                'cvm_lut']
        if op_type in ['cvm_clip', 'cvm_left_shift', 'cvm_right_shift']:
            oprec = get_attr(attr, 'precision')
        else:
            oprec = cprecs[1]
    elif op_name in ['Embedding']:
        oprec = cprecs[1]
    infer_precs[name] = oprec
    logger.debug("Symbol %-20s name=%-40s prec=%-2d, using precs=%s",
            op_name, name, oprec, precs[name][out_key])

    if oprec > 32:
        sym = _mrt_sim_quantize(sym, 0, params, graph, precs[name][out_key])
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
        self._ltypes = {}
        self._rtypes = {}
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
                assert False

def std_dump(sym, params, inputs_ext, data, model_name,
        batch=False, data_dtype="int8"):
    if not batch:
        for k, v in inputs_ext.items():
            v['shape'] = (1, *v['shape'][1:])
    datadir = "/data/std_out/" + model_name
    data = sim.load_real_data(data, 'data', inputs_ext)
    inputs_ext['data']['data'] = data
    spass.sym_dump_layer_outputs(sym, params, inputs_ext, datadir,
            data_dtype=data_dtype)

    spass.mxnet_to_nnvm(sym, params, inputs_ext,
            datadir+"/symbol", datadir+"/params")












