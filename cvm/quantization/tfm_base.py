from sym_utils import *
import mrt as _mrt

import numpy as np

class Transformer(object):
    """ Base transformer object

        All subclass inherited from this should be registered maually
            using helper function `register_transformer`, and then
            all class function should be well-considered to override
            or use helper function `register_pass` to annotate using
            function defined in base class (that is this object),
            if there's no point to redefine duplicate function.

        Subclass should only implement function defined in base object,
            and we advise any helper function to be named with underline
            prefix.

        Please refer to file `tfm_ops.py` for more examples about
            operator transformers.

    Attributes:
    ==========
    op_name: Transformer is associated with operator which is defined
            in mxnet, and the variable indicates the type name of mxnet
            symbol.
            Attention please, the base transformer should not be instantiated
            since it's just an abstarct aggregation of graph pass, and it's
            named `none` by default.
    """
    op_name = "none"

    def __init__(self):
        if self.op_name == "none":
            raise RuntimeError("Base transformer should not be instantiated")

    def validate(self, op, **kwargs):
        """ All operators should be validated before another pass,
                neither correcting the invalid format nor asserting
                error to announce unsupported graph.

            Do nothing by default.
        """
        return op

    def rewrite(self, op, **kwargs):
        """ Operators may need to rewrite to equivalent graph which is
                easier to quantize for later procedure.

            Do nothing by default.
        """
        return op

    def quantize(self, op, **kwargs):
        """ Main procedure for quantization.

            Do nothing by default.
        """
        return op

    def compile(self, op, **kwargs):
        """ Compile mxnet symbol into nnvm symbol.

            Throw exception by default.
        """
        raise NotImplementedError( \
                "Operator %s not implemented function:compile" % self.op_name)

    def fuse_transpose(self, op, **kwargs):
        return op

    def calculate_ops(self, op, **kwargs):
        """ Calculate the amount of computations for operator.

            Returns the output size by default.
        """
        base_ops = kwargs.get('base_ops', 1)
        infer_shapes = kwargs['infer_shapes']
        count = sum(np.product(shp) for shp in infer_shapes[op.attr('name')])
        return count * base_ops

_tfm_manager = {}
def register_transformer(op_name):
    def wrapper(tfm):
        tfm.op_name = op_name
        if op_name in _tfm_manager:
            raise NameError("Transformer %s has been registered" % op_name)
        _tfm_manager[op_name] = tfm()

        rpass = [k for k, v in tfm.__dict__.items() \
                if not k.startswith("_") and callable(v)]
        for p in rpass:
            tfm = register_pass(p)(tfm)
        return tfm
    return wrapper

def get_transformer(op):
    op_name = op.attr('op_name')
    if op_name not in _tfm_manager:
        raise NotImplementedError( \
                "Transformer %s has not been registered" % op_name)
    return _tfm_manager[op_name]

_op_manager = {}
_pass_manager = {k:[] for k, v in Transformer.__dict__.items() \
        if not k.startswith("_") and callable(v)}
def register_pass(pass_t):
    def wrapper(tfm):
        if tfm.op_name not in _op_manager:
            _op_manager[tfm.op_name] = []
        if pass_t in _op_manager[tfm.op_name]:
            raise NameError( \
                    "Transformer %s pass:%s has been registered" \
                    % (tfm.op_name, pass_t))
            return tfm
        _op_manager[tfm.op_name].append(pass_t)
        if pass_t in _pass_manager:
            _pass_manager[pass_t].append(tfm.op_name)
        return tfm
    return wrapper

def pass_info(arg=None):
    if arg is None:
        return _pass_manager
    if isinstance(arg, mx.sym.Symbol):
        return _op_manager.get(arg.attr('op_name'), [])
    return _pass_manager.get(arg, [])

def apply_pass(pass_t, updates=[]):
    def wrapper(op, **kwargs):
        tfm = get_transformer(op)
        assert pass_t in pass_info(op), \
                "Transformer %s has not been registered pass:%s" \
                % (op.attr('op_name'), pass_t)
        ret = getattr(tfm, pass_t)(op, **kwargs)
        for n in updates:
            kwargs[n][ret.attr('name')] = kwargs[n][op.attr('name')]
        return ret
    return wrapper

# === symbol helper ===

def fuse_constant(symbol, params):
    def _impl(op, params, graph):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if is_var(op, params):
            pass
        elif childs is None:
            params[name] = get_nd_op(op_name)(**attr)
            op = mx.sym.var(name, shape=params[name].shape)
        elif all([is_params(c, params) for c in childs]):
            in_params = [params[c.attr('name')] for c in childs]
            params[name] = get_nd_op(op_name)(*in_params, **attr)
            op = mx.sym.var(name, shape=params[name].shape)
        return op
    return topo_visit_transformer(symbol, params, _impl)

def attach_input_shape(symbol, params, input_shape):
    def _impl(op, params, graph):
        if is_inputs(op, params):
            op = mx.sym.var(op.attr('name'), shape=input_shape)
        return op
    return topo_visit_transformer(symbol, params, _impl)

def infer_shape(symbol, params, input_shape=None):
    infer_shapes = {}
    def _impl(op, params, graph):
        name, op_name = op.attr('name'), op.attr('op_name')
        _, oshp, _ = op.infer_shape()

        if is_params(op, params):
            if oshp is None:
                oshp = [params[name].shape]
                op = mx.sym.var(name, shape=oshp[0])
            assert params[name].shape == oshp[0], \
                    "Parameter %s's shape %s is inconsistent with \
                    params dict %s" % (name, oshp[0], params[name].shape)
        elif is_inputs(op, params):
            if input_shape is None:
                assert oshp is not None, "It seems that graph doesn't set \
                        input_shape, please invoke attach_input_shape first."
            else:
                oshp = [input_shape]
                op = mx.sym.var(name, shape=oshp[0])
        infer_shapes[name] = oshp
        return op
    topo_visit_transformer(symbol, params, _impl)
    return infer_shapes

def _collect_attribute(op, **kwargs):
    attr_name, func = kwargs['attr_name'], kwargs['func']
    func(op.attr(attr_name))
    return op

def collect_op_names(symbol, params):
    op_names = set()
    _ = topo_visit_transformer(symbol, params, _collect_attribute,
            attr_name='op_name', func=op_names.add)
    return op_names

def requant_operator(X, def_prec, oscale=None, **kwargs):
    th_dict, precs = kwargs['th_dict'], kwargs['precs']
    scales, name = kwargs['scales'], kwargs['name']
    Xopn, xn = X.attr('op_name'), X.attr('name')
    X_name = _mrt._uniq_name(xn)
    oprec = precs[xn].get(name, _mrt.PREC(def_prec))
    exactly = True if oscale else False
    oscale = oscale if oscale else _mrt.scale(th_dict[xn], oprec.p)
    iscale = scales[xn]
    iprec = precs[xn][_mrt.out_key]


    print("tested here")
    exit()
    pass

# === symbol pass == 

def calculate_ops(symbol, params, normalize=True):
    ops, infer_shapes = [0], infer_shape(symbol, params)
    def _impl(op, **kwargs):
        ops[0] += apply_pass("calculate_ops")(op, **kwargs)
    topo_visit_transformer(symbol, params, _impl,
            infer_shapes=infer_shapes)

    ops = ops[0]
    if normalize:
        LEVELS = ['', 'K', 'M', 'G', 'T', 'P']
        idx = 0
        while ops > 1000:
            ops /= 1000
            idx += 1
        ops = "{:5.2f}{}".format(ops, LEVELS[idx])
    return ops

def fuse_transpose(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("fuse_transpose", updates=["infer_shapes"]),
            infer_shapes=infer_shapes)

def validate(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("validate"), infer_shapes=infer_shapes)

def rewrite(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("rewrite"), infer_shapes=infer_shapes)

def quantize(symbol, params, inputs_ext, calib_ctx, data):
    infer_shapes = infer_shape(symbol, params)
    mrt = _mrt.MRT(symbol, params, inputs_ext)      # initialize
    mrt.set_data('data', data)                      # set input data
    mrt.calibrate(ctx=calib_ctx)                    # calibration
    mrt.set_output_prec(8)                          # set output prec
    return topo_visit_transformer(symbol, params,
            apply_pass("quantize"), infer_shapes=infer_shapes,
            th_dict=mrt.th_dict, precs=mrt.precs, scales=mrt.scales,
            op_input_precs=mrt._op_input_precs)
