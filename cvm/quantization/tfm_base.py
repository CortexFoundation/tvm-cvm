from sym_utils import *

import numpy as np

class Transformer(object):
    op_name = "none"

    def __init__(self):
        if self.op_name == "none":
            raise RuntimeError("Base transformer should not be instantiated")

    def _error(self, fname):
        raise NotImplementedError( \
                "Operator %s not implemented function:%s" \
                % (self.op_name, fname))

    def _pass(self, op):
        return op

    def rewrite(self, op, **kwargs):
        return self._error("rewrite")

    def quantize(self, op, **kwargs):
        return self._error("quantize")

    def compile(self, op, **kwargs):
        return self._error("compile")

    def fuse_transpose(self, op, **kwargs):
        return self._pass(op)

    def calculate_ops(self, op, **kwargs):
        """ Calculate the compute times of operator
            and returns the op's ops.
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
            raise NameError("Operator %s has been registered" % op_name)
        _tfm_manager[op_name] = tfm()
        return tfm
    return wrapper

def get_transformer(op):
    op_name = op.attr('op_name')
    if op_name not in _tfm_manager:
        raise NotImplementedError( \
                "Operator %s has not been registered" % op_name)
    return _tfm_manager[op_name]

# symbol pass

def fuse_constant(symbol, params):
    def _impl(op, params, graph):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if is_var(op, params):
            pass
        elif childs is None:
            params[name] = get_nd_op(op_name)(**attr)
            op = mx.sym.var(name, shape=params[name].shape)
        else:
            flag = all([is_params(c, params) for c in childs])
            if flag:
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

def calculate_ops(symbol, params, normalize=True):
    ops, infer_shapes = [0], infer_shape(symbol, params)
    def _impl(op, **kwargs):
        ops[0] += get_transformer(op).calculate_ops(op, **kwargs)
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
    def _impl(op, **kwargs):
        return get_transformer(op).fuse_transpose(op, **kwargs)
    return topo_visit_transformer(symbol, params, _impl)

def rewrite(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    def _impl(op, **kwargs):
        return get_transformer(op).rewrite(op, **kwargs)
    return topo_visit_transformer(symbol, params, _impl,
            infer_shapes=infer_shapes)
