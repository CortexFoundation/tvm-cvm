from sym_utils import *

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
        childs = kwargs['childs']
        attrs = kwargs['attr']
        sym = get_nnvm_op(self.op_name)(*childs, name=N.n(),
                                        **attrs)
        return sym

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

def apply_pass(pass_t, **updates):
    def wrapper(op, **kwargs):
        tfm = get_transformer(op)
        assert pass_t in pass_info(op), \
                "Transformer %s has not been registered pass:%s" \
                % (op.attr('op_name'), pass_t)
        kwargs.update(updates)
        ret = getattr(tfm, pass_t)(op, **kwargs)
        for n in updates:
            kwargs[n][ret.attr('name')] = kwargs[n][op.attr('name')]
        return ret
    return wrapper

_NoneName = object()
class N(object):
    _global_name = _NoneName
    _count = {}
    _name_manager = {}

    @staticmethod
    def n(name=""):
        assert N._global_name != _NoneName, \
                "register name manager first please"
        if name not in N._name_manager:
            N._name_manager[name] = 0
        _n = "mrt"
        if N._global_name:
            _n += "_" + N._global_name
        if name != "":
            _n += "_" + name
        _n += "_%d" % N._name_manager[name]
        N._name_manager[name] += 1
        return _n

    @staticmethod
    def _set_global(name):
        if name not in N._count:
            N._count[name] = 0
        else:
            name += str(N._count[name])
        N._global_name = name

    @staticmethod
    def register_nm(name):
        def wrapper(pass_f):
            def run(symbol, params, *args, **kwargs):
                old_name = N._global_name
                N._set_global(name)
                N._count[name] += 1
                ret = pass_f(symbol, params, *args, **kwargs)
                N._global_name = old_name
                return ret
            return run
        return wrapper

# === symbol helper ===

@N.register_nm("gv")
def graph_validate(symbol, params):
    def _name_replace(op, params, graph):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if is_inputs(op, params):
            assert "data" not in graph, "multiple inputs"
            op = mx.sym.var("data", attr=attr)
        elif is_params(op, params):
            op = mx.sym.var(N.n("params"), attr=attr)
            params[op.attr('name')] = params[name]
        elif childs is None:
            op = get_mxnet_op(op_name)(name=N.n(op_name), **attr)
        else:
            op = get_mxnet_op(op_name)(*childs, name=N.n(op_name), **attr)
        return op
    sym, params = topo_visit_transformer(symbol, params, _name_replace)

    new_params = {s.attr('name'):params[s.attr('name')] \
            for s in topo_sort(sym) if is_params(s, params)}

    return sym, new_params

@N.register_nm("fc")
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

@N.register_nm("ais")
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

@N.register_nm("fuse_transpose")
def fuse_transpose(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("fuse_transpose", infer_shapes=infer_shapes))

@N.register_nm("sum")
def validate(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("validate", infer_shapes=infer_shapes))

@N.register_nm("rewrite")
def rewrite(symbol, params):
    infer_shapes = infer_shape(symbol, params)
    return topo_visit_transformer(symbol, params,
            apply_pass("rewrite", infer_shapes=infer_shapes))

@N.register_nm("cvm")
def compile(symbol, params):
    def _as_list(arr):
        return arr if isinstance(arr, list) else [arr]

    infer_shapes = infer_shape(symbol, params)
    graph = {}
    for op in topo_sort(symbol):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        childs = [] if childs is None else childs
        childs = [get_node(c, graph) for c in childs]
        childs = [x for y in childs for x in _as_list(y)]
        op = apply_pass("compile", infer_shapes=infer_shapes)(
                op, childs=childs, attr=attr)
        graph[name] = op

    nodes = []
    for sym in symbol:
        node = get_node(sym, graph)
        assert node is not None
        nodes.append(node)
    if len(nodes) > 1:
        return nnvm.sym.Group(nodes)
    return nodes[0], params

