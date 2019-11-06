from tfm_pass import *
from sym_utils import *

import mxnet as mx
import nnvm

import numpy as np


@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_transformer("null")
class Null(Transformer):
    def quantize(self, op, **kwargs):
        op_name, name = op.attr('op_name'), op.attr('name')
        params, th_dict = kwargs['params'], kwargs['th_dict']
        if is_inputs(op, params):
            precs, scales = kwargs['precs'], kwargs['scales']
            scales[name] = scale(th_dict[name], precs[name][OUT_KEY])
            attr = { 'precision': str(precs[name]) }
            return mx.sym.var(name, attr=attr)
        return op

    def compile(self, op, **kwargs):
        return nnvm.sym.Variable(op.attr('name'), **kwargs['attr'])

    def calculate_ops(self, op, **kwargs):
        return 0


@register_pass("rewrite")
@register_pass("calculate_ops")
@register_transformer("transpose")
class Transpose(Transformer):
    def validate(self, op, **kwargs):
        infer_shapes = kwargs['infer_shapes']
        shp = infer_shapes[op.attr('name')][get_entry_id(op)]
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if not get_attr(attr, 'axes', []):
            attr['axes'] = list(reversed(range(len(shp))))
            op = mx.sym.transpose(*childs, **attr)
        return op

    def fuse_transpose(self, op, **kwargs):
        name, attr = op.attr('name'), op.list_attr()
        axes = get_attr(attr, 'axes')
        X = sym_iter(op.get_children())[0]
        if X.attr('op_name') == Transpose.op_name:
            tname, tattr = X.attr('name'), X.list_attr()
            caxes = get_attr(tattr, 'axes')
            axes = [caxes[ii] for ii in axes]
            op = X.get_children()[0]
            if axes != sorted(axes):
                op = mx.sym.transpose(op, axes=axes, name=name)
        return op


@register_pass("validate")
@register_pass("rewrite")
@register_pass("calculate_ops")
@register_transformer("relu")
class Relu(Transformer):
    def fuse_transpose(self, op, **kwargs):
        X, name = op.get_children()[0], op.attr('name')
        if X.attr('op_name') == Transpose.op_name:
            t_name, t_attr = X.attr('name'), X.list_attr()
            X = X.get_children()[0]
            op = mx.sym.relu(X, name=name)
            op = mx.sym.transpose(op, name=t_name, **t_attr)
        return op

    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        sym = get_nnvm_op(self.op_name)(*childs, name=N.n('relu'))
        return sym


@register_transformer("Activation")
class Activation(Transformer):
    def validate(self, op, **kwargs):
        attr = op.list_attr()
        assert attr['act_type'] in [Relu.op_name], \
            "Only supported relu activation"
        return op

    def fuse_transpose(self, op, **kwargs):
        attr = op.list_attr()
        if attr['act_type'] == Relu.op_name:
            op = Relu().fuse_transpose(op, **kwargs)
        return op

    def rewrite(self, op, **kwargs):
        attr = op.list_attr()
        if attr['act_type'] == Relu.op_name:
            op = Relu().rewrite(op, **kwargs)
        return op

    def calculate_ops(self, op, **kwargs):
        attr = op.list_attr()
        if attr['act_type'] == Relu.op_name:
            op = Relu().calculate_ops(op, **kwargs)
        return op

    def compile(self, op, **kwargs):
        attrs = kwargs['attr']
        act_type = attrs['act_type']
        if act_type == Relu.op_name:
            sym = Relu().compile(op, **kwargs)
        return sym


@register_pass("fuse_transpose")
@register_transformer("Convolution")
class Convolution(Transformer):
    def validate(self, op, **kwargs):
        op = self._validate_layout(op)
        W = sym_iter(op.get_children())[1]
        W_shp = kwargs['infer_shapes'][W.attr('name')][get_entry_id(W)]
        I, KH, KW = W_shp[1:]
        assert I*KH*KW < 65536, "convolution ops overflow"
        return op

    def _validate_layout(self, op):
        name = op.attr('name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        X, W = childs[0], childs[1]
        X_name, W_name = X.attr('name'), W.attr('name')
        layout = get_attr(attr, 'layout', "NCHW")
        if layout == "NCW":
            no_bias = get_attr(attr, 'no_bias', False)
            dilate, kernel = get_attr(attr, 'dilate'), get_attr(attr, 'kernel')
            pad, stride = get_attr(attr, 'pad'), get_attr(attr, 'stride')
            num_filter = get_attr(attr, 'num_filter')
            num_group = get_attr(attr, 'num_group', 1)
            attr = {
                'layout': "NCHW", 'no_bias': no_bias,
                'dilate': (*dilate, 1), 'kernel': (*kernel, 1),
                'pad': (*pad, 0), 'stride': (*stride, 1),
                'num_filter': num_filter, 'num_group': num_group,
            }
            X = mx.sym.expand_dims(X, axis=3)
            params[W_name] = params[W_name].expand_dims(axis=3)
            W = graph[W_name] = mx.sym.var(W_name, shape=params[W_name].shape)
            B = None if no_bias else childs[2]
            op = get_mxnet_op(op_name)(X, W, B, **attr, name=name)
            op = mx.sym.squeeze(op, axis=3)
        else:
            assert layout == "NCHW", "Convolution(%s) only supported \
                    NCHW layout vs. %s" % (name, layout)
        return op

    def rewrite(self, op, **kwargs):
        #TODO: matrix decomposition
        # op = self._fuse_bias(op, kwargs["infer_shapes"])
        return op

    def _fuse_bias(self, op, infer_shapes):
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        if get_attr(attr, 'no_bias', False):
            return op

        attr['no_bias'] = True
        X, W, B = childs
        oshp = infer_shapes[op.attr('name')][0]
        op = mx.sym.Convolution(X, W, **attr, name=name)
        B = mx.sym.reshape(B, (1, oshp[1], 1, 1), name=N.n('reshape'))
        print (B.infer_shape())
        op = mx.sym.broadcast_add(op, B, name=N.n('broadcast_add'))
        return op

    def quantize(self, op, **kwargs):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs = sym_iter(op.get_children())
        cns = [c.attr('name') for c in childs] if childs else []
        def_prec = kwargs['op_input_precs'][op_name]
        X, xprec, xs = requant_operator(childs[0], name, def_prec, **kwargs)
        W, wprec, ws = requant_parameter(cns[1], def_prec)
        B, bprec = None, None
        if not get_attr(attr, 'no_bias', False):
            bs = ws * xs
            bias_prec = PREC(_get_bit(th_dict[cns[2]] * bs))
            B, bprec, _ = requant_parameter(cns[2], bias_prec, bs)
        oscale = scales[name] = ws * xs
        sym = get_mxnet_op(op_name)(X, W, B, **attr, name=name)
        precs[name][out_key] = PREC(get_bit(th_dict[name] * oscale))
        return sym

    def calculate_ops(self, op, **kwargs):
        W = sym_iter(op.get_children())[1]
        infer_shapes = kwargs['infer_shapes']
        W_shape = infer_shapes[W.attr('name')][get_entry_id(W)]
        kwargs['base_ops'] = np.product(W_shape[1:]) * 2
        if not get_attr(op.list_attr(), 'no_bias', False):
            kwargs['base_ops'] += 1
        return super().calculate_ops(op, **kwargs)

    def compile(self, op, **kwargs):
        op.attr('name'), op.attr('op_name')
        op.get_children(), op.list_attr()
        childs = kwargs['childs']
        attrs = kwargs['attr']
        kernel = get_attr(attrs, 'kernel')
        layout = get_attr(attrs, 'layout', 'NCHW')
        kernel_layout = get_attr(attrs, 'kernel_layout', 'OIHW')
        op_name, new_attrs = 'conv2d', {}
        new_attrs['channels'] = get_attr(attrs, 'num_filter')
        new_attrs['kernel_size'] = kernel
        new_attrs['strides'] = get_attr(attrs, 'stride', (1, 1))
        new_attrs['padding'] = get_attr(attrs, 'pad', (0, 0))
        new_attrs['dilation'] = get_attr(attrs, 'dilate', (1, 1))
        new_attrs['groups'] = get_attr(attrs, 'num_group', 1)
        new_attrs['layout'] = layout
        new_attrs['kernel_layout'] = kernel_layout
        new_attrs['use_bias'] = not get_attr(attrs, 'no_bias', False)
        return get_nnvm_op(op_name)(*childs, name=N.n('convolution'),
                                    **new_attrs)


@register_pass("validate")
@register_pass("fuse_transpose")
@register_transformer("FullyConnected")
class FullyConnected(Transformer):
    def rewrite(self, op, **kwargs):
        infer_shapes, params = kwargs['infer_shapes'], kwargs['params']
        op = self._matrix_decomposition(op, params, infer_shapes)
        return op


    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name, new_attrs = 'dense', {}
        new_attrs['units'] = get_attr(attrs, 'num_hidden')
        new_attrs['use_bias'] = not get_attr(attrs, 'no_bias', False)
        try:
            mx.sym.FullyConnected(mx.sym.var('x'), num_hidden=1, flatten=True)
            has_flatten = True
        except mx.base.MXNetError:
            has_flatten = False
        use_flatten = get_attr(attrs, 'flatten', True)
        if has_flatten and use_flatten:
            childs[0] = nnvm.sym.flatten(childs[0], name=N.n('flatten'))
        return get_nnvm_op(op_name)(*childs, name=N.n('fullyconnected'),
                                    **new_attrs)

    def _matrix_decomposition(self, op, params, infer_shapes):
        name, attr = op.attr('name'), op.list_attr()
        childs = sym_iter(op.get_children())
        X, W = childs[:2]

        MATRIX_MAXIMUM_SIZE = 65536
        C = infer_shapes[W.attr('name')][get_entry_id(W)][1]
        if C <= MATRIX_MAXIMUM_SIZE:
            return op

        if X.attr('op_name') != Flatten.op_name:
            X = mx.sym.flatten(X, name=N.n('flatten'))

        no_bias = get_attr(attr, 'no_bias', False)
        attr['no_bias'] = True

        # matrix decomposition
        # Y = B + X*W^T = B + X1*W1^T + X2*W2^T + ...
        # Wi.shape = (num_hidden, step), W = [W1, W2, ...]
        # Xi.shape = (batch_size, step), X = [X1, X2, ...]
        nodes, step, start = [], MATRIX_MAXIMUM_SIZE, 0
        wgt = params[W.attr('name')]
        while start < C:
            stop = min(start+step, C)
            Xk = mx.sym.slice_axis(X, axis=1,
                    begin=start, end=stop, name=N.n("slice_axis"))
            Wk_name = N.n('slice_axis')
            params[Wk_name] = wgt.slice_axis(axis=1, begin=start, end=stop)
            Wk = mx.sym.var(Wk_name, shape=params[Wk_name].shape)
            tmp = mx.sym.FullyConnected(Xk, Wk, name=N.n("dense"), **attr)
            nodes.append(tmp)
            start += step

        while len(nodes) > 1:
            a, b = nodes.pop(0), nodes.pop(0)
            tmp = mx.sym.elemwise_add(a, b, name=N.n("elemwise_add"))
            nodes.append(tmp)

        op = nodes[0]
        if not no_bias:
            op = mx.sym.broadcast_add(op, childs[2],
                    name=N.n('broadcast_add'))

        return op

    def calculate_ops(self, op, **kwargs):
        W = sym_iter(op.get_children())[1]
        infer_shapes = kwargs['infer_shapes']
        W_shape = infer_shapes[W.attr('name')][get_entry_id(W)]
        kwargs['base_ops'] = np.product(W_shape[1:]) * 2
        if not get_attr(op.list_attr(), 'no_bias', False):
            kwargs['base_ops'] += 1
        return super().calculate_ops(op, **kwargs)


@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_transformer("softmax")
class Softmax(Transformer):
    def calculate_ops(self, op, **kwargs):
        infer_shapes = kwargs['infer_shapes']
        X = sym_iter(op.get_children())[0]
        xshp = infer_shapes[X.attr('name')][get_entry_id(X)]
        axis = get_attr(op.list_attr(), 'axis', -1)
        kwargs['base_ops'] = 2 + 2 * xshp[axis]
        return super().calculate_ops(op, **kwargs)


@register_pass("fuse_transpose")
@register_transformer("Pooling")
class Pooling(Transformer):
    def validate(self, op, **kwargs):
        name, attr = op.attr('name'), op.list_attr()
        layout = get_attr(attr, 'layout', 'NCHW')
        assert layout == 'NCHW'
        pool_type = get_attr(attr, 'pool_type', 'max')
        assert pool_type in ['max', 'avg'], \
            "Pooling(%s) only supported type for max and avg." % name
        count_include_pad = get_attr(attr, 'count_include_pad', True)
        assert get_attr(attr, 'count_include_pad', True), \
            "Pooling(%s) only supported count_include_pad for True." % name

        global_pool = get_attr(attr, 'global_pool', False)
        pooling_convention = get_attr(attr, 'pooling_convention', 'valid')
        assert pooling_convention == 'valid' or global_pool, \
            "Pooling(%s) only supported convention for valid." % name

        return op

    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        kernel = get_attr(attrs, 'kernel')
        global_pool = 'global' if get_attr(attrs, 'global_pool', False) else ''
        pool_type = attrs['pool_type']
        op_name = '_'.join([global_pool, pool_type, 'pool2d']).strip('_')
        new_attrs = {}
        if not global_pool:
            new_attrs['pool_size'] = kernel
            new_attrs['strides'] = get_attr(attrs, 'stride', (1, 1))
            new_attrs['padding'] = get_attr(attrs, 'pad', (0, 0))
            new_attrs['ceil_mode'] = (get_attr(attrs, 'pooling_convention',
                        'valid') == 'full')
            if pool_type == 'avg':
                new_attrs['count_include_pad'] = \
                        get_attr(attrs, 'count_include_pad', True)
        return get_nnvm_op(op_name)(*childs, name=N.n('pooling'),
                                    **new_attrs)

    def rewrite(self, op, **kwargs):
        params, graph = kwargs['params'], kwargs['graph']
        infer_shapes = kwargs['infer_shapes']
        name, attr = op.attr('name'), op.list_attr()
        childs = sym_iter(op.get_children())
        pool_type = get_attr(attr, 'pool_type', 'max')
        is_global = get_attr(attr, 'global_pool', False)
        if pool_type == 'avg' and is_global:
            X = childs[0]
            X_name = X.attr('name')
            X_shape = infer_shapes[X_name][get_entry_id(X)]
            scale_name = N.n('avg_scale')
            graph[scale_name] = scale_sym = mx.sym.var(scale_name, shape=(1,))
            params[scale_name] = nd.array([1. / (X_shape[2] * X_shape[3])])
            op = mx.sym.sum(childs[0], axis=(2, 3), name=N.n('sum'))
            op = mx.sym.broadcast_mul(op, scale_sym, name=N.n('braodcast_mul'))
        elif pool_type == 'avg':
            X = childs[0]
            X_shape = infer_shapes[X.attr('name')][get_entry_id(X)]
            in_channel = X_shape[1]
            kernel = get_attr(attr, 'kernel')
            if isinstance(kernel, int):
                kernel = (kernel, kernel)
            conv_attr = {
                'no_bias': 'True',
                'dilate': '(1, 1)',
                'kernel': kernel,
                'stride': attr['stride'],
                'pad': attr['pad'],
                'layout': 'NCHW',
                'num_filter': in_channel,
                'num_group': in_channel,
            }
            conv_name = N.n('pool_conv')
            W_name = N.n('weight')
            W_shape = (in_channel, 1, *kernel)
            graph[W_name] = W = mx.sym.var(W_name, shape=W_shape)
            params[W_name] = nd.full(shape=W_shape, val=(1/np.product(kernel)))
            op = mx.sym.Convolution(X, W, **conv_attr, name=conv_name)
        return op


    def calculate_ops(self, op, **kwargs):
        X, attr = sym_iter(op.get_children())[0], op.list_attr()
        pool_type = get_attr(attr, 'pool_type', 'max')
        infer_shapes = kwargs['infer_shapes']
        if get_attr(attr, 'global_pool', False):
            _, _, K1, K2 = infer_shapes[X.attr('name')][get_entry_id(X)]
        else:
            K1, K2 = get_attr(attr, 'kernel')
        kwargs['base_ops'] = K1 * K2
        if pool_type == 'avg':
            kwargs['base_ops'] += 1
        return super().calculate_ops(op, **kwargs)

@register_pass("compile")
@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_pass("calculate_ops")
@register_transformer("broadcast_mul")
class BroadcastMul(Transformer):
    pass

@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_pass("calculate_ops")
@register_transformer("broadcast_add")
class BroadcastAdd(Transformer):
    pass


@register_pass("validate")
@register_pass("rewrite")
@register_pass("calculate_ops")
@register_transformer("Concat")
class Concat(Transformer):
    def fuse_transpose(self, op, **kwargs):
        name, childs = op.attr('name'), sym_iter(op.get_children())
        if any([c.attr('op_name') != Transpose.op_name for c in childs]):
            return op
        axeses = [tuple(get_attr(c.list_attr(), 'axes')) for c in childs]
        axeses = set([axes for axes in axeses])
        if len(axeses) == 1:
            dim = get_attr(op.list_attr(), 'dim')
            axes = get_attr(childs[0].list_attr(), 'axes')
            Xs = [X.get_children()[0] for X in childs]
            op = mx.sym.concat(*Xs, dim=axes[dim], name=name)
            op = mx.sym.transpose(op, axes=axes, name=N.n('fuse_transpose'))
        return op


@register_pass('compile')
@register_pass("rewrite")
@register_transformer("sum")
class Sum(Transformer):
    def validate(self, op, **kwargs):
        X, attr = sym_iter(op.get_children())[0], op.list_attr()
        name = op.attr('name')
        xshp = kwargs['infer_shapes'][X.attr('name')][get_entry_id(X)]
        axis = get_attr(attr, 'axis', [])
        # convert exclude into False
        if get_attr(attr, 'exclude', False):
            attr['axis'] = [i for i, _ in enumerate(xshp) if i not in axis]
            attr['exclude'] = False
            if len(attr['axis']) == 0:
                return X
        op = mx.sym.sum(X, **attr, name=name)
        return op

    def fuse_transpose(self, op, **kwargs):
        name, attr, X = op.attr('name'), op.list_attr(), op.get_children()[0]
        xshp = kwargs['infer_shapes'][X.attr('name')][get_entry_id(X)]
        axis = get_attr(attr, 'axis', [i for i in range(len(xshp))])
        keepdims = get_attr(attr, 'keepdims', False)
        if X.attr('op_name') == Transpose.op_name and not keepdims:
            axes, op = get_attr(X.list_attr(), 'axes'), X.get_children()[0]
            axis = [axes[i] for i in axis]
            op = mx.sym.sum(op, axis=axis, keepdims=keepdims, name=name)
        return op

    def calculate_ops(self, op, **kwargs):
        infer_shapes = kwargs['infer_shapes']
        oshp = infer_shapes[op.attr('name')][get_entry_id(op)]
        X = sym_iter(op.get_children())[0]
        ishp = infer_shapes[X.attr('name')][get_entry_id(X)]
        kwargs['base_ops'] = np.product(oshp) / np.product(ishp)
        return super().calculate_ops(op, **kwargs)


@register_pass("validate")
@register_pass("fuse_transpose")
@register_transformer("BatchNorm")
class BatchNorm(Transformer):
    def rewrite(self, op, **kwargs):
        params, infer_shapes = kwargs["params"], kwargs["infer_shapes"]
        name = op.attr('name')
        childs, attr= sym_iter(op.get_children()), op.list_attr()
        X, X_name = childs[0], childs[0].attr('name')
        gamma = params[childs[1].attr('name')]
        beta = params[childs[2].attr('name')]
        data_mean = params[childs[3].attr('name')]
        data_var = params[childs[4].attr('name')]

        fix_gamma = get_attr(attr, 'fix_gamma', True)
        gamma = 1 if fix_gamma else gamma
        axis = get_attr(attr, 'axis', 1)

        epsilon = float(attr['eps']) if 'eps' in attr else 1e-5
        scale = gamma / nd.sqrt(data_var + epsilon)
        bias = beta - scale * data_mean

        if X.attr('op_name') == 'Convolution':
            # Since convolution is "NCHW" format, axis must be one
            assert axis == 1, "Channel in input must be axis 1"
            cchilds, cattr = sym_iter(X.get_children()), X.list_attr()

            conv_name = X.attr('name')
            W_name = cchilds[1].attr('name')
            weight = params[W_name]
            params[W_name] = weight * scale.reshape(*scale.shape, 1, 1, 1)
            W = mx.sym.var(W_name, shape=params[W_name].shape)

            B_name = N.n('bias')
            if not get_attr(cattr, 'no_bias', False):
               B_name = cchilds[2].attr('name')
               bias += params[B_name]
            params[B_name] = bias
            B = mx.sym.var(B_name, shape=bias.shape)

            cattr['no_bias'] = False
            op = mx.sym.Convolution(cchilds[0], W,
                   B, **cattr, name=conv_name)
        else:
            ishp = infer_shapes[X_name][get_entry_id(X)]
            reshp = [s if i==axis else 1 for i,s in enumerate(ishp)]
            w_name = N.n('weight')
            params[w_name] = scale.reshape(reshp)
            W = mx.sym.var(w_name, shape=reshp)
            node = mx.sym.broadcast_mul(X, W, name=N.n("broadcast_mul"))
            bias_name = N.n('bias')
            params[bias_name] = bias.reshape(reshp)
            B = mx.sym.var(bias_name, shape=reshp)
            op = mx.sym.broadcast_add(op, B, name=N.n("broadcast_add"))
        return op

    def calculate_ops(self, op, **kwargs):
        kwargs['base_ops'] = 4
        return super().calculate_ops(op, **kwargs)


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("calculate_ops")
@register_transformer("Flatten")
class Flatten(Transformer):
    pass


@register_pass("validate")
@register_transformer("slice")
class Slice(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        begin = get_attr(attrs, 'begin', None)
        end = get_attr(attrs, 'end', None)
        stride = get_attr(attrs, 'step', None)
        new_attrs = {'begin': begin, 'end': end}
        if stride is not None:
            new_attrs['stride'] = stride
        return get_nnvm_op('strided_slice')(childs[0],
                name=N.n('slice'), **new_attrs)


@register_pass("validate")
@register_transformer("Reshape")
class Reshape(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name = 'reshape'
        new_attrs = {}
        new_attrs['shape'] = get_attr(attrs, 'shape', 'reshape')
        return get_nnvm_op(op_name)(*childs,
                name=N.n('reshape'), **new_attrs)


@register_transformer("Custom")
class Custom(Transformer):
    def validate(self, op, **kwargs):
        attr = op.list_attr()
        op_type = attr['op_type']
        assert op_type in ['cvm_clip', 'cvm_left_shift',
                        'cvm_right_shift', 'cvm_lut']
        return op

    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attr = kwargs['attr']
        op_type = attr['op_type']
        new_attrs = {}
        if op_type == 'cvm_clip':
            new_attrs['precision'] = attr['precision']
            sym = get_nnvm_op(op_type)(*childs, name=N.n('cvm_clip'),
                                        **new_attrs)
        elif op_type == 'cvm_lut':
            new_attrs['in_dim'] = attr.get['in_dim']
            sym = get_nnvm_op(op_type)(*childs, name=N.n('cvm_lut'),
                                        **new_attrs)
        else:
            new_attrs['precision'] = attr['precision']
            new_attrs['shift_bit'] = attr['shift_bit']
            sym = get_nnvm_op(op_type)(*childs, name=N.n('cvm_shift'),
                                         **new_attrs)
        return sym


@register_pass("compile")
@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_transformer("elemwise_add")
class ElemwiseAdd(Transformer):
    def fuse_transpose(self, op, **kwargs):
        return _ft_multi_input(op)

def _ft_multi_input(op):
    name, childs = op.attr('name'), sym_iter(op.get_children())
    # Assert all the inputs are transpose
    if any([c.attr('op_name') != Transpose.op_name for c in childs]):
        return op
    # Check all the inputs shapes are consistent
    axeses = [tuple(get_attr(c.list_attr(), 'axes')) for c in childs]
    axeses = set([axes for axes in axeses])
    # Fuse transpose
    if len(axeses) == 1:
        axes = get_attr(childs[0].list_attr(), 'axes')
        Xs = [X.get_children()[0] for X in childs]
        opname = op.attr('op_name')
        op = get_mxnet_op(opname)(*Xs, name=name)
        op = mx.sym.transpose(op, axes=axes, name=N.n('fuse_transpose'))
    return op
