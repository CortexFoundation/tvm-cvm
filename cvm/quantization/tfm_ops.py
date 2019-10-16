from tfm_base import *
from sym_utils import *

import mxnet as mx
import nnvm

import numpy as np


@register_pass("validate")
@register_pass("fuse_transpose")
@register_transformer("null")
class Null(Transformer):
    def quantize(self, op, **kwargs):
        op_name, name = op.attr('op_name'), op.attr('name')
        params, th_dict = kwargs['params'], kwargs['th_dict']
        if is_inputs(op, params):
            precs, scales = kwargs['precs'], kwargs['scales']
            scales[name] = scale(th_dict[name], precs[name])
            attr = { 'precision': str(precs[name]) }
            return mx.sym.var(name, attr=attr)
        return op

    def compile(self, op, **kwargs):
        return nnvm.sym.Variable(op.attr('name'), op.list_attr())

    def calculate_ops(self, op, **kwargs):
        return 0


@register_pass("validate")
@register_pass("calculate_ops")
@register_transformer("transpose")
class Transpose(Transformer):
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
                op = mx.sym.transpose(op, axes=axes, name=name+"_fuse_tranpose")
        return op


@register_pass("validate")
@register_pass("calculate_ops")
@register_transformer("relu")
class Relu(Transformer):
    def fuse_transpose(self, op, **kwargs):
        X = op.get_children()[0]
        if X.attr('op_name') == Transpose.op_name:
            t_name, t_attr = X.attr('name'), X.list_attr()
            X = X.get_children()[0]
            op = mx.sym.relu(X)
            op = mx.sym.transpose(op, name=t_name, **t_attr)
        return op


@register_pass("fuse_transpose")
@register_transformer("Convolution")
class Convolution(Transformer):
    def validate(self, op, **kwargs):
        op = self._validate_layout(op)
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
        op = mx.sym.Convolution(X, W, **attr, name=op.attr('name'))
        B = mx.sym.reshape(B, (1, oshp[1], 1, 1))
        print (B.infer_shape())
        op = mx.sym.broadcast_add(op, B)
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
@register_pass("fuse_transpose")
@register_transformer("FullyConnected")
class FullyConnected(Transformer):
    def calculate_ops(self, op, **kwargs):
        W = sym_iter(op.get_children())[1]
        infer_shapes = kwargs['infer_shapes']
        W_shape = infer_shapes[W.attr('name')][get_entry_id(W)]
        kwargs['base_ops'] = np.product(W_shape[1:]) * 2
        if not get_attr(op.list_attr(), 'no_bias', False):
            kwargs['base_ops'] += 1
        return super().calculate_ops(op, **kwargs)


@register_pass("validate")
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
            scale_name = X_name + '_avg_scale'
            assert scale_name not in graph
            graph[scale_name] = scale_sym = mx.sym.var(scale_name, shape=(1,))
            params[scale_name] = nd.array([1. / (X_shape[2] * X_shape[3])])
            op = mx.sym.sum(childs[0], axis=(2, 3))
            op = mx.sym.broadcast_mul(op, scale_sym)
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
            conv_name = name.replace('pool', 'pool_conv')
            W_name = conv_name + '_weight'
            assert W_name not in graph
            W_shape = (in_channel, 1, *kernel)
            graph[W_name] = W = mx.sym.var(W_name, shape=W_shape)
            print(X_shape, W_shape)
            params[W_name] = nd.full(shape=W_shape, val=(1/np.product(kernel)))
            op = mx.sym.Convolution(X, W, **conv_attr, name=conv_name)
        else:
            assert pool_type == 'max', "Unsupported Pooling \
                    %s(%s, pool_type=%s)"%(Pooling.op_name, name, pool_type)
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


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("calculate_ops")
@register_transformer("broadcast_mul")
class BroadcastMul(Transformer):
    pass


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("calculate_ops")
@register_transformer("broadcast_add")
class BroadcastAdd(Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_transformer("Concat")
class Concat(Transformer):
    def fuse_transpose(self, op, **kwargs):
        name, childs = op.attr('name'), sym_iter(op.get_children())
        same, axeses = True, set()
        for X in childs:
            if X.attr('op_name') != Transpose.op_name:
                same = False
                break
            axeses.add(get_attr(X.list_attr(), 'axes'))
        if same and len(axeses) == 1:
            dim, axes = get_attr(op.list_attr(), 'dim'), list(axeses)[0]
            Xs = [X.get_children()[0] for X in childs]
            op = mx.sym.concat(*Xs, dim=axes[dim])
            op = mx.sym.transpose(op, axes=axes, name=name+'_fuse_transpose')
        return op


@register_transformer("sum")
class Sum(Transformer):
    def validate(self, op, **kwargs):
        X, attr = sym_iter(op.get_children())[0], op.list_attr()
        xshp = kwargs['infer_shapes'][X.attr('name')][get_entry_id(X)]
        axis = get_attr(attr, 'axis', [])
        # convert exclude into False
        if get_attr(attr, 'exclude', False):
            attr['axis'] = [i for i, _ in enumerate(xshp) if i not in axis]
            attr['exclude'] = False
            if len(attr['axis']) == 0:
                return X
        op = mx.sym.sum(X, **attr)
        return op

    def fuse_transpose(self, op, **kwargs):
        name, attr, X = op.attr('name'), op.list_attr(), op.get_children()[0]
        xshp = kwargs['infer_shapes'][X.attr('name')][get_entry_id(X)]
        axis = get_attr(attr, 'axis', [i for i in range(len(xshp))])
        keepdims = get_attr(attr, 'keepdims', False)
        if X.attr('op_name') == Transpose.op_name and not keepdims:
            axes, op = get_attr(X.list_attr(), 'axes'), X.get_children()[0]
            axis = [axes[i] for i in axis]
            op = mx.sym.sum(op, axis=axis, keepdims=keepdims)
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

            conv_name = name + "_conv"
            W_name = conv_name + '_weight'
            weight = params[cchilds[1].attr('name')]
            params[W_name] = weight * scale.reshape(*scale.shape, 1, 1, 1)
            W = mx.sym.var(W_name, shape=params[W_name].shape)

            B_name = conv_name + '_bias'
            if not get_attr(cattr, 'no_bias', False):
               bias += params[cchilds[2].attr('name')]
            params[B_name] = bias
            B = mx.sym.var(B_name, shape=bias.shape)

            cattr['no_bias'] = False
            op = mx.sym.Convolution(cchilds[0], W,
                   B, **cattr, name=conv_name)
        else:
            ishp = infer_shapes[X_name][get_entry_id(X)]
            reshp = [s if i==axis else i for i,s in enumerate(ishp)]
            w_name = name + "_weight"
            params[w_name] = scale.reshape(reshp)
            W = mx.sym.var(w_name, shape=reshp)
            node = mx.sym.broadcast_mul(X, W, name=name+"_mul")
            bias_name = name + "_bias"
            params[bias_name] = bias.reshape(reshp)
            B = mx.sym.var(bias_name, shape=reshp)
            op = mx.sym.broadcast_add(op, B, name=name+"_add")
        return op

    def calculate_ops(self, op, **kwargs):
        kwargs['base_ops'] = 4
        return super().calculate_ops(op, **kwargs)


