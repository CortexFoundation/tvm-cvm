from tfm_pass import *
from sym_utils import *

from mxnet import ndarray as nd
import mxnet as mx
import nnvm

import numpy as np


@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("null")
class Null(Transformer):
    def quantize(self, op, **kwargs):
        if is_inputs(op, kwargs['params']):
            name, attr = op.attr('name'), op.list_attr()
            prec = kwargs['precs'][name][OUT_KEY]
            kwargs['scales'][name] = scale(kwargs['th_dict'][name], prec)
            extra_attr = { 'precision': str(prec) }
            return mx.sym.var(name, **attr, attr=extra_attr)
        return op

    def compile(self, op, **kwargs):
        return nnvm.sym.Variable(op.attr('name'), **kwargs['attr'])

    def calculate_ops(self, op, **kwargs):
        return 0


@register_pass("rewrite")
@register_pass("quantize")
@register_pass("calculate_ops")
@register_pass('compile')
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
@register_pass("quantize")
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


@register_pass("calculate_ops")
@register_transformer("LeakyReLU")
class LeakyReLU(Transformer):
    def validate(self, op, **kwargs):
        name, attr = op.attr('name'), op.list_attr()
        act = get_attr(attr, 'act_type', 'leaky')
        assert act == 'leaky', "Unsupported LeakyReLU %s for act_type: %s" \
                % (name, act)
        return op

    def fuse_transpose(self, op, **kwargs):
        X, name = op.get_children()[0], op.attr('name')
        if X.attr('op_name') == Transpose.op_name:
            t_name, t_attr = X.attr('name'), X.list_attr()
            X = X.get_children()[0]
            op = mx.sym.relu(X, name=name)
            op = mx.sym.transpose(op, name=t_name, **t_attr)
        return op

    def rewrite(self, op, **kwargs):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()

        slope = get_attr(attr, 'slope', 0.25)
        X = childs[0]
        posi_X = mx.sym.relu(X)
        nega_X = mx.sym.negative(X)
        nega_X = mx.sym.relu(nega_X)
        slope_name = N.n('slope')
        kwargs['params'][slope_name] = nd.array([slope])
        kwargs['graph'][slope_name] = slope_sym = \
                mx.sym.var(slope_name, shape=(1,))
        scale_X = mx.sym.broadcast_mul(nega_X, slope_sym)
        op = posi_X - scale_X
        return op


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_transformer("_mul_scalar")
class MulScalar(Transformer):
    def rewrite(self, op, **kwargs):
        params, graph = kwargs['params'], kwargs['graph']
        infer_shapes = kwargs['infer_shapes']
        name = op.attr('name')
        scalar = get_attr(op.list_attr(), 'scalar')
        if scalar == 0:
            shp = infer_shapes[name][get_entry_id(op)]
            params[name] = nd.zeros(shp)
            op = mx.sym.var(name, shape=shp)
        else:
            X = op.get_children()[0]
            sname = N.n('scalar')
            params[sname] = nd.array([scalar])
            graph[sname] = mx.sym.var(sname, shape=(1,))
            op = mx.sym.broadcast_mul(X, graph[sname], name=name)
        return op


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_transformer("_div_scalar")
class DivScalar(Transformer):
    def rewrite(self, op, **kwargs):
        graph = kwargs['graph']
        name = op.attr('name')
        attr, childs = op.list_attr(), sym_iter(op.get_children())

        scalar = get_attr(attr, 'scalar')
        sname = N.n('scalar')
        kwargs['params'][sname] = nd.array([1/scalar])
        graph[sname] = mx.sym.var(sname, shape=(1,))
        return mx.sym.broadcast_mul(childs[0], graph[sname], name=name)


@register_pass("quantize")
@register_pass("prepare_for_compile")
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
@register_pass("prepare_for_compile")
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
        # return _restore(op, **kwargs)
        return _quantize_xwb(op, **kwargs)

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
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_transformer('expand_dims')
class ExpandDims(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name, new_attrs = 'expand_dims', {}
        new_attrs['axis'] = get_attr(attrs, 'axis', 'expand_dims')
        return get_nnvm_op(op_name)(*childs, **new_attrs)


@register_transformer('Embedding')
class Embedding(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        indices, weight = childs
        op_name = 'take'
        return get_nnvm_op(op_name)(weight, indices, axis=0)


@register_pass("validate")
@register_pass("rewrite")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("quantize")
@register_transformer("repeat")
class Repeat(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        data = childs[0]
        new_attrs = {}
        op_name = 'repeat'
        new_attrs['repeats'] = get_attr(attrs, 'repeats', 'repeat')
        if 'axis' in attrs:
            new_attrs['axis'] = get_attr(attrs, 'axis')
        else:
            data = get_nnvm_op('flatten')(data)
            new_attrs['axis'] = 0
        return get_nnvm_op(op_name)(childs[0], **new_attrs)


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_transformer('_contrib_box_nms')
class BoxNms(Transformer):
    def prepare_for_compile(self, op, **kwargs):
        attrs = op.list_attr()
        iou_thresh = get_attr(attrs, ' overlap_thresh', 0.5) * 100
        iou_thresh = int(iou_thresh)
        attrs['overlap_thresh'] = iou_thresh
        assert attrs['valid_thresh'] == int(attrs['valid_thresh'])
        attrs['valid_thresh'] = int(attrs['valid_thresh'])
        return get_mxnet_op(self.op_name)(
                sym_iter(op.get_children()), **attrs, name=op.attr('name'))

    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        force_suppress = get_attr(attrs, 'force_suppress', False)
        iou_thresh = get_attr(attrs, ' overlap_thresh')
        top_k = get_attr(attrs, 'topk', -1)
        valid_thresh = get_attr(attrs, 'valid_thresh', 0)
        coord_start = get_attr(attrs, 'coord_start', 2)
        score_index = get_attr(attrs, 'score_index', 1)
        id_index = get_attr(attrs, 'id_index', -1)
        in_format = get_attr(attrs, 'in_format', 'corner')
        out_format = get_attr(attrs, 'out_format', 'corner')
        op_name = 'get_valid_counts'
        ret = get_nnvm_op(op_name)(childs[0],
                score_threshold=valid_thresh)
        op_name = 'non_max_suppression'
        nms_out = get_nnvm_op(op_name)(ret[1], ret[0],
                iou_threshold=iou_thresh,
                force_suppress=force_suppress, top_k=top_k,
                coord_start=coord_start,
                score_index=score_index, id_index=id_index,
                return_indices=False, invalid_to_bottom=True)
        return nms_out


@register_pass("validate")
@register_pass("rewrite")
@register_pass("calculate_ops")
@register_pass("quantize")
@register_pass("fuse_transpose")
@register_transformer('slice_like')
class SliceLike(Transformer):
    # def quantize(self, op, **kwargs):
    #     th_dict, scales = kwargs['th_dict'], kwargs['scales']
    #     name, op_name = op.attr('name'), op.attr('op_name')
    #     childs, attr = sym_iter(op.get_children()), op.list_attr()
    #     cns = [c.attr('name') for c in childs] if childs else []

    #     oprec = kwargs['op_input_precs'][op_name]
    #     X, _, xs = requant(childs[0], oprec, oname=name, **kwargs)
    #     oscale = scales[name] = xs
    #     op = get_mxnet_op(op_name)(X, childs[1], **attr, name=name)
    #     kwargs['precs'][name][OUT_KEY] = get_bit(th_dict[name] * oscale)

    #     logger = logging.getLogger('log.mrt.realize')
    #     logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
    #            op_name, name, scales[name], cns)
    #     op = requant_output(op, name, **kwargs)
    #     return op

    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        new_attrs = {'axis': get_attr(attrs, 'axes', ())}
        op_name = 'slice_like'
        return get_nnvm_op(op_name)(*childs, **new_attrs)


@register_pass("validate")
@register_pass("rewrite")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("quantize")
@register_transformer('slice_axis')
class SliceAxis(Transformer):
    def prepare_for_compile(self, op, **kwargs):
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        X = childs[0]
        cshape = infer_shapes[X.attr('name')]
        axis = get_attr(attr, 'axis')
        axis_begin = get_attr(attr, 'begin')
        axis_end = get_attr(attr, 'end', None)
        axis_end = cshape[axis]
        begin = [0 for s in cshape]
        end = [s for s in cshape]
        begin[axis], end[axis] = axis_begin, axis_end
        return get_mxnet_op('slice')(X, begin=begin, end=end, name=name)


@register_transformer('UpSampling')
class UpSampling(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        scale = get_attr(attrs, 'scale')
        op_name, new_attrs = 'upsampling', {'scale': int(scale)}
        return get_nnvm_op(op_name)(childs[0], **new_attrs)


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("prepare_for_compile")
@register_transformer("FullyConnected")
class FullyConnected(Transformer):
    def rewrite(self, op, **kwargs):
        infer_shapes, params = kwargs['infer_shapes'], kwargs['params']
        op = self._matrix_decomposition(op, params, infer_shapes)
        return op

    def quantize(self, op, **kwargs):
        # return _restore(op, **kwargs)
        return _quantize_xwb(op, **kwargs)

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


@register_pass("calculate_ops")
@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_transformer("sigmoid")
class Sigmoid(Transformer):
    def quantize(self, op, **kwargs):
        # return _restore(op, **kwargs)
        return _quantize_table(op, **kwargs)


@register_pass("calculate_ops")
@register_pass("validate")
@register_pass("rewrite")
@register_pass("fuse_transpose")
@register_transformer("exp")
class Exp(Transformer):
    def quantize(self, op, **kwargs):
        # return _restore(op, **kwargs)
        return _quantize_table(op, **kwargs)


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

    def quantize(self, op, **kwargs):
        params, graph = kwargs['params'], kwargs['graph']
        scales = kwargs['scales']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []

        oprec = kwargs['op_input_precs'][op_name]
        xs = scale(kwargs['th_dict'][childs[0].attr('name')], oprec)
        X, xprec, xs = requant_operator(childs[0], oprec, xs,
                oname=name, **kwargs)
        axis = get_attr(attr, 'axis', -1)
        lambd = 10
        alpha = int(lambd*xs)
        var = mx_const(alpha, graph, params)
        max_axis = mx.sym.max(X, axis=axis, keepdims=True)
        offset = mx.sym.broadcast_sub(max_axis, var, name=N.n('softmax_offset'))
        offset = realize(offset, 0, xprec)
        norm = mx.sym.broadcast_sub(X, offset, name=N.n('softmax_normalize'))
        norm = mx.sym.relu(norm, name=N.n('Softmax_filter'))
        norm = realize(norm, 0, xprec)

        data = nd.arange(0, alpha+1)
        table = nd.exp(data/xs)

        tprec = get_bit(math.exp(lambd))
        table = nd.clip(table, a_min=0, a_max=get_range(tprec))
        W_name = N.n('cvm_lut_weight')
        params[W_name] = weight = table.round().reshape(alpha+1, 1)
        wattr = { 'precision': str(tprec) }
        W = graph[W_name] = mx.sym.var(W_name, shape=weight.shape, attr=wattr)
        lut = mx.sym.Custom(norm, W, in_dim=alpha+1,
                name=name, op_type='cvm_lut')
        sum_lut = mx.sym.sum(lut, axis=axis, keepdims=True,
                name=N.n("softmax_sum"))

        oprec = min(15, 31 - tprec)
        assert oprec > 8, "operator softmax(%s) lambda(%d) is too large" \
                % (name, lambd)
        oscale = get_range(oprec)
        var_scale = mx_const(oscale, graph, params)
        prob = mx.sym.broadcast_mul(lut, var_scale,
                name=N.n("softmax_output_scale"))
        var_one = mx_const(1, graph, params)
        half_lut = realize(sum_lut, 1, 31)
        prob = mx.sym.broadcast_add(prob, half_lut, name=N.n("softmax_round"))
        op = mx.sym.broadcast_div(prob, sum_lut, name=N.n("softmax_prob"))
        op = op.astype('int32').astype('float32')
        # op = mx.sym.floor(op) # simulate integer division
        op = realize(op, 0, oprec)
        kwargs['precs'][name][OUT_KEY] = oprec
        scales[name]= oscale

        logger = logging.getLogger('log.mrt.realize')
        logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
               op_name, name, scales[name], cns)
        op = requant_output(op, name, **kwargs)
        return op


@register_pass("fuse_transpose")
@register_pass("quantize")
@register_pass("prepare_for_compile")
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
        pool_type = get_attr(attrs, 'pool_size', 'max')
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
            op = mx.sym.sum(childs[0], axis=(2, 3), name=N.n('sum'), keepdims=True)
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


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("compile")
@register_pass("prepare_for_compile")
@register_transformer("broadcast_mul")
class BroadcastMul(Transformer):
    def quantize(self, op, **kwargs):
        # return _restore(op, **kwargs)
        scales = kwargs['scales']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []

        oprec = kwargs['op_input_precs'][op_name]
        X, xprec, xs = requant(childs[0], oprec, oname=name, **kwargs)
        B, bprec, bs = requant(childs[1], oprec, oname=name, **kwargs)
        oscale = scales[name] = xs * bs
        op = get_mxnet_op(op_name)(X, B, **attr, name=name)
        kwargs['precs'][name][OUT_KEY] = get_bit(kwargs['th_dict'][name]*oscale)

        logger = logging.getLogger('log.mrt.realize')
        logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
               op_name, name, scales[name], cns)
        op = requant_output(op, name, **kwargs)
        return op


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("compile")
@register_transformer("broadcast_add")
class BroadcastAdd(Transformer):
    def quantize(self, op, **kwargs):
        # return _restore(op, **kwargs)
        return _quantize_scale(op, **kwargs)


@register_pass("compile")
@register_transformer("broadcast_div")
class BroadcastDiv(Transformer):
    pass


@register_pass("compile")
@register_transformer("broadcast_sub")
class BroadcastSub(Transformer):
    pass


@register_pass("compile")
@register_transformer("broadcast_to")
class BroadcastTo(Transformer):
    pass


@register_pass("compile")
@register_transformer("broadcast_greater")
class BroadcastGreater(Transformer):
    pass


@register_pass("rewrite")
@register_pass("validate")
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

    def quantize(self, op, **kwargs):
        # return _restore(op, **kwargs)
        return _quantize_scale(op, **kwargs)

    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name = 'concatenate'
        new_attrs = {'axis': get_attr(attrs, 'dim', 1)}
        return get_nnvm_op(op_name)(*childs,
                name=N.n('concat'), **new_attrs)


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

    def quantize(self, op, **kwargs):
        scales = kwargs['scales']
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        cns = [c.attr('name') for c in childs] if childs else []

        oprec = kwargs['op_input_precs'][op_name]
        X, xprec, xs = requant_operator(childs[0], oprec, oname=name, **kwargs)
        oscale = scales[name] = xs
        op = get_mxnet_op(op_name)(X, **attr, name=name)
        kwargs['precs'][name][OUT_KEY] = get_bit(kwargs['th_dict'][name]*oscale)

        logger = logging.getLogger('log.mrt.realize')
        logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
               op_name, name, scales[name], cns)
        op = requant_output(op, name, **kwargs)
        return op


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
            op = mx.sym.broadcast_add(node, B, name=N.n("broadcast_add"))
        return op

    def calculate_ops(self, op, **kwargs):
        kwargs['base_ops'] = 4
        return super().calculate_ops(op, **kwargs)


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("calculate_ops")
@register_pass("prepare_for_compile")
@register_transformer("Flatten")
class Flatten(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name = 'flatten'
        sym = get_nnvm_op(op_name)(*childs, name=N.n(),
                                        **attrs)
        return sym


@register_transformer('floor')
class Floor(Transformer):
    def prepare_for_compile(self, op, **kwargs):
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        node = childs[0]
        return node


@register_transformer('ceil')
class Ceil(Transformer):
    def prepare_for_compile(self, op, **kwargs):
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        node = childs[0]
        return node


@register_transformer('round')
class Round(Transformer):
    def prepare_for_compile(self, op, **kwargs):
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        node = childs[0]
        return node


@register_transformer('fix')
class Fix(Transformer):
    def prepare_for_compile(self, op, **kwargs):
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        node = childs[0]
        return node


@register_transformer('Cast')
class Cast(Transformer):
    def prepare_for_compile(self, op, **kwargs):
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        node = childs[0]
        return node


@register_pass("validate")
@register_transformer("slice")
class Slice(Transformer):
    def prepare_for_compile(self, op, **kwargs):
        childs, attr = sym_iter(op.get_children()), op.list_attr()
        X = childs[0]
        cshape = infer_shapes[X.attr('name')]
        begin = get_attr(attr, 'begin')
        end = get_attr(attr, 'end')
        begin = [0 if s is None else s for s in begin]
        end = [cshape[i] if s is None else s for i,s in enumerate(end)]
        return get_mxnet_op('slice')(X, begin=begin, end=end, name=name)

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
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
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


@register_pass("prepare_for_compile")
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
            new_attrs['in_dim'] = attr['in_dim']
            sym = get_nnvm_op(op_type)(*childs, name=N.n('cvm_lut'),
                                        **new_attrs)
        else:
            new_attrs['precision'] = attr['precision']
            new_attrs['shift_bit'] = attr['shift_bit']
            sym = get_nnvm_op(op_type)(*childs, name=N.n('cvm_shift'),
                                         **new_attrs)
        return sym


@register_pass("validate")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("calculate_ops")
@register_transformer("clip")
class Clip(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name, new_attrs = 'clip', {}
        new_attrs['a_min'] = get_attr(attrs, 'a_min')
        new_attrs['a_max'] = get_attr(attrs, 'a_max')
        return get_nnvm_op(op_name)(*childs, **new_attrs)


@register_transformer('_minimum')
class Minimum(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name = 'broadcast_min'
        return get_nnvm_op(op_name)(*childs,
                name=N.n('_minimum'), **attrs)


@register_transformer('_maximum')
class Maximum(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name = 'broadcast_max'
        return get_nnvm_op(op_name)(*childs,
                name=N.n('_maximum'), **attrs)


@register_pass('compile')
@register_transformer('max')
class Max(Transformer):
    pass


@register_pass('compile')
@register_transformer('min')
class Min(Transformer):
    pass


@register_transformer('argmax')
class Argmax(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name = 'argmax'
        new_attrs = {}
        new_attrs['axis'] = get_attr(attrs, 'axis', 0)
        new_attrs['keepdims'] = get_attr(attrs, 'keepdims', False)
        return get_nnvm_op(op_name)(*childs,
                name=N.n('_argmax'), **new_attrs)


@register_transformer('argmin')
class Argmax(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        attrs = kwargs['attr']
        op_name = 'argmin'
        new_attrs = {}
        new_attrs['axis'] = get_attr(attrs, 'axis', 0)
        new_attrs['keepdims'] = get_attr(attrs, 'keepdims', False)
        return get_nnvm_op(op_name)(*childs,
                name=N.n('_argmin'), **new_attrs)


@register_pass("compile")
@register_transformer("abs")
class Abs(Transformer):
    pass


@register_pass("compile")
@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_transformer("elemwise_add")
class ElemwiseAdd(Transformer):
    def fuse_transpose(self, op, **kwargs):
        return _ft_multi_input(op)

    def quantize(self, op, **kwargs):
        # return _restore(op, **kwargs)
        return _quantize_scale(op, **kwargs)


@register_pass("compile")
@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_transformer("elemwise_sub")
class ElemwiseSub(Transformer):
    def fuse_transpose(self, op, **kwargs):
        return _ft_multi_input(op)

    def quantize(self, op, **kwargs):
        # return _restore(op, **kwargs)
        return _quantize_scale(op, **kwargs)


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("quantize")
@register_pass("prepare_for_compile")
@register_transformer("Dropout")
class Dropout(Transformer):
    def compile(self, op, **kwargs):
        childs = kwargs['childs']
        return childs[0]

    def fuse_transpose(self, op, **kwargs):
        X, name = op.get_children()[0], op.attr('name')
        op_name = op.attr('name')
        if X.attr('op_name') == Transpose.op_name:
            t_name, t_attr = X.attr('name'), X.list_attr()
            X = X.get_children()[0]
            op = get_mxnet_op(op_name)(X, name=name)
            op = mx.sym.transpose(op, name=t_name, **t_attr)
        return op


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("rewrite")
@register_pass("quantize")
@register_transformer("_arange")
class Arange(Transformer):
    pass


@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_transformer("tile")
class Tile(Transformer):
    pass

@register_pass("validate")
@register_pass("calculate_ops")
@register_pass("fuse_transpose")
@register_pass("rewrite")
@register_pass("quantize")
@register_transformer("negative")
class Negative(Transformer):
    pass


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

def _quantize_scale(op, **kwargs):
    scales = kwargs['scales']
    th_dict, precs = kwargs['th_dict'], kwargs['precs']
    name, op_name = op.attr('name'), op.attr('op_name')
    attr, childs = op.list_attr(), sym_iter(op.get_children())
    cns = [c.attr('name') for c in childs] if childs else []

    oprec = kwargs['op_input_precs'][op_name]
    in_th = max([th_dict[n] for n in cns])
    oscale = scales[name] = scale(in_th, oprec)
    new_childs = []
    for c in childs:
        c, cprec, _ = requant(c, oprec, oscale=oscale, oname=name, **kwargs)
        new_childs.append(c)
    op = get_mxnet_op(op_name)(*new_childs, **attr, name=name)
    precs[name][OUT_KEY] = get_bit(th_dict[name] * oscale)

    logger = logging.getLogger('log.mrt.realize')
    logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
           op_name, name, scales[name], cns)
    op = requant_output(op, name, **kwargs)
    return op

def _quantize_xwb(op, **kwargs):
    th_dict, scales = kwargs['th_dict'], kwargs['scales']
    name, op_name = op.attr('name'), op.attr('op_name')
    childs, attr = sym_iter(op.get_children()), op.list_attr()
    cns = [c.attr('name') for c in childs] if childs else []

    oprec = kwargs['op_input_precs'][op_name]
    X, xprec, xs = requant_operator(childs[0], oprec, oname=name, **kwargs)
    W, wprec, ws = requant_parameter(cns[1], oprec, oname=name, **kwargs)
    B, bprec = None, None
    if not get_attr(attr, 'no_bias', False):
        bs = ws * xs
        bias_prec = get_bit(th_dict[cns[2]] * bs)
        B, bprec, _ = requant_parameter(cns[2], bias_prec, bs,
            oname=name, **kwargs)
    oscale = scales[name] = ws * xs
    op = get_mxnet_op(op_name)(X, W, B, **attr, name=name)
    kwargs['precs'][name][OUT_KEY] = get_bit(th_dict[name] * oscale)

    logger = logging.getLogger('log.mrt.realize')
    logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
           op_name, name, scales[name], cns)
    op = requant_output(op, name, **kwargs)
    return op

def _restore(op, **kwargs):
    params, graph = kwargs['params'], kwargs['graph']
    th_dict, precs, scales = kwargs['th_dict'], kwargs['precs'], kwargs['scales']
    name, op_name = op.attr('name'), op.attr('op_name')
    childs, attr = sym_iter(op.get_children()), op.list_attr()
    cns = [c.attr('name') for c in childs] if childs else []

    childs = [] if childs is None else childs
    new_childs = [c / scales[c.attr('name')] \
        if scales.get(c.attr('name'), 1) != 1 else c \
                 for c in childs]

    out = get_mxnet_op(op_name)(*new_childs, **attr, name=name)

    oprec = precs[name].get(OUT_KEY, 16)
    oscale = scales[name] = 1
    oscale = scales[name] = scale(th_dict[name], oprec)
    out = (out * oscale)
    precs[name][OUT_KEY] = oprec

    out = requant_output(out, name, **kwargs)
    return out

def _quantize_table(op, **kwargs):
    params, graph = kwargs['params'], kwargs['graph']
    th_dict, precs, scales = kwargs['th_dict'], kwargs['precs'], kwargs['scales']
    name, op_name = op.attr('name'), op.attr('op_name')
    childs, attr = sym_iter(op.get_children()), op.list_attr()
    cns = [c.attr('name') for c in childs] if childs else []

    iprec = kwargs['op_input_precs'][op_name]
    xs = scale(th_dict[cns[0]], iprec)
    # xs= scales[cns[0]]
    X, xprec, xs = requant_operator(childs[0], iprec, \
            oscale=xs, oname=name, **kwargs)
    alpha = get_range(xprec)
    var = mx_const(alpha, graph, params)
    X = mx.sym.broadcast_add(X, var, name=N.n(op_name+'_offset'))

    out = get_nd_op(op_name)(nd.arange(-alpha, alpha+1) / xs)
    oprec = precs[name].get(OUT_KEY, 16)
    oscale = scales[name] = scale(out.abs().max().asscalar(), oprec)

    W_name = N.n("cvm_lut_weight")
    params[W_name] = weight = (out * oscale).round().reshape(2*alpha+1, 1)
    wattr = { 'precision': str(oprec)}
    W = graph[W_name] = mx.sym.var(W_name, shape=weight.shape, attr=wattr)
    op = mx.sym.Custom(X, W, in_dim=2*alpha+1, name=name, op_type='cvm_lut')
    precs[name][OUT_KEY] = oprec

    logger = logging.getLogger('log.mrt.realize')
    logger.debug("operator  %-20s name=%-40s oscale=%s, iscale=%s",
           op_name, name, scales[name], cns)
    op = requant_output(op, name, **kwargs)
    return op

