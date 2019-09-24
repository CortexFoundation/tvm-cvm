from tensorflow_parser import TFParser
from tensorflow.core.framework import tensor_pb2 as tpb2
from tensorflow.core.framework import tensor_shape_pb2 as tspb2
from tensorflow.core.framework import attr_value_pb2 as apb2
from tensorflow.python.framework import dtypes
import mxnet as mx
from mxnet import nd

import os
import logging
import utils

# import heapq
import sym_utils as sutils

ts = set()
tl = []

def argmin(data, axis=None, keepdims=False, exclude=False):
    """Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a argmin operation is performed.
        The default, axis=None, will find the indices of minimum element all of the elements of
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.argmin(data, axis, keepdims, exclude)


def argmax(data, axis=None, keepdims=False, exclude=False):
    """Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    data : relay.Expr
        The input data

    axis : None or int or tuple of int
        Axis or axes along which a argmax operation is performed.
        The default, axis=None, will find the indices of the maximum element of the elements of
        the input array. If axis is negative it counts from the last to the first axis.

    keepdims : bool
        If this is set to True, the axes which are reduced are left in the result as dimensions
        with size one.
        With this option, the result will broadcast correctly against the input array.

    exclude : bool
        If `exclude` is true, reduction will be performed on the axes that are
        NOT in axis instead.

    Returns
    -------
    result : relay.Expr
        The computed result.
    """
    axis = [axis] if isinstance(axis, int) else axis
    return _make.argmax(data, axis, keepdims, exclude)

def _reduce_all():
    def _impl(inputs, attr, params):
        axis = params.pop(inputs[1].name_hint).asnumpy()
        axis = tuple(axis)
        return AttrCvt(
            op_name='all',
            extras={'axis': axis},
            transforms={'keep_dims':'keepdims'},
            ignores=['name', 'Tidx'])([inputs[0]], attr)
    return _impl

def _elemwise(name):
    def _impl(inputs, attr, *args):
        assert len(inputs) == 2, "{} take 2 inputs, {} given".format(name, len(inputs))
        return sutils.get_mxnet_op(name)(*inputs)
    return _impl

def _argx(func, func_name):
    """ A common wrapper for argmin and argmax operations """
    def _impl(inputs, attr, params):
        try:
            # In Tensorflow, `axis` argument is a Tensor, not attribute. We
            # support the case where it inputs from a scalar constant.
            axis_input_name = inputs[1].name_hint
            axis_input_vlaue = [params[axis_input_name].asnumpy()[0]]
        except (IndexError, KeyError):
            raise TypeError( \
                "Unsupported argument for `{}` : `axis` should be a constant".format(func_name))
        return func(inputs[0], axis=axis_input_vlaue, keepdims=False)
    return _impl

def _pooling(name):
    def _impl(inputs, attr, params):

        attr['data_format'] = attr['data_format'].decode("utf-8")
        flip_layout = False

        input_shape = attr['_input_shapes'][inputs[0]]

        if attr['data_format'] == 'NHWC':
            attr['kernel_shape'] = (attr['ksize'][1], attr['ksize'][2])
            attr['strides'] = (attr['strides'][1], attr['strides'][2])
        elif attr['data_format'] == 'NCHW':
            attr['kernel_shape'] = (attr['ksize'][2], attr['ksize'][3])
            attr['strides'] = (attr['strides'][2], attr['strides'][3])
        else:
            msg = 'Value {} of attribute "data_format" of operator Pooling ' \
                  'is not valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attrs['data_format']))

        if attr['_target_layout'] == "NCHW" and attr['data_format'] == "NHWC":
            tmp_shape = attr['_input_shapes'][inputs[0]]
            input_shape = [tmp_shape[ii] for ii in (0, 3, 1, 2)]
            inputs[0] = _op.transpose(inputs[0], axes=(0, 3, 1, 2))
            attr['data_format'] = "NCHW"
            flip_layout = True

        # Fix padding
        attr['padding'] = attr['padding'].decode("utf-8")

        if attr['padding'] == 'VALID':
            attr['padding'] = [0, 0]
        elif attr['padding'] == 'SAME':
            stride_h, stride_w = attr['strides']
            kernel_h, kernel_w = attr['kernel_shape']
            if attr['data_format'] == 'NHWC':
                in_h = input_shape[1]
                in_w = input_shape[2]
            else:
                in_h = input_shape[2]
                in_w = input_shape[3]

            pad_v = _get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, kernel_w, stride_w)

            attr['padding'] = [pad_v[0], pad_h[0], pad_v[1], pad_h[1]]
        else:
            msg = 'Value {} in attribute "padding" of operator Pooling is ' \
                  'not valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attr['padding']))

        if name == "avg_pool":
            attr['count_include_pad'] = False

        out = AttrCvt(
            op_name=_dimension_picker(name),
            transforms={
                'kernel_shape':'pool_size',
                'data_format':'layout'},
            ignores=['ksize'],
            extras={'ceil_mode': False},
            custom_check=_dimension_constraint())(inputs, attr)

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 1))

        return out
    return _impl

def _batch_norm():
    def _impl(inputs, attr, params):
        # Rearrange inputs from
        # (data, moving_mean, moving_variance, beta, gamma)
        #     to
        # (data, gamma, beta, moving_mean, moving_var)
        new_inputs = [inputs[0], inputs[4], inputs[3], inputs[1], inputs[2]]

        axis = 3
        if 'data_format' in attr:
            attr['data_format'] = attr['data_format'].decode("utf-8")
            if attr['data_format'] == 'NCHW':
                axis = 1

        return AttrCvt(
            op_name='batch_norm',
            transforms={'scale_after_normalization':'scale', 'variance_epsilon':'epsilon'},
            extras={'axis': axis},
            ignores=['data_format'],
            disables=['momentum'])(new_inputs, attr)
    return _impl

def _batch_to_space_nd():
    def _impl(inputs, attr, params):
        input_node = inputs[0]
        input_shape = attr['_input_shapes'][input_node]
        block_shape = params.pop(inputs[1].name_hint).asnumpy().tolist()
        crops = params.pop(inputs[2].name_hint).asnumpy().tolist()
        M = len(block_shape)
        batch = input_shape[0]
        # From https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-to-space-n-d:
        # Reshape input to reshaped of shape:
        # [block_shape[0], ..., block_shape[M-1], batch / prod(block_shape),
        #  input_shape[1], ..., input_shape[N-1]]
        shape1 = block_shape + [batch // np.prod(block_shape)] + input_shape[1:]
        reshaped = tvm.relay.reshape(input_node, newshape=shape1)
        # Permute dimensions of reshaped to produce permuted of shape
        # [batch / prod(block_shape), input_shape[1], block_shape[0], ...,
        # input_shape[M], block_shape[M-1], input_shape[M+1], ..., input_shape[N-1]]
        axes = [M] + [axis for i in range(M) for axis in [M + i + 1, i]] + \
            list(range(2 * M + 1, len(shape1)))
        permuted = tvm.relay.transpose(reshaped, axes=axes)
        # Reshape permuted to produce reshaped_permuted of shape
        # [batch / prod(block_shape), input_shape[1] * block_shape[0], ...,
        #  input_shape[M] * block_shape[M-1], input_shape[M+1], ..., input_shape[N-1]]
        shape2 = [0] + [-3] * M + [-2]
        reshaped_permuted = tvm.relay.reshape(permuted, newshape=shape2)
        # Crop the start and end of dimensions [1, ..., M] of reshaped_permuted according to crops
        # to produce the output of shape:
        # [batch / prod(block_shape), input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1],
        #  ..., input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],
        #  input_shape[M+1], ..., input_shape[N-1]]
        reshaped_permuted_shape = _infer_out_shapes(reshaped_permuted, params)[0]
        cropped = reshaped_permuted
        for axis in range(1, M+1):
            crop = crops[axis - 1]
            if crop != [0, 0]:
                indices = tvm.relay.arange(
                    crop[0],
                    reshaped_permuted_shape[axis] - crop[1],
                    dtype='int32'
                )
                cropped = tvm.relay.take(cropped, indices=indices, axis=axis)

        return cropped

    return _impl

def _bias_add():
    def _impl(inputs, attr, params):
        # Must expand for proper broadcasting in NCHW.
        if attr['data_format'].decode("utf-8") == 'NCHW':
            bias = _op.reshape(inputs[1], newshape=(1, -1, 1, 1))
        else:
            bias = inputs[1]
        return _op.add(inputs[0], bias)
    return _impl


def _broadcast_to():
    def _impl(inputs, attr, params):
        if isinstance(inputs[1], _expr.Var):
            shape = params[inputs[1].name_hint]
        else:
            shape = _infer_value(inputs[1], params)
        shape = list(shape.asnumpy().reshape([-1]))
        return _op.broadcast_to(inputs[0], shape)
    return _impl

def _cast():
    def _impl(inputs, attr, params):
        return inputs[0].astype(attr['DstT'].name)
    return _impl

class AttrCvt(object):
    """Common attribute conveter. An AttrConverter instance is a callable:
    ```
    attr_converter = AttrConverter(op_name, transforms={'a':'b', 'c':('d', 1)})
    new_op_name, new_attr = attr_converter(attrs)
    ```

    Parameters
    ----------
    op_name : str or callable
        If set as str, returned operator name is the str.
        If set as callable, returned operator is the str returned by calling:
        `op_name = func(attr)`
    transforms : dict of `new_name, or (new_name, default_value, transform function)`
        If only a new_name is provided, it's like renaming the attribute name.
        If default_value if provded, then the attribute is considered as optional.
        If transform function is provided, the original attribute value is handled
        by transform function.
    excludes : list
        A list of excluded attributes that should `NOT` appear.
        Raise NotImplementedError if occured.
    disables : list
        A list of attributes that is disabled in relay. Log warnings.
    ignores : list
        A list of attributes that is ignored in relay. Debug level logging.
    extras : dict
        A series of additional attributes should be added anyway to the returned
        attribute dict.
    custom_check : callable
        A custom function takes attribute, and return True/False.
        Raise RuntimeError if not bool(True) returned.
    """

    def __init__(self, op_name, transforms=None,
                 excludes=None, disables=None, ignores=None,
                 extras=None, custom_check=None):
        self._op_name = op_name
        self._transforms = transforms if transforms else {}
        self._excludes = excludes if excludes else []
        self._disables = disables if disables else []
        self._ignores = ignores if ignores else []
        self._extras = extras if extras else {}
        self._custom_check = custom_check

    def __call__(self, inputs, attrs, *args):
        self._ignores.append('_output_shapes')
        self._ignores.append('_input_shapes')
        self._ignores.append('T')
        self._ignores.append('use_cudnn_on_gpu')
        self._ignores.append('_node_name')
        self._ignores.append('is_training')
        self._ignores.append('_target_layout')

        # apply custom check
        if self._custom_check:
            func, msg = self._custom_check
            if not func(attrs):
                raise RuntimeError("Check failed: {}".format(msg))
        # get new op_name
        if isinstance(self._op_name, str):
            op_name = self._op_name
        else:
            assert callable(self._op_name), "op_name can either be string or callable"
            op_name = self._op_name(attrs)
        # convert attributes
        new_attrs = {}
        for k in attrs.keys():
            if k in self._excludes:
                raise tvm.error.OpAttributeUnimplemented(
                    'Attribute {} in operator {} is not supported.'.format(k, op_name))
            elif k in self._disables:
                logging.warning("Attribute %s is disabled in relay.%s", k, op_name)
            elif k in self._ignores:
                logging.debug("Attribute %s is ignored in relay.%s", k, op_name)
            elif k in self._transforms:
                new_name, defaults, transform = self._parse_default(self._transforms[k])
                if defaults is None:
                    new_attr = self._required_attr(attrs, k)
                else:
                    new_attr = attrs.get(k, None)
                if new_attr is None:
                    new_attrs[new_name] = defaults
                else:
                    new_attrs[new_name] = transform(new_attr)
            else:
                # copy
                new_attrs[k] = attrs[k]
        # add extras
        new_attrs.update(self._extras)
        return _get_relay_op(op_name)(*inputs, **new_attrs)

    def _parse_default(self, target):
        """Helper function to parse default values."""
        if not isinstance(target, (list, tuple)):
            k, v, t = target, None, lambda x: x
        elif len(target) == 1:
            k, v, t = target[0], None, lambda x: x
        elif len(target) == 2:
            k, v, t = target[0], target[1], lambda x: x
        elif len(target) > 2:
            k, v, t = target[0], target[1], target[2]
        else:
            k = None  # should raise
        if not isinstance(k, str):
            msg = "{} is not a valid target, (name, default) expected.".format(target)
            raise ValueError(msg)
        return k, v, t

    def _parse_bool(self, value):
        """Helper function to parse default boolean values."""
        if isinstance(value, str):
            return value.strip().lower() in ['true', '1', 't', 'y', 'yes']
        return bool(value)

    def _required_attr(self, attr, key):
        """Wrapper for getting required attributes."""
        assert isinstance(attr, dict)
        if key not in attr:
            raise tvm.error.OpAttributeRequired(
                'Attribute {} not found in operator {}'.format(key, self._op_name))
        return attr[key]

def _check_numerics():
    def _impl(inputs, attr, params):
        # Making a copy node assuming no need to verify
        return AttrCvt(op_name="copy", ignores=['message'])(inputs, attr)
    return _impl

def _concat():
    def _impl(inputs, attr, params):
        pop_node = inputs.pop(0)
        axis = params[pop_node.name_hint]
        params.pop(pop_node.name_hint)
        return AttrCvt(
            op_name="concatenate", ignores=['N'],
            extras={'axis': int(axis.asnumpy()[0])})([inputs], attr)
    return _impl

def _concatV2():
    def _impl(inputs, attr, params):
        pop_node = inputs.pop(len(inputs)-1)
        axis = params[pop_node.name_hint]
        params.pop(pop_node.name_hint)
        return AttrCvt(
            op_name="concatenate", ignores=['T', 'N', 'Tidx'],
            extras={'axis': int(axis.asnumpy()[0])})([inputs], attr)
    return _impl

def _conv(opname):
    def _impl(inputs, attr, params):
        attr['data_format'] = attr['data_format'].decode("utf-8")
        flip_layout = False

        # NCHW Layout require weights transpose
        if attr['data_format'] == 'NCHW':
            tmp_shape = attr['_input_shapes'][inputs[1]]
            tmp_shape = [tmp_shape[ii] for ii in (3, 2, 0, 1)]
            inputs[1] = _op.transpose(inputs[1], axes=(3, 2, 0, 1))
            attr['_input_shapes'][inputs[1]] = tmp_shape

        input_shape = attr['_input_shapes'][inputs[0]]
        weights_shape = attr['_input_shapes'][inputs[1]]

        if attr['_target_layout'] == "NCHW" and attr['data_format'] == "NHWC":
            input_shape = [input_shape[ii] for ii in (0, 3, 1, 2)]
            inputs[0] = _op.transpose(inputs[0], axes=(0, 3, 1, 2))
            if opname == 'conv':
                weights_shape = [weights_shape[ii] for ii in (3, 2, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(3, 2, 0, 1))
            else:
                weights_shape = [weights_shape[ii] for ii in (2, 3, 0, 1)]
                inputs[1] = _op.transpose(inputs[1], axes=(2, 3, 0, 1))

            attr['data_format'] = "NCHW"
            attr['strides'] = [attr['strides'][ii] for ii in (0, 3, 1, 2)]
            flip_layout = True

        if attr['data_format'] == 'NHWC':
            kernel_h, kernel_w, _, depth_mult = weights_shape
            attr['kernel_shape'] = (weights_shape[0], weights_shape[1])
            if opname == 'conv':
                attr['channels'] = weights_shape[3]
            else:
                attr['channels'] = input_shape[3] * depth_mult

            if 'dilations' in attr:
                attr['dilations'] = (attr['dilations'][1], attr['dilations'][2])
            attr['strides'] = (attr['strides'][1], attr['strides'][2])
        elif attr['data_format'] == 'NCHW':
            depth_mult, _, kernel_h, kernel_w = weights_shape
            attr['kernel_shape'] = (weights_shape[2], weights_shape[3])
            if opname == 'conv':
                attr['channels'] = weights_shape[0]
            else:
                attr['channels'] = input_shape[0] * depth_mult
                if attr['channels'] < 0:
                    attr['channels'] *= -1

            if 'dilations' in attr:
                attr['dilations'] = (attr['dilations'][2], attr['dilations'][3])
            attr['strides'] = (attr['strides'][2], attr['strides'][3])
        else:
            msg = 'Value {} in attribute "data_format" of operator Conv is ' \
                  'not valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attr['data_format']))


        if opname == 'depthwise':
            attr['groups'] = attr['channels']

        # Fix padding
        attr['padding'] = attr['padding'].decode("utf-8")

        if attr['padding'] == 'VALID':
            attr['padding'] = [0, 0]
        elif attr['padding'] == 'SAME':
            stride_h, stride_w = attr['strides']
            kernel_h, kernel_w = attr['kernel_shape']
            if attr['data_format'] == 'NHWC':
                in_h = input_shape[1]
                in_w = input_shape[2]
            else:
                in_h = input_shape[2]
                in_w = input_shape[3]

            dilation_h = attr['dilations'][0]
            dilation_w = attr['dilations'][1]
            dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
            dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
            pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w)


            if attr['data_format'] == 'NHWC':
                inputs[0] = _op.nn.pad(data=inputs[0],
                                       pad_width=((0, 0),
                                                  (pad_v[0], pad_v[1]),
                                                  (pad_h[0], pad_h[1]),
                                                  (0, 0)))
            else:
                inputs[0] = _op.nn.pad(data=inputs[0],
                                       pad_width=((0, 0),
                                                  (0, 0),
                                                  (pad_v[0], pad_v[1]),
                                                  (pad_h[0], pad_h[1])))

            attr['padding'] = [0, 0]

        else:
            msg = 'Value {} in attribute "padding" of operator Conv is not ' \
                  'valid.'
            raise tvm.error.OpAttributeInvalid(msg.format(attr['padding']))

        if 'kernel_layout' not in attr:
            if opname == 'conv':
                attr['kernel_layout'] = 'HWIO' if attr['data_format'] == 'NHWC' else 'OIHW'
            else:
                attr['kernel_layout'] = 'HWOI' if attr['data_format'] == 'NHWC' else 'OIHW'

        use_bias = len(inputs) == 3
        channel_axis = 1 if attr['data_format'] == "NCHW" else 3

        out = AttrCvt(
            op_name=_dimension_picker('conv'),
            transforms={
                'kernel_shape': 'kernel_size',
                'data_format': 'data_layout',
                'dilations': ('dilation', (0, 0)),
                'group': ('groups', 1)},
            custom_check=_dimension_constraint())([inputs[0], inputs[1]], attr)

        if use_bias:
            out = _op.nn.bias_add(out, inputs[2], axis=channel_axis)

        if flip_layout:
            out = _op.transpose(out, axes=(0, 2, 3, 1))

        return out
    return _impl

def _decode_image():
    def _impl(inputs, attr, params):
        # Image decode wrapper: Expecting user to feed decoded input to next layer drop this layer.
        warnings.warn("DecodeJpeg: It's a pass through, please handle preprocessing before input")
        return inputs[0]
    return _impl

def _depth_to_space():
    def _impl(inputs, attr, params):
        # Need to handle data layouts differently.
        input_shape = attr['_input_shapes'][inputs[0]]
        block_size = int(attr['block_size'])
        if attr['data_format'].decode("utf-8") == 'NHWC':
            in_n, in_h, in_w, in_c = input_shape
            new_c = int(in_c / (block_size * block_size))

            # First expand input to larger dimension.
            expanded = _op.reshape(
                inputs[0], newshape=(in_n, in_h, in_w, block_size, block_size, new_c))
            # Now reorder to expand spatial blocks.
            transposed = _op.transpose(expanded, axes=(0, 1, 3, 2, 4, 5))
            # Finally reshape to proper output.
            new_h = in_h * block_size
            new_w = in_w * block_size
            newshape = (in_n, new_h, new_w, new_c)

        else: # Handle NCHW layout
            in_n, in_c, in_h, in_w = input_shape
            new_c = int(in_c / (block_size * block_size))

            expanded = _op.reshape(
                inputs[0], newshape=(in_n, block_size, block_size, new_c, in_h, in_w))
            transposed = _op.transpose(expanded, axes=(0, 3, 4, 1, 5, 2))
            new_h = in_h * block_size
            new_w = in_w * block_size
            newshape = (in_n, new_c, new_h, new_w)

        return AttrCvt(
            op_name="reshape",
            extras={'newshape': newshape},
            ignores=['data_format', 'block_size'])([transposed], attr)

    return _impl

def _broadcast(name):
    def _impl(inputs, attr, params):
        return AttrCvt(
            op_name=name,
            ignores=['name', 'Tidx']
        )(inputs, attr)
    return _impl

def _elu():
    def _impl(inputs, attr, params):
        alpha = tvm.relay.const(-1.0, attr['T'].name)
        return alpha * _op.nn.relu(tvm.relay.const(1, attr['T'].name) \
                                   - _op.exp(inputs[0])) + _op.nn.relu(inputs[0])
    return _impl

def _expand_dims():
    def _impl(inputs, attr, params):
        dim_input = inputs.pop(1)
        axis = params.pop(_get_name_hint(dim_input)).asnumpy()[0]
        return AttrCvt(op_name="expand_dims", ignores=['Tdim', 'N'],
                       extras={'axis': int(axis), 'num_newaxis': 1})(inputs, attr)
    return _impl

def _fill():
    def _impl(inputs, attr, params):
        output_shape = attr['_output_shapes'][0]
        # Output shape must be defined to avoid errors. If any axis is not, we must
        # try to compute its shape.
        if -1 in output_shape:
            output_shape = _infer_value(inputs[0], params).asnumpy().reshape([-1]).tolist()

        fill_arg = params.pop(inputs.pop(1).name_hint)
        return _op.full(tvm.relay.const(fill_arg.asnumpy()[0], attr['T'].name),
                        output_shape, attr['T'].name)
    return _impl

def _fused_batch_norm():
    def _impl(inputs, attr, params):
        # Tensorflow: (data, gamma, beta, moving_mean, moving_variance)
        # Relay:       (data, gamma, beta, moving_mean, moving_varience)
        axis = 3
        need_cast = False

        if 'data_format' in attr:
            attr['data_format'] = attr['data_format'].decode("utf-8")
            if attr['data_format'] == 'NCHW':
                axis = 1
        if 'U' in attr:
            need_cast = True
            inputs[0] = _op.cast(inputs[0], dtype=attr['U'].name)

        out = AttrCvt(op_name='batch_norm',
                      transforms={'scale_after_normalization':'scale',
                                  'variance_epsilon':'epsilon'},
                      extras={'axis': axis},
                      ignores=['data_format', 'U'],
                      disables=['momentum'])(inputs, attr)

        if need_cast:
            out = _op.cast(out, dtype=attr['T'].name)
        return out
    return _impl

def _gather():
    "GatherV2, Gather"
    def _impl(inputs, attr, params):

        axis = 0
        if len(inputs) > 2:
            axis = params[inputs.pop(2).name_hint].asnumpy()[0]
        new_input = []
        new_input.append(inputs.pop(0))
        new_input.append(inputs.pop(0))
        return AttrCvt(op_name="take",
                       extras={'axis': tvm.const(axis, 'int32')},
                       ignores=['Tindices', 'Tparams', 'validate_indices', \
                                'Taxis', '_class'])(new_input, attr)
    return _impl

def _identity():
    def _impl(inputs, attr, params):
        return inputs[0]
    return _impl

def _logical(name):
    def _impl(inputs, attr, params):
        return AttrCvt(op_name=name)(inputs, attr)
    return _impl

def _lrn():
    def _impl(inputs, attr, params):
        attr_new = {}
        depth_radius = attr.get('depth_radius', 5)
        size = (depth_radius * 2) + 1
        attr_new['axis'] = 3 # Fix axis, NHWC format
        attr_new['size'] = size
        attr_new['bias'] = attr.get('bias', 1)
        attr_new['alpha'] = attr.get('alpha', 1) * size
        attr_new['beta'] = attr.get('beta', 0.5)
        return AttrCvt(op_name='lrn')(inputs, attr_new)
    return _impl

def _matmul():
    def _impl(inputs, attr, params):
        channels = _infer_channels(inputs[1], params, not attr['transpose_b'])
        if attr['transpose_a']:
            inputs[0] = _op.transpose(inputs[0], axes=(1, 0))
        if not attr['transpose_b']:
            inputs[1] = _op.transpose(inputs[1], axes=(1, 0))
        return AttrCvt(op_name="dense",
                       extras={'units': channels},
                       ignores=['transpose_a', 'transpose_b', 'T'])(inputs, attr)

    return _impl

def _mean():
    def _impl(inputs, attr, params):
        axis = params.pop(inputs[1].name_hint)
        return AttrCvt(op_name="mean", ignores=['Tdim', 'Tidx'],
                       transforms={'keep_dims': 'keepdims'},
                       extras={'axis': tuple(axis.asnumpy())})([inputs[0]], attr)
    return _impl

def _pack():
    def _impl(inputs, attr, params):
        axis = int(attr["axis"])
        inputs_reshaped = [_op.expand_dims(i, axis=axis, num_newaxis=1) for i in inputs]
        return _op.concatenate(inputs_reshaped, axis)
    return _impl

def _pad(name):
    def _impl(inputs, attr, params):
        padlist_key = inputs[1].name_hint
        if padlist_key in params:
            padlist = params.pop(padlist_key).asnumpy()
        else:
            raise tvm.error.OpAttributeRequired(
                'Attribute {} not found in operator Pad.'.format(padlist_key))
        paddings = tuple([tuple(l) for l in padlist])
        attr['pad_width'] = paddings
        attr['pad_value'] = 0
        new_inputs = [inputs[0]]
        if name == 'PadV2':
            constant_values = params.pop(inputs[2].name_hint).asnumpy()
            attr['pad_value'] = constant_values[0]
        return AttrCvt(
            op_name='pad',
            ignores=['Tpaddings'],)(new_inputs, attr)
    return _impl

def _prod():
    def _impl(inputs, attr, params):
        axis = params.pop(inputs[1].name_hint).asnumpy()[0]
        keepdims = attr['keep_dims']
        return _op.prod(inputs[0], int(axis), keepdims=keepdims)
    return _impl

def _range():
    def _impl(inputs, attr, params):
        start = params.pop(inputs[0].name_hint).asnumpy()[0]
        limit = params.pop(inputs[1].name_hint).asnumpy()[0]
        delta = params.pop(inputs[2].name_hint).asnumpy()[0]

        name = attr["_node_name"]
        params[name] = tvm.nd.array([start, limit, delta])
        return [_expr.var(name,
                          shape=params[name].shape,
                          dtype='int32')]
    return _impl

def _rank():
    def _impl(inputs, attr, params):
        input_shape = attr['_input_shapes'][inputs[0]]

        name = attr["_node_name"]
        params[name] = tvm.nd.array([len(input_shape)])
        return [_expr.var(name,
                          shape=params[name].shape,
                          dtype='int32')]

    return _impl

def _relu6():
    def _impl(inputs, attr, params):
        return _op.clip(inputs[0], a_min=0, a_max=6)
    return _impl

def _shape():
    def _impl(inputs, attr, params):
        return np.array(attr['_input_shapes'][inputs[0]], dtype='int32')
    return _impl

def _reshape():
    def _impl(inputs, attr, params):
        try:
            pop_node = inputs[1]
            shape_arg = params.pop(pop_node.name_hint)
            inputs.pop(1)

            return AttrCvt(
                op_name="reshape",
                extras={'newshape':tuple(shape_arg.asnumpy())},
                ignores=['Tshape'])(inputs, attr)
        except AttributeError:
            # Shape operator is already pruned, hence
            # try to infer shape by precompute prune if possible.
            params_new = _infer_value(inputs[1], params)
            inputs.pop(1)
            return AttrCvt(
                op_name="reshape",
                extras={'newshape':tuple(params_new.asnumpy().astype('int64').flatten())},
                ignores=['Tshape'])(inputs, attr)
    return _impl

def _resize_bilinear():
    def _impl(inputs, attr, params):
        size = attr['_output_shapes'][0][1:3]
        # Important that the size is defined. If an axis is not, we need to infer what
        # the shape should be.
        if -1 in size:
            size = _infer_value(inputs[1], params).asnumpy().reshape([-1]).tolist()
        attr['size'] = size
        inputs.pop(1)
        # NHWC
        attr['layout'] = 'NHWC'

        return AttrCvt(op_name="resize",
                       ignores=['Tdim'],
                       extras={'method': "BILINEAR"})(inputs, attr)
    return _impl

def _rsqrt():
    def _impl(inputs, attr, *args):
        inputs.append(tvm.relay.const(-0.5, attr['T'].name))
        return AttrCvt(op_name="power")(inputs, attr)
    return _impl

def _reverse_v2():
    def _impl(inputs, attr, params):
        axis = params.pop(inputs[1].name_hint).asnumpy()[0]
        return AttrCvt(
            op_name="reverse",
            ignores=['Tidx'],
            extras={'axis': int(axis)})([inputs[0]], attr)
    return _impl

def _where():
    def _impl(inputs, attr, params):
        return AttrCvt(op_name="where")(inputs, attr)
    return _impl

def _selu():
    def _impl(inputs, attr, params):
        alpha = tvm.relay.const(-1.6732632423543772848170429916717, attr['T'].name)
        gamma = tvm.relay.const(1.0507009873554804934193349852946, attr['T'].name)
        return gamma * (alpha * _op.nn.relu(tvm.relay.const(1, attr['T'].name) \
                                            - _op.exp(inputs[0])) + _op.nn.relu(inputs[0]))
    return _impl

def _slice():
    def _impl(inputs, attr, params):
        begin = params.pop(_get_name_hint(inputs[1])).asnumpy().tolist()
        size = params.pop(_get_name_hint(inputs[2])).asnumpy().tolist()
        data_shape = attr['_input_shapes'][inputs[0]]
        data_dim = len(data_shape)
        end = size
        for i in range(data_dim):
            if size[i] == -1:
                end[i] = data_shape[i] - begin[i]
            else:
                end[i] += begin[i]
        return _op.strided_slice(inputs[0], begin=begin, end=size)
    return _impl

def _softmax():
    def _impl(inputs, attr, params):
        return AttrCvt(op_name='softmax',
                       transforms={'axis': ('axis', 1)})([inputs[0]], attr)
    return _impl

def _softplus():
    # op description: https://www.tensorflow.org/api_docs/python/tf/math/softplus
    def _impl(inputs, attr, params):
        exp_out = AttrCvt('exp')(inputs, attr)
        inputs.append(tvm.relay.const(1, attr['T'].name))
        rh = tvm.relay.const(1, attr['T'].name)
        add_out = _get_relay_op('add')(exp_out, rh)
        return _get_relay_op('log')(add_out)
    return _impl

def _space_to_batch_nd():
    def _impl(inputs, attr, params):
        input_node = inputs[0]
        input_shape = attr['_input_shapes'][input_node]
        block_shape = params.pop(inputs[1].name_hint).asnumpy().tolist()
        paddings = params.pop(inputs[2].name_hint).asnumpy().tolist()
        N = len(input_shape)
        M = len(block_shape)
        batch = input_shape[0]
        remaining_shape_length = N - M - 1
        paddings = [(0, 0)] + paddings + [(0, 0)] * remaining_shape_length
        # From https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/space-to-batch-n-d:
        # Zero-pad the start and end of dimensions [1, ..., M] of the input according to paddings
        # to produce padded of shape padded_shape.
        padded = tvm.relay.nn.pad(input_node, pad_width=paddings)
        # Reshape padded to reshaped_padded of shape:
        # [batch] + [padded_shape[1] / block_shape[0], block_shape[0], ...,
        # padded_shape[M] / block_shape[M-1], block_shape[M-1]] + remaining_shape
        shape1 = [batch] + [item for i in range(M) for item in [-4, -1, block_shape[i]]] + [-2]
        reshaped_padded = tvm.relay.reshape(padded, newshape=shape1)
        # Permute dimensions of reshaped_padded to produce permuted_reshaped_padded of shape:
        # block_shape + [batch] + [padded_shape[1] / block_shape[0], ...,
        # padded_shape[M] / block_shape[M-1]] + remaining_shape
        axes = [2 * i + 2 for i in range(M)] + [0] + [2 * i + 1 for i in range(M)] + \
               list(range(1 + 2 * M, 1 + 2 * M + remaining_shape_length))
        permuted_reshaped_padded = tvm.relay.transpose(reshaped_padded, axes=axes)
        permuted_reshaped_padded_shape = _infer_out_shapes(permuted_reshaped_padded, params)[0]
        # Reshape permuted_reshaped_padded to flatten block_shape into the batch dimension,
        # producing an output tensor of shape:
        # [batch * prod(block_shape)] + [padded_shape[1] / block_shape[0], ...,
        # padded_shape[M] / block_shape[M-1]] + remaining_shape
        shape2 = [batch * np.prod(block_shape)] + list(permuted_reshaped_padded_shape)[M + 1:]
        reshaped_permuted_reshaped_padded = tvm.relay.reshape(permuted_reshaped_padded,
                                                              newshape=shape2)
        return reshaped_permuted_reshaped_padded

    return _impl

def _split(has_size_vector):
    # TF documentation https://www.tensorflow.org/api_docs/python/tf/split
    def _impl(inputs, attr, params):
        try:
            # order and number of inputs are different:
            # if has_size_vector:
            #     https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/split-v
            # else:
            #     https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/split

            # in addition, `axis` and `num_or_size_splits` can be tensors in TensorFlow,
            # we can only support constants
            if has_size_vector:
                input_node_index = 0
                input_axis_index = 2
                size_splits_input_name = _get_name_hint(inputs[1])
                size_splits = params[size_splits_input_name].asnumpy()
                section_beginnings = np.cumsum(size_splits)[:-1]
                indices_or_sections = tuple(section_beginnings)
            else:
                input_node_index = 1
                input_axis_index = 0
                indices_or_sections = attr['num_split']
            input_node = inputs[input_node_index]
            axis_input_name = _get_name_hint(inputs[input_axis_index])
            axis_input_value = params[axis_input_name].asnumpy()[0]
        except (IndexError, KeyError):
            raise TypeError( \
                "Unsupported argument for split: `axis` and `num_or_size_splits` " \
                "should be constants")
        return _op.split(input_node,
                         indices_or_sections=indices_or_sections,
                         axis=int(axis_input_value))
    return _impl

def _square():
    def _impl(inputs, attr, params):
        return _op.multiply(inputs[0], inputs[0])
    return _impl

def _squeeze():
    def _impl(inputs, attr, params):
        if len(attr['squeeze_dims']) == 0:
            attr['squeeze_dims'] = None
        return AttrCvt(
            op_name="squeeze",
            transforms={'squeeze_dims':'axis'},
            ignores=['T'])(inputs, attr)
    return _impl

def _stridedSlice():
    def _impl(inputs, attr, params):
        """Strided Slice.
        Operator description: https://www.tensorflow.org/api_docs/python/tf/strided_slice
        Tensorflow mask validation: https://github.com/tensorflow/tensorflow/blob/master/
        tensorflow/core/util/strided_slice_op.cc#L147-L368
        """
        begin = params.pop(inputs[1].name_hint).asnumpy().tolist()
        end = params.pop(inputs[2].name_hint).asnumpy().tolist()
        stride = params.pop(inputs[3].name_hint).asnumpy().tolist()
        begin_mask = int(attr.get('begin_mask', 0))
        end_mask = int(attr.get('end_mask', 0))
        ellipsis_mask = int(attr.get('ellipsis_mask', 0))
        new_axis_mask = int(attr.get('new_axis_mask', 0))
        shrink_axis_mask = int(attr.get('shrink_axis_mask', 0))
        data_shape = attr['_input_shapes'][inputs[0]]
        data_dim = len(data_shape)
        stride_dim = len(stride)

        def _transform_mask(stride_dim, ellipsis_mask):
            """Handle mask inputs to create new begin, end, stride and output shape"""
            m_begin = [0] * data_dim
            m_end = [0] * data_dim
            m_stride = [0] * data_dim
            fshape_indices = []
            #Count new axis after ellipsis_mask, consider while applying ellipsis_mask.
            ellipsis_seen = False
            new_axes_after_ellipsis = 0
            for i in range(stride_dim):
                mask = 1 << i
                if ellipsis_seen and (mask & new_axis_mask) != 0:
                    new_axes_after_ellipsis += 1
                if (mask & ellipsis_mask) != 0:
                    ellipsis_seen = True
            if not ellipsis_seen:
                #Used later for extending the stride attributes in the below loop.
                ellipsis_mask |= (1 << stride_dim)
                stride_dim += 1
            final_index = 0
            for index in range(stride_dim):
                mask = 1 << index
                if mask & ellipsis_mask:
                    #Identify the end index for applying ellipsis_mask
                    to_index = min(((data_dim - (stride_dim-index)) + 1 \
                                     + new_axes_after_ellipsis), data_dim)
                    for i in range(final_index, to_index):
                        m_begin[final_index] = 0
                        m_end[final_index] = data_shape[final_index]
                        m_stride[final_index] = 1
                        fshape_indices.append(final_index)
                        final_index += 1
                elif mask &new_axis_mask:
                    fshape_indices.append(-1)
                elif not mask & new_axis_mask:
                    if final_index == len(m_begin):
                        break
                    if mask & begin_mask:
                        m_begin[final_index] = data_shape[final_index] \
                                                     if stride[index] < 0 else 0
                    elif begin[index]:
                        m_begin[final_index] = begin[index]
                    if mask & end_mask:
                        m_end[final_index] = 0 if stride[index] < 0 \
                                                 else data_shape[final_index]
                    elif end[index]:
                        m_end[final_index] = end[index]
                    m_stride[final_index] = stride[index]
                    if mask & shrink_axis_mask:
                        #Tensorflow make axis with shrink_axis_mask as dimension 1
                        m_begin[final_index] = data_shape[final_index] + begin[index] \
                                                 if begin[index] < 0 else begin[index]
                        m_end[final_index] = begin[index] + 1
                        m_stride[final_index] = 1
                        fshape_indices.append(-2)
                    else:
                        fshape_indices.append(final_index)

                    final_index += 1
            return m_begin, m_end, m_stride, fshape_indices

        fshape_indices = None
        if begin_mask or end_mask or ellipsis_mask or new_axis_mask or shrink_axis_mask:
            begin, end, stride, fshape_indices = _transform_mask(stride_dim, ellipsis_mask)
        out = _op.strided_slice(inputs[0], begin=begin, end=end, strides=stride)
        out_shape = _infer_out_shapes(out, params)[0]
        if not fshape_indices:
            fshape_indices = range(len(out_shape))

        #Create final output shape.
        final_output = []
        for gather_index in fshape_indices:
            if gather_index == -1:
                final_output.append(1)
            elif gather_index == -2:
                pass
            else:
                final_output.append(out_shape[gather_index])

        if not final_output:
            return out
        return _op.reshape(out, newshape=tuple(final_output))
    return _impl

def _sum():
    def _impl(inputs, attr, params):
        axis = params.pop(inputs[1].name_hint).asnumpy()
        # convert to tuple for preventing invalid parameter format error
        axis = tuple(axis)
        return AttrCvt(
            op_name='sum',
            extras={'axis': axis},
            transforms={'keep_dims':'keepdims'},
            ignores=['name', 'Tidx'])([inputs[0]], attr)
    return _impl

def _tile():
    def _impl(inputs, attr, params):
        reps = params[inputs.pop().name_hint].asnumpy()
        new_input = []
        new_input.append(inputs.pop(0))

        return AttrCvt(
            op_name='tile',
            extras={'reps': tuple(reps)},
            ignores=['Tmultiples'])(new_input, attr)
    return _impl

def _transpose():
    def _impl(inputs, attr, params):
        # If perm is not specified, axes is left empty,
        # otherwise its value is get from params
        param_name = _get_name_hint(inputs[1])
        if param_name in params:
            axes = tuple(params.get(param_name).asnumpy())
        else:
            axes = None
        return _op.transpose(inputs[0], axes=axes)
    return _impl

def _unpack():
    def _impl(inputs, attr, params):
        input_node = inputs[0]
        axis = attr['axis']
        input_shape = attr['_input_shapes'][input_node]
        axis_length = input_shape[axis]
        if axis_length < 0:
            raise TypeError("Unstack with unknown axis length")
        splitted = _op.split(input_node,
                             indices_or_sections=axis_length,
                             axis=axis)
        #name=attr.get('_node_name', 'unstack'))
        if axis == 0:
            axis = None
        else:
            axis = [axis]
        return _expr.TupleWrapper(
            _expr.Tuple([_op.squeeze(split_item, axis=axis) \
            for split_item in splitted]), len(splitted))
    return _impl

def _conv2d(inputs, attrs, params):
    data_format = attrs['data_format'].decode("utf-8")
    input_eids = attrs['_input_eids']
    infer_shapes = attrs['_infer_shapes']

    assert data_format in ["NCHW", "NHWC"]

    data_shp = infer_shapes[inputs[0].attr('name')][input_eids[0]]
    weight_shp = infer_shapes[inputs[1].attr('name')][input_eids[1]]

    if data_format == "NHWC":
        inputs[0] = mx.sym.transpose(inputs[0], axes=(0, 3, 1, 2))
        data_shp = [data_shp[ii] for ii in (0, 3, 1, 2)]

    # Transpose weight format into "OIHW"
    # TODO(wlt): note if op is depthwise, 
    #   original weight format is "HWOI" instead of "HWIO".
    inputs[1] = mx.sym.transpose(inputs[1], axes=(3, 2, 0, 1))
    weight_shp = [weight_shp[ii] for ii in (3, 2, 0, 1)]
    # params[weight_name] = nd.transpose(params[weight_name], axes=(3, 2, 0, 1))

    print ([x.attr('name') for x in inputs])

    H_idx, W_idx = data_format.find("H"), data_format.find("W")
    dilations = attrs.get('dilations', (1, 1))
    if isinstance(dilations, int):
        dilations = (dilations, dilations)
    elif len(dilations) == 4:
        dilations = (dilations[H_idx], dilations[W_idx])

    strides = attrs.get('strides')
    if isinstance(strides, int):
        strides = (strides, strides[0])
    elif len(strides) == 4:
        strides = (strides[H_idx], strides[W_idx])

    def _get_pad_pair(input1d, kernel1d, stride1d):
        if input1d % stride1d == 0:
            pad = max(kernel1d - stride1d, 0)
        else:
            pad = max(kernel1d - (input1d % stride1d), 0)

        pad_before = pad // 2
        pad_after = pad - pad_before

        return [pad_before, pad_after]

    padding = attrs['padding'].decode("utf-8")
    if padding == 'VALID':
        padding = (0, 0)
    elif padding == 'SAME':
        stride_h, stride_w = strides
        kernel_h, kernel_w = weight_shp[2:]
        in_h, in_w = data_shp[2], data_shp[3]
        dilation_h, dilation_w = dilations
        dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
        dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
        pad_v = _get_pad_pair(in_h, dilated_kernel_h, stride_h)
        pad_h = _get_pad_pair(in_w, dilated_kernel_w, stride_w)
        print ("Padding", pad_v, pad_h)
        padding = (0, 0)

    assert data_shp[1] % weight_shp[1] == 0
    groups = data_shp[1] // weight_shp[1]
    conv_attr = {
        'no_bias': (len(inputs) == 2),
        'dilate': dilations,
        'kernel': weight_shp[2:],
        'stride': strides,
        'pad': padding,
        'layout': 'NCHW',
        'num_filter': weight_shp[0],
        'num_group': groups,
    }
    print (data_shp, weight_shp)
    print (conv_attr)
    sym = mx.sym.Convolution(*inputs, **conv_attr)

    if data_format == "NHWC":
        sym = mx.sym.transpose(sym, axes=(0, 2, 3, 1))
    return sym

_convert_map = {
    'Add'                               : _elemwise('add'),
    'All'                               : _reduce_all(),
    'ArgMax'                            : _argx(argmax, 'argmax'),
    'ArgMin'                            : _argx(argmin, 'argmin'),
    'AvgPool'                           : _pooling('avg_pool'),
    'BatchNormWithGlobalNormalization'  : _batch_norm(),
    'BatchToSpaceND'                    : _batch_to_space_nd(),
    'BiasAdd'                           : _bias_add(),
    'BroadcastTo'                       : _broadcast_to(),
    'Cast'                              : _cast(),
    'Ceil'                              : AttrCvt('ceil'),
    'CheckNumerics'                     : _check_numerics(),
    'Concat'                            : _concat(),
    'ConcatV2'                          : _concatV2(),
    # 'Conv2D'                            : _conv('conv'),
    'Conv2D'                            : _conv2d,
    'DecodeJpeg'                        : _decode_image(),
    'DepthwiseConv2dNative'             : _conv('depthwise'),
    'DepthToSpace'                      : _depth_to_space(),
    'Equal'                             : _broadcast('equal'),
    'Elu'                               : _elu(),
    'Exp'                               : AttrCvt('exp'),
    'ExpandDims'                        : _expand_dims(),
    'Fill'                              : _fill(),
    'Floor'                             : AttrCvt('floor'),
    'FusedBatchNorm'                    : _fused_batch_norm(),
    'FusedBatchNormV2'                  : _fused_batch_norm(),
    'Gather'                            : _gather(),
    'GatherV2'                          : _gather(),
    'Greater'                           : _broadcast('greater'),
    'GreaterEqual'                      : _broadcast('greater_equal'),
    'Identity'                          : _identity(),
    'LeakyRelu'                         : AttrCvt('leaky_relu'),
    'Less'                              : _broadcast('less'),
    'LessEqual'                         : _broadcast('less_equal'),
    'Log'                               : AttrCvt('log'),
    'LogicalAnd'                        : _logical('logical_and'),
    'LogicalOr'                         : _logical('logical_or'),
    'LogicalNot'                        : _logical('logical_not'),
    'LRN'                               : _lrn(),
    'MatMul'                            : _matmul(),
    'MaxPool'                           : _pooling('max_pool'),
    'Maximum'                           : _elemwise('maximum'),
    'Mean'                              : _mean(),
    'Minimum'                           : _elemwise('minimum'),
    'Mul'                               : _elemwise('multiply'),
    'NotEqual'                          : _broadcast('not_equal'),
    'Pack'                              : _pack(),
    'Pad'                               : _pad('Pad'),
    'PadV2'                             : _pad('PadV2'),
    'Pow'                               : _elemwise('power'),
    'Prod'                              : _prod(),
    'Range'                             : _range(),
    'Rank'                              : _rank(),
    'RealDiv'                           : _elemwise('divide'),
    'Relu'                              : AttrCvt('relu'),
    'Relu6'                             : _relu6(),
    'Reshape'                           : _reshape(),
    'ResizeBilinear'                    : _resize_bilinear(),
    'ResizeBicubic'                     : _resize_bilinear(),
    'ReverseV2'                         : _reverse_v2(),
    'Round'                             : AttrCvt('round'),
    'Rsqrt'                             : _rsqrt(),
    'Select'                            : _where(),
    'Selu'                              : _selu(),
    'Shape'                             : _shape(),
    'Sigmoid'                           : AttrCvt('sigmoid'),
    'Sign'                              : AttrCvt('sign'),
    'Slice'                             : _slice(),
    'Softmax'                           : _softmax(),
    'Softplus'                          : _softplus(),
    'SpaceToBatchND'                    : _space_to_batch_nd(),
    'Split'                             : _split(False),
    'SplitV'                            : _split(True),
    'Sqrt'                              : AttrCvt('sqrt'),
    'Square'                            : _square(),
    'Squeeze'                           : _squeeze(),
    'StridedSlice'                      : _stridedSlice(),
    'Sub'                               : _elemwise('subtract'),
    'Sum'                               : _sum(),
    'Tanh'                              : AttrCvt('tanh'),
    'Tile'                              : _tile(),
    'Transpose'                         : _transpose(),
    'Unpack'                            : _unpack(),

}

fieldOrgTypes = (int, bool, float, bytes)

def convert_field(attrFields, logger=logging):
    fields = attrFields.ListFields()
    if len(fields) > 1:
        logger.error("Multiple AttrValue fields found.")
        exit()
    elif not len(fields):
        logger.error("Null AttrValue field found.")
        exit()
    _, fieldValue = fields[0]
    if isinstance(fieldValue, fieldOrgTypes):
        return fieldValue
    elif isinstance(fieldValue, tspb2.TensorShapeProto):
        return tuple([dim.size for dim in \
                fieldValue.ListFields()[0][1]])
    elif isinstance(fieldValue, tpb2.TensorProto):
        # the length of ffields must be 3
        # which is respectively: num, shape, tensor
        ffields = fieldValue.ListFields()
        ff = ffields[1][1].ListFields()
        # the length of ff must be  
        if len(ff) == 1:
            shapes = tuple([dim.size for dim in ffields[1][1].ListFields()[0][1]])
            data = ffields[2][1]
            if isinstance(data, bytes):
                return (ffields[0][1], shapes, data)
            elif str(type(data)) == "<class 'google.protobuf.pyext._message.RepeatedScalarContainer'>":
                return (ffields[0][1], shapes, data[0])
            else:
                logger.error("data error 2")
                exit()
        elif not len(ff):
            return (ffields[0][1], None, ffields[2][1][0])
        else:
            logger.error("data error 1")
            exit()
    elif isinstance(fieldValue, apb2.AttrValue.ListValue):
        return tuple(fieldValue.ListFields()[0][1])
    else:
        logger.error("Unsupported field type '%s'", type(fieldValue))
        exit()

def _parse_attr(attrs):
    fields = ["s", "i", "f", "b", "type", "shape", "tensor", "func"]
    new_attrs = {}
    for k, v in attrs.items():
        ret = []
        if v.HasField("list"):
            for f in fields:
                if getattr(v.list, f):
                    if f == "type":
                        ret += [dtypes.as_dtype(x) for x in list(getattr(v.list, f))]
                    else:
                        ret += list(getattr(v.list, f))
        else:
            for f in fields:
                if v.HasField(f):
                    if f == "type":
                        ret = dtypes.as_dtype(getattr(v, f))
                    else:
                        ret = getattr(v, f)
        new_attrs[k] = ret
    return new_attrs


def convert_operator(op_name, inputs, attrs, params, logger=logging):
    if op_name not in _convert_map:
        raise NotImplementedError("Operator {} not implemented.".format(op_name))

    # attr = { k: convert_field(v) for k, v in attrs.items() }

    sym = _convert_map[op_name](inputs, attrs, params)
    return sym

import tensor_util
def convert_tfnode(tfnode, graph, params, infer_shapes, logger=logging):
    name, op_name = tfnode.name, tfnode.op
    attr, org_inputs = tfnode.attr, tfnode.input

    if op_name not in currSupportedOps:
        logger.critical("Not supported op '%s'", tfnode.op)
        exit()

    print ("%-16s" % op_name,
           "%-40s" % name, [k for k, _ in attr.items()])
    if op_name == 'Const':
        for k, v in attr.items():
            if k == 'value':
                np_array = tensor_util.tensor_to_numpy(v.tensor)
                assert len(np_array.shape) > 0, "value:%s and dtype:%s" \
                    % (v.tensor, np_array.dtype)
                params[name] = nd.array(np_array)
                graph[name] = [mx.sym.var(name,
                                          shape=params[name].shape,
                                          dtype=params[name].dtype)]
                infer_shapes[name] = [tuple(np_array.shape)]
                print (infer_shapes[name])
            elif k not in ('dtype', '_output_shapes', '_class'):
                raise NotImplementedError \
                    ("Other attributes for a Const(param) Node {} ? .".format(k))
    elif op_name in ['Placeholder', 'PlaceholderWithDefault']:
        input_shape = \
            tensor_util.tensor_shape_proto_to_list(attr['shape'].shape)
        dtype = tensor_util.tensor_type_to_numpy(attr['dtype'].type)
        graph[name] = [mx.sym.var(name, shape=input_shape, dtype=dtype)]
        infer_shapes[name] = [tuple(input_shape)]
    else:
        inputs, input_eids = [], []
        for in_name in org_inputs:
            input_entry = in_name.split(":")
            node_name = input_entry[0]
            assert node_name in graph, "toposort error: input '%s', node '%s'." % (node_name, name)
            child = graph[node_name]
            if len(child) > 1 and len(input_entry) > 1:
                eid = int(input_entry[1])
            else:
                eid = 0
            inputs.append(child[eid])
            input_eids.append(eid)

        parsed_attrs = _parse_attr(attr)
        parsed_attrs["_input_eids"] = input_eids
        parsed_attrs["_infer_shapes"] = infer_shapes
        # TODO(ryt): convert operators
        sym = convert_operator(op_name, inputs, parsed_attrs, params)
        graph[name] = sym
        _, infer_shapes[name], _ = sym.infer_shape()
        print (infer_shapes[name])
    return

currSupportedOps = {
                       'Const',
                       'Pad',
                       'Identity',
                       'FusedBatchNorm',
                       'MatMul',
                       'Relu', 'Relu6',
                       'Softmax', 'Mean',
                       'MaxPool', 'AvgPool',
                       'BiasAdd', 'Add', 'Placeholder',
                       'Conv2D', 'DepthwiseConv2dNative',
                       'Shape', 'Reshape',
                       'Fill',
                       'ConcatV2',
                       'StridedSlice',
                       'Pack'
                   }

currSupportedAttrs = {
                        'Conv2D': { 'strides', 'data_format', 'padding',
                                   'dilations', 'use_cudnn_on_gpu', 'T' },
                        'Const': { 'value', 'dtype' }
                     }

currRealizedOps = { }


def topo_sort(tfgraph, logger=logging):
    node_map = {}
    deps, ninps, res = {}, [], {}
    for node in tfgraph.node:
        node_map[node.name] = node
        if node.op not in currSupportedOps:
            logger.error("the op '%s' of node '%s' is not supported",
                    node.op, node.name)
            exit()
        # TODO(ryt): input name may concat output index such as:
        #   'Model/cell_0/RnnCell' and 'Model/cell_0/RnnCell:0'
        for inp in node.input:
            inp = inp.split(":")[0]
            if inp not in deps:
                deps[inp] = set()
            deps[inp].add(node.name)
        if not len(node.input):
            ninps.append(node.name)
        else:
            res[node.name] = len(node.input)

    # topo sort
    topos = []
    while len(ninps):
        cname = ninps.pop()
        topos.append(node_map[cname])
        if cname not in deps:
            continue
        for name in deps[cname]:
            if res[name] > 1:
                res[name] -= 1
            else:
                res.pop(name)
                ninps.append(name)
    if res:
        logger.critical("deps cannot reduce -> %s", res)
        exit()
    return topos

def convert_model(pbfile):
    # load the original model
    logger = logging.getLogger("Loading Original Model")
    tfparser = TFParser(pbfile)
    tfgraph = tfparser.parse()
    logger.info("Model successfully loaded from path [%s].", pbfile)

    ops = {n.op for n in topo_sort(tfgraph)}
    print ("Ops", ops)
    graph, params, infer_shapes = {}, {}, {}
    for tfnode in topo_sort(tfgraph):
        # if tfnode.op == "Conv2D":
            # inputs = tfnode.input
            # print(inputs[0])
            # test = [atk for k, v in node_out_map[]]
        # op_name, attrs = tfnode.op, tfnode.attr
        # for k, _ in attrs.items():
        #     ts.add(k)
        convert_tfnode(tfnode, graph, params, infer_shapes)

    logger.info("Operators successfully converted.")



modelfile = [
            # "/tmp/tf/resnet50_v1/model.pb",
            "/data/tfmodels/inception_v3/model.pb",
            # "/data/tfmodels/keras/inception_v3/model.pb",
            # "/data/tfmodels/mobilenet/model.pb"
            ]

# if True:
#     utils.log_init()
#     for pb in modelfile:
#         convert_model(pb)

def dump_single_sym(sym,
        path = os.path.expanduser("~/.dump/test_sym.json")):
    with open(path, "w") as f:
        f.write(sym.tojson())

# conv_attr = {
#     'layout': 'NCHW',
#     'pad': (1, 1),
#     'num_filter': 16,
#     'dilate': (1, 1),
#     'num_group': 1,
#     'stride': (1, 1),
#     'no_bias': False,
#     'kernel': (3, 3)
# }
# sym = mx.sym.Convolution(X, W, **conv_attr, name=conv_name)
# sym = mx.sym.Convolution(**conv_attr, name='test_ryt')
# dump_single_sym(sym)
# 
# print(ts)

if __name__ == '__main__':
    utils.log_init()
    model_path = modelfile[0]
    convert_model(model_path)
    print(ts)
    print(tl)
