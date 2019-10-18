from tensorflow_parser import TFParser
from tensorflow.core.framework import tensor_pb2 as tpb2
from tensorflow.core.framework import tensor_shape_pb2 as tspb2
from tensorflow.core.framework import attr_value_pb2 as apb2
from tensorflow.python.framework import dtypes
import mxnet as mx
from mxnet import nd
import numpy as np

import os
import logging
import utils
import cvm_op as cvm
import numpy as np
# from python.tvm.relay.op import op as _op

# import heapq
import sym_utils as sutils
import sym_pass as spass


def _bias_add(inputs, attrs, params):
    input_eids = attrs["_input_eids"]
    infer_shapes = attrs["_infer_shapes"]
    data_shp = infer_shapes[inputs[0].attr('name')][input_eids[0]]
    bias_shp = infer_shapes[inputs[1].attr('name')][input_eids[1]]
    data_format = attrs["data_format"].decode("utf-8")
    if data_format == "NCHW":
        inputs[1] = mx.sym.reshape(inputs[1], (1, bias_shp[0], 1, 1))
    return mx.sym.broadcast_add(*inputs)

def _identity(inputs, attrs, params):
    return inputs[0]

def _matmul(inputs, attrs, params):
    input_eids = attrs["_input_eids"]
    infer_shapes = attrs["_infer_shapes"]
    bshp = infer_shapes[inputs[1].attr('name')][input_eids[1]]
    if attrs["transpose_a"]:
        inputs[0] = mx.sym.transpose(inputs[0], axes=(1, 0))
    if not attrs["transpose_b"]:
        inputs[1] = mx.sym.transpose(inputs[1], axes=(1, 0))
        bshp = (bshp[1], bshp[0])
    dense_attr = {
        'no_bias': len(inputs) == 2,
        'num_hidden': bshp[0],
    }
    return mx.sym.FullyConnected(*inputs, **dense_attr)

def _mean(inputs, attrs, params):
    input_eids = attrs["_input_eids"]
    infer_shapes = attrs["_infer_shapes"]
    data_shp = infer_shapes[inputs[0].attr('name')][input_eids[0]]
    axis = params.pop(inputs[1].attr('name')).asnumpy().astype('int')
    sym = mx.sym.sum(inputs[0],
            axis=tuple(axis),
            keepdims=attrs["keep_dims"])
    scalar = 1. / np.product([data_shp[ii] for ii in axis])
    scale = mx.sym.var(sym.attr('name') + "_scale", shape=(1,))
    params[scale.attr('name')] = nd.array([scalar])
    sym = mx.sym.broadcast_mul(sym, scale)
    return sym

def _pad(inputs, attrs, params):
    padding = params[inputs[1].attr('name')]
    sym = mx.sym.Custom(inputs[0],
            padding=padding.asnumpy().astype(attrs['Tpaddings']).tolist(),
            op_type="cvm_pad")
    return sym

def _relu6(inputs, attrs, params):
    return mx.sym.clip(inputs[0], a_min=0, a_max=6)

def _shape(inputs, attr, params):
    name = sutils.gen_name('shape')
    infer_shapes, input_eids = attr['_infer_shapes'], attr['_input_eids']
    params[name] = nd.array(
            infer_shapes[inputs[0].attr('name')][input_eids[0]], dtype='int32')
    return mx.sym.var(name=name, shape=params[name].shape)

def _softmax(inputs, attrs, params):
    axis = attrs.get("axis", -1)
    return mx.sym.softmax(*inputs, axis=axis)

def _get_pad_pair(input1d, kernel1d, stride1d):
    if input1d % stride1d == 0:
        pad = max(kernel1d - stride1d, 0)
    else:
        pad = max(kernel1d - (input1d % stride1d), 0)

    pad_before = pad // 2
    pad_after = pad - pad_before

    return [pad_before, pad_after]

def _conv2d(opname):
    def _impl(inputs, attrs, params):
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
        # Note if op is depthwise, original weight format is "HWOI" 
        #   instead of "HWIO".
        if opname == 'conv':
            inputs[1] = mx.sym.transpose(inputs[1], axes=(3, 2, 0, 1))
            weight_shp = [weight_shp[ii] for ii in (3, 2, 0, 1)]
        else:
            inputs[1] = mx.sym.transpose(inputs[1], axes=(2, 3, 0, 1))
            weight_shp = [weight_shp[ii] for ii in (2, 3, 0, 1)]

        H_idx, W_idx = data_format.find("H"), data_format.find("W")
        dilations = attrs.get('dilations', (1, 1))
        if isinstance(dilations, int):
            dilations = (dilations, dilations)
        elif len(dilations) == 4:
            dilations = (dilations[H_idx], dilations[W_idx])

        strides = attrs.get('strides')
        if isinstance(strides, int):
            strides = (strides, strides)
        elif len(strides) == 4:
            strides = (strides[H_idx], strides[W_idx])

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
            assert pad_v[0] == pad_v[1]
            assert pad_h[0] == pad_h[1]
            padding = (pad_v[0], pad_h[0])

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
        sym = mx.sym.Convolution(*inputs, **conv_attr)

        if data_format == "NHWC":
            sym = mx.sym.transpose(sym, axes=(0, 2, 3, 1))
        return sym
    return _impl

def _elemwise(name):
    def _impl(inputs, attrs, params):
        assert len(inputs) == 2, \
                "{} take 2 inputs, {} given".format(name, len(inputs))
        return sutils.get_mxnet_op(name)(*inputs)
    return _impl

def _pool2d(pool_type):
    def _impl(inputs, attrs, params):
        data_format = attrs['data_format'].decode("utf-8")
        input_eids = attrs['_input_eids']
        infer_shapes = attrs['_infer_shapes']

        assert data_format in ["NCHW", "NHWC"]
        assert attrs['T'] in [dtypes.float32, dtypes.float16, dtypes.float64]

        data_shp = infer_shapes[inputs[0].attr('name')][input_eids[0]]

        if data_format == "NHWC":
            inputs[0] = mx.sym.transpose(inputs[0], axes=(0, 3, 1, 2))
            data_shp = [data_shp[ii] for ii in (0, 3, 1, 2)]

        H_idx, W_idx = data_format.find("H"), data_format.find("W")
        strides = attrs.get('strides')
        if isinstance(strides, int):
            strides = (strides, strides)
        elif len(strides) == 4:
            strides = (strides[H_idx], strides[W_idx])

        kernel_size = attrs['ksize']
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        elif len(kernel_size) == 4:
            kernel_size = (kernel_size[H_idx], kernel_size[W_idx])

        padding = attrs['padding'].decode("utf-8")
        if padding == 'VALID':
            padding = (0, 0)
        elif padding == 'SAME':
            stride_h, stride_w = strides
            kernel_h, kernel_w = kernel_size
            in_h, in_w = data_shp[2], data_shp[3]

            pad_v = _get_pad_pair(in_h, kernel_h, stride_h)
            pad_h = _get_pad_pair(in_w, kernel_w, stride_w)
            assert pad_v[0] == pad_v[1]
            assert pad_h[0] == pad_h[1]
            padding = (pad_v[0], pad_h[0])

        pool_attr = {
            'pool_type': pool_type,
            'stride': strides,
            'pad': padding,
            'kernel': kernel_size,
        }

        if pool_type == "avg":
            # TODO: this actually should be count_include_pad False
            pool_attr['count_include_pad'] = True

        sym = mx.sym.Pooling(*inputs, **pool_attr)
        if data_format == "NHWC":
            sym = mx.sym.transpose(sym, axes=(0, 2, 3, 1))
        return sym
    return _impl

def _batch_normalization(inputs, attrs, params):
    data_format = attrs['data_format'].decode("utf-8")
    input_eids = attrs['_input_eids']
    infer_shapes = attrs['_infer_shapes']

    assert data_format in ["NCHW", "NHWC"]
    assert attrs['T'] in [dtypes.float32, dtypes.float16, dtypes.float64]
    assert attrs["is_training"] == False

    data_shp = infer_shapes[inputs[0].attr('name')][input_eids[0]]

    if data_format == "NHWC":
        inputs[0] = mx.sym.transpose(inputs[0], axes=(0, 3, 1, 2))
        data_shp = [data_shp[ii] for ii in (0, 3, 1, 2)]

    bn_attr = {
        'eps': attrs["epsilon"],
    }
    sym = mx.sym.BatchNorm(*inputs, **bn_attr)

    if data_format == "NHWC":
        sym = mx.sym.transpose(sym, axes=(0, 2, 3, 1))
    return sym

def _relu(inputs, attrs, params):
    return mx.sym.relu(*inputs)

def _concat_v2(inputs, attrs, params):
    axis_sym = inputs.pop(len(inputs) - 1)
    axis = params.pop(axis_sym.attr('name'))
    sym = mx.sym.concat(*inputs, dim=int(axis.reshape((1,)).asscalar()))
    return sym

def _strided_slice(inputs, attrs, params):
    input_eids, infer_shapes = attrs['_input_eids'], attrs['_infer_shapes']
    begin = params[inputs[1].attr('name')].asnumpy().astype('int').tolist()
    end = params[inputs[2].attr('name')].asnumpy().astype('int').tolist()
    stride = params[inputs[3].attr('name')].asnumpy().astype('int').tolist()

    begin_mask = attrs.get('begin_mask', 0)
    end_mask = attrs.get('end_mask', 0)
    ellipsis_mask = attrs.get('ellipsis_mask', 0)
    new_axis_mask = attrs.get('new_axis_mask', 0)
    shrink_axis_mask = attrs.get('shrink_axis_mask', 0)

    data_shape = infer_shapes[inputs[0].attr('name')][input_eids[0]]
    data_dim, stride_dim = len(data_shape), len(stride)

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

    out = mx.sym.slice(inputs[0], begin=begin, end=end, step=stride)
    _, out_shape, _ = out.infer_shape()
    out_shape = out_shape[0]

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

    if tuple(final_output) == out_shape:
        return out
    return mx.sym.reshape(out, shape=final_output)

def _pack(inputs, attrs, params):
    axis = attrs['axis']
    inputs_reshped = [mx.sym.expand_dims(i, axis=axis) for i in inputs]
    op = mx.sym.concat(*inputs_reshped, dim=axis)
    return mx.sym.cast(op, attrs['T'])

def _reshape(inputs, attrs, params):
    X, shape = inputs

    graph = {}
    for op in sutils.topo_sort(shape):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sutils.sym_iter(op.get_children()), op.list_attr()
        if childs is not None:
            childs = [graph[c.attr('name')] for c in childs]

        if sutils.is_var(op, params):
            pass
        elif childs is None:
            params[name] = sutils.get_nd_op(op_name)(**attr)
            op = mx.sym.var(name, shape=params[name].shape)
        else:
            childs = [graph[c.attr('name')] for c in childs]
            assert all([sutils.is_params(c, params) for c in childs])
            in_params = [params[c.attr('name')] for c in childs]
            if op_name == "expand_dims" and in_params[0].shape == ():
                params[name] = nd.array([in_params[0].asnumpy()],
                        dtype=in_params[0].dtype)
            elif op_name == "Reshape" and sutils.get_attr(attr, 'shape') == []:
                assert in_params[0].shape == (1,)
                params[name] = nd.array(in_params[0].asnumpy()[0],
                        dtype=in_params[0].dtype)
            else:
                params[name] = sutils.get_nd_op(op_name)(*in_params, **attr)
            op = mx.sym.var(name, shape=params[name].shape)
        graph[name] = op

    assert sutils.is_params(graph[shape.attr('name')], params)
    shape = params[shape.attr('name')].asnumpy().tolist()
    shape[0] = -1 # since dim zero is batch, set -1 for flexiblity.
    return mx.sym.reshape(X, shape)


_convert_map = {
    'Add'                               : _elemwise('add'),
    'AvgPool'                           : _pool2d("avg"),
    'BiasAdd'                           : _bias_add,
    'ConcatV2'                          : _concat_v2,
    'Conv2D'                            : _conv2d('conv'),
    'DepthwiseConv2dNative'             : _conv2d('depthwise'),
    'FusedBatchNorm'                    : _batch_normalization,
    'Identity'                          : _identity,
    'MatMul'                            : _matmul,
    'MaxPool'                           : _pool2d("max"),
    'Maximum'                           : _elemwise('maximum'),
    'Mean'                              : _mean,
    'Minimum'                           : _elemwise('minimum'),
    'Mul'                               : _elemwise('multiply'),
    'Pack'                              : _pack,
    'Pad'                               : _pad,
    'Pow'                               : _elemwise('power'),
    'RealDiv'                           : _elemwise('divide'),
    'Relu'                              : _relu,
    'Relu6'                             : _relu6,
    'Reshape'                           : _reshape,
    'Shape'                             : _shape,
    'Softmax'                           : _softmax,
    'Sub'                               : _elemwise('subtract'),
    'StridedSlice'                      : _strided_slice,
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
                        ret += [tensor_util.tensor_type_to_numpy(x) \
                               for x in list(getattr(v.list, f))]
                    else:
                        ret += list(getattr(v.list, f))
        else:
            for f in fields:
                if v.HasField(f):
                    if f == "type":
                        ret = tensor_util.tensor_type_to_numpy(getattr(v, f))
                    else:
                        ret = getattr(v, f)
        new_attrs[k] = ret
    return new_attrs


def convert_operator(op_name, inputs, attrs, params, logger=logging):
    if op_name not in _convert_map:
        raise NotImplementedError("Operator {} not implemented.".format(op_name))
    '''
    elif op_name == 'Pad':
        padding = params[inputs[1].attr('name')]
        # print (padding.asnumpy(), padding.shape)
        cusPad = cvm.Pad(padding=padding)
        sym = cusPad(inputs[0])
    '''
    sym = _convert_map[op_name](inputs, attrs, params)
    # attr = { k: convert_field(v) for k, v in attrs.items() }
    return sym

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

import tensor_util
def convert_tfnode(tfnode, graph, params, infer_shapes, logger=logging):
    name, op_name = tfnode.name, tfnode.op
    attr, org_inputs = tfnode.attr, tfnode.input

    if op_name not in currSupportedOps:
        logger.critical("Not supported op '%s'", tfnode.op)
        exit()

    if op_name == 'Const':
        for k, v in attr.items():
            if k == 'value':
                np_array = tensor_util.tensor_to_numpy(v.tensor)
                params[name] = nd.array(np_array, dtype=np_array.dtype)
                graph[name] = [mx.sym.var(name,
                                          shape=params[name].shape,
                                          dtype=params[name].dtype)]
                infer_shapes[name] = [tuple(np_array.shape)]
            elif k not in ('dtype', '_output_shapes', '_class'):
                raise NotImplementedError \
                    ("Other attributes for a Const(param) Node {} ? .".format(k))
    elif op_name in ['Placeholder', 'PlaceholderWithDefault']:
        input_shape = \
            tensor_util.tensor_shape_proto_to_list(attr['shape'].shape)
        dtype = tensor_util.tensor_type_to_numpy(attr['dtype'].type)
        graph[name] = [mx.sym.var("data", shape=input_shape, dtype=dtype)]
        assert "data" not in infer_shapes
        infer_shapes["data"] = [tuple(input_shape)]
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
        sym = convert_operator(op_name, inputs, parsed_attrs, params)
        graph[name] = [sym]
        _, infer_shapes[sym.attr('name')], _ = sym.infer_shape()
    return graph[name]

def topo_sort(tfgraph, logger=logging):
    node_map = {}
    deps, ninps, res = {}, [], {}
    for node in tfgraph.node:
        node_map[node.name] = node
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

def convert_model(pbfile, layout="NHWC", outputs=None):
    # load the original model
    logger = logging.getLogger("Loading Original Model")
    tfparser = TFParser(pbfile)
    tfgraph = tfparser.parse()
    logger.info("Model successfully loaded from path [%s].", pbfile)

    ops = {n.op for n in topo_sort(tfgraph)}
    print ("Ops", ops)

    graph, params, infer_shapes = {}, {}, {}
    for tfnode in topo_sort(tfgraph):
        # print("name: {}, op: {}, inputs: {}".format(tfnode.name, tfnode.op, tfnode.input))
        convert_tfnode(tfnode, graph, params, infer_shapes)

    logger.info("Operators successfully converted.")

    # The last node of tfgraph is assumed as output by default.
    if outputs is None:
        outputs = [tfgraph.node[-1].name]
    nodes = []
    for oname in outputs:
        if ":" in oname:
            oname, onum = oname.split(":")
            eid = int(onum)
            nodes.append(graph[oname][eid])
        else:
            nodes.append(graph[oname][0])
    symbol = mx.sym.Group(nodes) if len(nodes) > 1 else nodes[0]

    if layout == "NHWC":
        symbol, params = spass.convert_input_format(symbol, params)

    return symbol, params

def dump(model, symbol, params):
    prefix = "./data/tf_%s" % (model)
    sym_file, params_file = utils.extend_fname(prefix)
    with open(sym_file, "w") as f:
        f.write(symbol.tojson())
    nd.save(params_file, params)
    logger.info("Model successfully dumped to '%s'", sym_file)

modelfile = [
            # "/tmp/tf/resnet50_v1/model.pb",
            "/data/tfmodels/inception_v3/model.pb",
            # "/data/tfmodels/keras/inception_v3/model.pb",
            "/data/tfmodels/mobilenet/model.pb"
            ]

if __name__ == '__main__':
    utils.log_init()
    model_path = modelfile[0]
    sym, params = convert_model(model_path)
    dump("inceptionv3", sym, params)

    model_path = modelfile[1]
    convert_model(model_path)
    dump("mobilenet", sym, params)
