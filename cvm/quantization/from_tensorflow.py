from tensorflow_parser import TFParser
from tensorflow.core.framework import tensor_pb2 as tpb2
from tensorflow.core.framework import tensor_shape_pb2 as tspb2
from tensorflow.core.framework import attr_value_pb2 as apb2
import mxnet as mx

import os
import logging
import utils

import heapq

ts = set()
fieldOrgTypes = (int, bool, float)

def convert_field(node, attrName, attrFields):
    fields = attrFields.ListFields()
    if len(fields) > 1:
        logger.error("Multiple AttrValue fields found in node '%s' --> " + \
                "op '%s' --> attr '%s' which is not supported.",
                node.name, node.op, attrName)
        exit()
    elif not len(fields):
        logger.error("Null AttrValue field found in node '%s' --> " + \
                "op '%s' --> attr '%s' which is not supported.",
                node.name, node.op, attrName)
        exit()
    _, fieldValue = fields[0]
    if isinstance(fieldValue, fieldOrgTypes):
        return fieldValue
    elif isinstance(fieldValue, bytes):
        return str(fieldValue, encoding='utf-8')
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
            return (ffields[0][1], shapes, ffields[2][1])
        elif not len(ff):
            return (ffields[0][1], None, ffields[2][1][0])
    elif isinstance(fieldValue, apb2.AttrValue.ListValue):
        return tuple(fieldValue.ListFields()[0][1])
    else:
        logger.error("Unsupported field type '%s' found in node '%s' --> " + \
                "op '%s' --> attr '%s'.", type(fieldValue), node.name,
                node.op, attrName)
        # exit() 

def create_sym(node):
    conv_attr = { 'layout': 'NCHW',
    'pad': (1, 1),
    'num_filter': 16,
    'dilate': (1, 1),
    'num_group': 1,
    'stride': (1, 1),
    'no_bias': False,
    'kernel': (3, 3) }
    node = mx.sym.Convolution(X, W, **conv_attr, name=conv_name)

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

allAttrs = { 'Conv2D': { 'strides', 'data_format', 'padding',
             'dilations', 'use_cudnn_on_gpu', 'T' },
             'Const': { 'value', 'dtype' } }

currRealizedOps = { }

def convert_model(pbfile):

    # load the original model
    logger = logging.getLogger("Loading Original Model")
    tfparser = TFParser(pbfile)
    tfgraph = tfparser.parse()
    logger.info("Model successfully loaded from path [%s].", pbfile)

    # parse the original model
    logger = logging.getLogger("Parsing Original Graph")
    inputs, outputs, attrs, ninps, res = {}, {}, {}, [], {}
    for node in tfgraph.node:
        if node.op not in currSupportedOps:
            logger.error("the op '%s' of node '%s' is not supported",
                    node.op, node.name)
            exit()
        inputs[node.name] = set(node.input)
        for inp in node.input:
            if inp not in outputs.keys():
                outputs[inp] = set()
            outputs[inp].add(node.name)
        attrs[node.name] = { attrName: convert_field(node, attrName, attrFields) \
                           for attrName, attrFields in node.attr.items() }
        if not len(node.input):
            ninps.append(node.name)
        else:
            res[node.name] = len(node.input)
    logger.info("Model successfully parsed.")

    # topo sort
    logger = logging.getLogger("Topo sort")
    topos = []
    while len(ninps):
        cname = ninps.pop()
        topos.append(cname)
        if cname not in outputs.keys():
            continue
        for name in outputs[cname]:
            if res[name] > 1:
                res[name] -= 1
            else:
                res.pop(name)
                ninps.append(name)
    logger.info("Topo sort completed.")


modelfile = {
            "/tmp/tf/resnet50_v1/model.pb",
            "/data/tfmodels/inception_v3/model.pb",
            "/data/tfmodels/keras/inception_v3/model.pb",
            "/data/tfmodels/mobilenet/model.pb"
            }

utils.log_init()

for pb in modelfile:
    convert_model(pb)



print(ts)
