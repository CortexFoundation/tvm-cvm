from tensorflow_parser import TFParser
from tensorflow.core.framework import tensor_pb2 as tpb2
from tensorflow.core.framework import tensor_shape_pb2 as tspb2
from tensorflow.core.framework import attr_value_pb2 as apb2
import mxnet as mx

import os
import logging
import utils

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
        ffields = fieldValue.ListFields()
        shapes = tuple([dim.size for dim in \
                ffields[1][1].ListFields()[0][1]])
        return (ffields[0][1], shapes, ffields[2][1])
    elif isinstance(fieldValue, apb2.AttrValue.ListValue):
        return tuple(fieldValue.ListFields()[0][1])
    else:
        logger.error("Unsupported field type '%s' found in node '%s' --> " + \
                "op '%s' --> attr '%s'.", type(fieldValue), node.name,
                node.op, attrName)
        # exit() 

utils.log_init()

# load
logger = logging.getLogger("Loading Original Model")
pbfile = "/tmp/tf/resnet50_v1/model.pb"
tfparser = TFParser(pbfile)
tfgraph = tfparser.parse()
logger.info("Model Successfully Loaded from path [%s]", pbfile)

# parse
logger = logging.getLogger("Parsing Original Graph")
supportedOps = {'Const', 'Conv2D', 'Pad', 'Identity',
                'FusedBatchNorm', 'MatMul', 'Relu',
                'Softmax', 'Mean', 'MaxPool',
                'MaxPool', 'BiasAdd', 'Placeholder',
                'Add'}
allAttrs = { 'Conv2D': { 'strides', 'data_format', 'padding',
             'dilations', 'use_cudnn_on_gpu', 'T' },
             'Const': { 'value', 'dtype' } }

nodeMap = {}

for node in tfgraph.node:
    inputs = [inp for inp in node.input]
    attrs = { attrName: convert_field(node, attrName, attrFields) \
        for attrName, attrFields in node.attr.items() }
    nodeMap[node.name] = { 'inputs': inputs, 'attrs': attrs }

logger.info("Model Successflly Parsed")
print(ts)
