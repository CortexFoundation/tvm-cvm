from tensorflow_parser import TFParser
from tensorflow.core.framework import tensor_pb2 as tpb2
from tensorflow.core.framework import tensor_shape_pb2 as tspb2
from tensorflow.core.framework import attr_value_pb2 as apb2
import mxnet as mx
from mxnet import nd

import os
import logging
import utils

import heapq

ts = set()
fieldOrgTypes = (int, bool, float)

def convert_field(node, attrName, attrFields, logger):
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
        logger.error("Unsupported field type '%s' found in node '%s' --> " + \
                "op '%s' --> attr '%s'.", type(fieldValue), node.name,
                node.op, attrName)
        exit()

node_map_out = {}

import tensor_util
def convert_tfnode(tfnode, graph, params, logger=logging):
    name, op_name = tfnode.name, tfnode.op
    attr, inputs = tfnode.attr, tfnode.input
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
            elif k not in ('dtype', '_output_shapes', '_class'):
                raise NotImplementedError \
                    ("Other attributes for a Const(param) Node {} ? .".format(key))
    elif op_name in ['Placeholder', 'PlaceholderWithDefault']:
        input_shape = \
            tensor_util.tensor_shape_proto_to_list(attr['shape'].shape)
        dtype = tensor_util.tensor_type_to_numpy(attr['dtype'].type)
        graph[name] = mx.sym.var(name, shape=input_shape, dtype=dtype)
    else:
        inputs = []
        for in_name in inputs:
            input_entry = in_name.split(":")
            node_name = input_entry[0]
            assert node_name in graph
            child = graph[node_name]
            if len(child) > 1 and len(input_entry) > 1:
                eid = int(input_entry[1])
            else:
                eid = 0
            inputs.append(child[eid])

        # TODO(ryt): convert operators

    return

    attrs = { k:convert_field(tfnode, k, v, logger) for k, v in tfnode.attr.items() }
    if tfnode.op not in currSupportedOps:
        logger.info("Not supported op '%s'", tfnode.op)
    mxattrs = {}
    if tfnode.op == 'Const':
        dtype = attrs['dtype']
        _, shape, data = attrs['value']
        
        # ts.add((dtype, type(data)))
        
        pass
    elif tfnode.op == 'FusedBatchNorm':
        pass
    elif tfnode.op == 'Relu':
        pass
    elif tfnode.op == 'Conv2D':
        pass
         
    return 0

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
        node_map_out[node.name] = node
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
    logger.info("Topo sort completed.")
    return topos

def convert_model(pbfile):
    # load the original model
    logger = logging.getLogger("Loading Original Model")
    tfparser = TFParser(pbfile)
    tfgraph = tfparser.parse()
    logger.info("Model successfully loaded from path [%s].", pbfile)

    graph, params = {}, {}
    
    for tfnode in topo_sort(tfgraph):
        # print ("%-16s" % tfnode.op,
        #        "%-40s" % tfnode.name,
        #        tfnode.input)
        convert_tfnode(tfnode, graph, params)
        # sym = create_symbol(tfnode, nodes, logger)

    # symbol
    # for name in topos:
        # sym = create_symbol(name, nodes)



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
