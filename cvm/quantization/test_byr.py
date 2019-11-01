import transformer as tfm

import sym_pass as spass
import sym_utils as sutils
from sym_utils import topo_visit_transformer
import mxnet as mx
import nnvm
from ut_base import *

class TestFuseMultiplyInputs(TfmTest):
    def test_fmi(self):
        d1 = mx.sym.var('d1', shape=(2, 3))
        d2 = mx.sym.var('d2', shape=(2, 4))
        d3 = mx.sym.var('d3', shape=(2, 3))
        # d1 = mx.sym.var('d1', shape=(1, 2, 3))
        # d2 = mx.sym.var('d2', shape=(4, 2))
        # d3 = mx.sym.var('d3', shape=(2, 3))
        # op = mx.sym.Group([d1, d2, d3])
        op = mx.sym.concat(d1, d2, d3)
        sym = newnewchange(op, {})

        data = mx.sym.var('data', shape=(20,))
        # s1 = mx.sym.slice(data, begin=(0,), end=(6,))
        # r1 = mx.sym.reshape(s1, shape=(1, 2, 3))
        # s2 = mx.sym.slice(data, begin=(6,), end=(14,))
        # r2 = mx.sym.reshape(s2, shape=(4, 2))
        # s3 = mx.sym.slice(data, begin=(14,), end=(20,))
        # r3 = mx.sym.reshape(s3, shape=(2, 3))
        s1 = mx.sym.slice(data, begin=(0,), end=(6,))
        r1 = mx.sym.reshape(s1, shape=(2, 3))
        s2 = mx.sym.slice(data, begin=(6,), end=(14,))
        r2 = mx.sym.reshape(s2, shape=(2, 4))
        s3 = mx.sym.slice(data, begin=(14,), end=(20,))
        r3 = mx.sym.reshape(s3, shape=(2, 3))
        # des = mx.sym.Group([r1, r2, r3])
        des = mx.sym.concat(r1, r2, r3)

        self._assert_equal(sym, des)

def change(sym, params):
    data_input = {}
    infer_shapes = tfm.infer_shape(sym, params)
    for node in sutils.topo_sort(sym):
        name, op_name = node.attr('name'), node.attr('op_name')
        if sutils.is_inputs(node, params):
            data_input[name] = node
    dim_sum = 0
    dim_per = []
    dims = []
    for key, val in data_input.items():
        assert len(infer_shapes[key]) == 1
        dims.append(infer_shapes[key][0])
        dot = 1
        for it in dims[-1]:
            dot *= it
        dim_per.append(dot)
        dim_sum += dot

    data = mx.sym.var('data', shape=(dim_sum,))
    create_var = locals()
    first = 0
    last = 0
    indata = []
    for i in range(len(dim_per)):
        last = first + dim_per[i]
        create_var['data'+str(i)] = mx.sym.slice(data, begin=(first,),
                            end=(last,))
        create_var['data'+str(i)] = mx.sym.reshape(create_var['data'+str(i)],
                    shape=dims[i])
        indata.append(create_var['data'+str(i)])
        first = last
    out = mx.sym.Group(indata)
    return out


def model():
    d1 = mx.sym.var('d1', shape=(1,2,3))
    d2 = mx.sym.var('d2', shape=(2,4))
    d3 = mx.sym.var('d3', shape=(3,2))
    sym = mx.sym.Group([d1, d2, d3])
    sym = change(sym)
    print(sym.tojson())
    # data_input = {}
    # params = {}
    # infer_shapes = tfm.infer_shape(sym, params)
    # print(infer_shapes)
    # for node in sutils.topo_sort(sym):
    #     name, op_name = node.attr('name'), node.attr('op_name')
    #     if sutils.is_inputs(node, params):
    #         data_input[name] = node
    #         print('data', name, op_name, node, infer_shapes[name])
    #     else:
    #         print('other', name, op_name, node, infer_shapes[name])
    # print('here', data_input)
    # dim_sum = 0
    # dim_per = []
    # dims = []
    # for key, val in data_input.items():
    #     assert len(infer_shapes[key]) == 1
    #     dims.append(infer_shapes[key][0])
    #     dot = 1
    #     for it in dims[-1]:
    #         dot *= it
    #     dim_per.append(dot)
    #     dim_sum += dot
    #     print(dim_sum)

    # data = mx.sym.var('data', shape=(dim_sum,))
    # create_var = locals()
    # first = 0
    # last = 0
    # indata = []
    # for i in range(len(dim_per)):
    #     last = first + dim_per[i]
    #     print(first, last, type(first), type(last), 'gen')
    #     create_var['data'+str(i)] = mx.sym.slice(data, begin=(first,),
    #                         end=(last,))
    #     create_var['data'+str(i)] = mx.sym.reshape(create_var['data'+str(i)],
    #                 shape=dims[i])
    #     indata.append(create_var['data'+str(i)])
    #     first = last
    # out = mx.sym.Group(indata)
    # sym, params = tfm.init(out, {})
    # print (sym.tojson())

def newnewchange(sym, params):
    infer_shapes = tfm.infer_shape(sym, params)
    dim_sum, dim_per, dims = 0, {}, {}
    def _sum_input(node, params, **kwargs):
        name = node.attr('name')
        nonlocal dim_sum, dim_per, dims
        if sutils.is_inputs(node, params):
            dot = 1
            dims[name] = infer_shapes[name][0]
            for it in dims[name]:
                dot *= it
            dim_per[name] = dot
            dim_sum += dot
    topo_visit_transformer(sym, params, _sum_input)
    data_sum = mx.sym.var('data_input', shape=(dim_sum,))
    first, last = 0, 0
    def _change_node(op, params, graph, **kwargs):
        name = op.attr('name')
        if sutils.is_inputs(op, params):
            nonlocal first, last
            last = first + dim_per[name]
            op = mx.sym.slice(data_sum, begin=(first,), end=(last,))
            op = mx.sym.reshape(op, shape=dims[name])
            first = last
        return op
    sym, params = topo_visit_transformer(sym, params, _change_node)
    return sym

def newchange(sym, params):
    infer_shapes = tfm.infer_shape(sym, params)
    dim_sum = 0
    dim_per = {}
    dims = {}
    for node in sutils.topo_sort(sym):
        name, op_name = node.attr('name'), node.attr('op_name')
        if sutils.is_inputs(node, params):
            assert len(infer_shapes[name]) == 1
            dot = 1
            dims[name] = infer_shapes[name][0]
            for it in dims[name]:
                dot *= it
            dim_per[name] = dot
            dim_sum += dot
    data_sum = mx.sym.var('data_input', shape=(dim_sum,))
    graph = {}
    first = 0
    last = 0
    for op in sutils.topo_sort(sym):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sutils.sym_iter(op.get_children()), op.list_attr()
        if sutils.is_inputs(op, params):
            last = first + dim_per[name]
            op = mx.sym.slice(data_sum, begin=(first,), end=(last,))
            op = mx.sym.reshape(op, shape=dims[name])
            first = last
        if childs is not None:
            new_childs = [graph[child.attr('name')] for child in childs]
            op = sutils.get_mxnet_op(op_name)(*new_childs, **attr, name=name)
        graph[name] = op
    nodes = [graph[s.attr('name')] for s in sym]
    if len(nodes) > 1:
        nodes = [mx.sym.Group(nodes)]
    sym = nodes[0]
    return sym

if __name__ == "__main__":
    #model()
    exit()


    prefix = "/home/byr/resnet50_v1.sym.quantize"
    sym = mx.sym.load(prefix + ".json")
    params = mx.nd.load(prefix + ".params")

    count = 0
    graph = {}
    infer_shapes = tfm.infer_shape(sym, params, [1, 3, 224, 224])
    for op in sutils.topo_sort(sym):
        name, op_name = op.attr('name'), op.attr('op_name')
        childs, attr = sutils.sym_iter(op.get_children()), op.list_attr()
        print(op, op.attr('inputs'), op.attr('op_name') if childs is None else childs[0].attr('name'))
        if sutils.is_var(op, params):
            op = mx.sym.var(str(count), attr=attr)
        elif childs is None:
            op = sutils.get_mxnet_op(op_name)(**attr, name=str(count))
        else:
            childs = [graph[c.attr('name')] for c in childs]
            op = sutils.get_mxnet_op(op_name)(*childs, **attr, name=str(count))
        print(op, op.attr('inputs'))
        count += 1
        graph[name] = op

    nodes = [graph[s.attr('name')] for s in sym]
    if len(nodes) > 1:
        nodes = [mx.sym.Group(nodes)]
    sym = nodes[0]
    with open("/home/byr/tmp_byr.json", "w") as fout:
        fout.write(sym.tojson())

    exit()

    # sym = mx.sym.load("./data/resnet18_v1.json")
    # params = mx.nd.load("./data/resnet18_v1.params")
    sym, params = tfm.init(sym, params, (1, 3, 224, 224))
    nnvm_sym, nnvm_params = tfm.compile(sym, params)

    with open('/tmp/debug_byr.txt', 'w') as fout:
        fout.write(nnvm_sym.debug_str())
