import transformer as tfm

import sym_pass as spass
import sym_utils as sutils
import mxnet as mx
import nnvm

if __name__ == "__main__":
    prefix = "/tmp/resnet50_v1.sym.quantize"
    sym = mx.sym.load(prefix + ".json")
    params = mx.nd.load(prefix + ".params")

#    count = 0
#    graph = {}
#    for op in sutils.topo_sort(sym):
#        name, op_name = op.attr('name'), op.attr('op_name')
#        childs, attr = sutils.sym_iter(op.get_children()), op.list_attr()
#        if sutils.is_var(op, params):
#            op = mx.sym.var(str(count), attr=attr)
#        elif childs is None:
#            op = sutils.get_mxnet_op(op_name)(**attr, name=str(count))
#        else:
#            childs = [graph[c.attr('name')] for c in childs]
#            op = sutils.get_mxnet_op(op_name)(*childs, **attr, name=str(count))
#        count += 1
#        graph[name] = op
#
#    nodes = [graph[s.attr('name')] for s in sym]
#    if len(nodes) > 1:
#        nodes = [mx.sym.Group(nodes)]
#    sym = nodes[0]
#    with open("/tmp/tmp_byr.json", "w") as fout:
#        fout.write(sym.tojson())
#
#    exit()

    # sym = mx.sym.load("./data/resnet18_v1.json")
    # params = mx.nd.load("./data/resnet18_v1.params")
    sym, params = tfm.init(sym, params, (1, 3, 224, 224))
    nnvm_sym, nnvm_params = tfm.compile(sym, params)

    print (sym.debug_str)

    nnvm.sym.Symbol
