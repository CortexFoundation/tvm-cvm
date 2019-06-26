import numpy as np

from tvm import relay
from tvm.relay import testing
import tvm
from tvm.contrib import graph_runtime


data = tvm.relay.var('data', relay.TensorType((1, 28), dtype='int32'))
weight0 = tvm.relay.var('dense0_weight', relay.TensorType((16, 28), dtype='int32'))
weight1 = tvm.relay.var('dense1_weight', relay.TensorType((10, 16), dtype='int32'))
cond = tvm.relay.greater(data, tvm.relay.const(0))
net = tvm.relay.where(cond, data, relay.const(-2) * data)
net = tvm.relay.nn.dense(net, weight=weight0)
net = tvm.relay.nn.dense(net, weight=weight1)
net = relay.Function(relay.ir_pass.free_vars(net), net)
opt_level = 0
target = tvm.target.cuda()
with relay.build_config(opt_level=opt_level):
    deploy_graph, lib, params = relay.build_module.build(
        net, target, target_host="stackvm")
print (lib.get_source())
print (lib.imported_modules[0].get_source())
with open('/tmp/start_relay_cuda.json', "w") as fout:
    fout.write(deploy_graph)
exit()
