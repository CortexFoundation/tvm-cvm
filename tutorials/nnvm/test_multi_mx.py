import nnvm.compiler
import nnvm.symbol as sym
import tvm
import numpy as np
from tvm.contrib import graph_runtime, util
import mxnet as mx

if __name__ == '__main__':
    dtype = 'float32'
    data = mx.symbol.Variable('data')  # or some constructed NN
    op = mx.symbol.FullyConnected(data=data,
    num_hidden=512,
    name='FC1')
    print (op)
    print (op.tojson())
    z = op
    # compile graph from mxnet, and get symbol and params
    sym, params = nnvm.frontend.from_mxnet(op)

    shape = (1024, 1024)
    print('compile sys')
    graph, lib, params = nnvm.compiler.build(sym, target='cuda', shape={"x": shape, "y": shape}, params=params, dtype=dtype)
    print(graph.ir())
    print('create module')
    module = graph_runtime.create(graph, lib, tvm.gpu(0))

    print('random input data')
    x_np = (np.random.uniform(size=shape)*8).astype(dtype)
    y_np = (np.random.uniform(size=shape)*8).astype(dtype)
    #print(x_np)
    #print(y_np)

    module.set_input(x=x_np, y=y_np)
    module.run()
    out = module.get_output(0, tvm.nd.empty(shape, dtype))
    print(out.asnumpy())
