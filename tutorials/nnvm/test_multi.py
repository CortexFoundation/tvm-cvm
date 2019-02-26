import nnvm.compiler
import nnvm.symbol as sym
import tvm
import numpy as np
from tvm.contrib import graph_runtime, util

if __name__ == '__main__':
    dtype = 'int8'
    x = sym.Variable('x', dtype=dtype)
    y = sym.Variable('y', dtype=dtype)
    z = sym.matmul(x, y)
    compute_graph = nnvm.graph.create(z)
    print(compute_graph.ir())

    shape = (1024, 1024)
    graph, lib, params = nnvm.compiler.build(compute_graph, target='cuda', shape={"x": shape, "y": shape}, dtype=dtype)
    print(graph.ir())

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
