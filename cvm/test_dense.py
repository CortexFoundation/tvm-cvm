import tvm
from tvm.contrib import cvm
import numpy as np


def test_dense(use_bias=True):
    if not tvm.get_global_func("tvm.contrib.cvm.dense.forward", True):
        print("skip because cvm is not enabled...")
        return

    inType = 'int8'
    outType = 'int32'
    m = tvm.var("m")
    k = tvm.var("k")
    n = tvm.var("n")
    data = tvm.placeholder((m,k), dtype=inType, name='data')
    weight = tvm.placeholder((k,n), dtype=inType, name='weight')
    bias = tvm.placeholder((n,), dtype=outType, name='bias')
    if use_bias :
        out = cvm.dense(data, weight, bias)
    else:
        out = cvm.dense(data, weight)
    s =  tvm.create_schedule(out.op)

    def verify():
        target = 'llvm'
        ctx = tvm.context(target, 0)
        if use_bias:
            f = tvm.build(s, [data, weight, bias, out], target, target_host="llvm", name="dense")
        else:
            f = tvm.build(s, [data, weight, out], target, target_host="llvm", name="dense")
        m = 1024
        k = 64
        n = 1024
        d = tvm.nd.array(np.random.randint(low=-128, high=127, size=(m,k)).astype(data.dtype), ctx)
        w = tvm.nd.array(np.random.randint(low=-128, high=127, size=(k,n)).astype(weight.dtype), ctx)
        b = tvm.nd.array(np.random.randint(low=-128, high=127, size=(n,)).astype(bias.dtype), ctx)
        o = tvm.nd.array(np.zeros((m,n), outType), ctx)
        if use_bias:
            f(d, w, b, o)
        else:
            f(d, w, o)
        answer = np.dot(d.asnumpy().astype(o.dtype), w.asnumpy().astype(o.dtype))
        if use_bias:
            answer += b.asnumpy()
        tvm.testing.assert_allclose(o.asnumpy(), answer)
        print("pass")

    verify()

if __name__ == "__main__":
    test_dense(use_bias=False)

