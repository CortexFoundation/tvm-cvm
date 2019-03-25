import tvm
from tvm.contrib import cvm
import numpy as np


def test_conv2d():
    in_channel = 3
    out_channel = 32
    filter_h = 3
    filter_w = 3
    pad_h = 1
    pad_w = 1
    stride_h = 1
    stride_w = 1
    dilation_h = 1
    dilation_w = 1

    xshape = [4, 3, 32, 32]
    if not tvm.get_global_func("tvm.contrib.cvm.conv2d.forward", True):
        print("skip because cvm is not enabled...")
        return
    print(tvm.get_global_func("tvm.contrib.cvm.conv2d.forward"))
    wshape = cvm.conv2d_w_shape(in_channel, out_channel, filter_h, filter_w)

    X = tvm.placeholder(xshape, dtype='int8', name='X')
    W = tvm.placeholder(wshape, dtype='int8', name='W')
    Y = cvm.conv2d_forward(X, W)
    yshape = cvm.conv2d_output_shape(list(X.shape), list(W.shape))#[x.value for x in Y.shape]
    s =  tvm.create_schedule(Y.op)
    print(s)
    def verify():
        ctx = tvm.cpu(0)
        f2 = tvm.lower(s, [X, W, Y])
        print(tvm.lower(s, [X, W, Y], simple_mode=True))
        f = tvm.build(f2, name="conv2d")
        print (f)
        x = tvm.nd.array(np.random.uniform(-127, 127, xshape).astype(np.int8), ctx)
        w = tvm.nd.array(np.random.uniform(-127, 127, wshape).astype(np.int8), ctx)
        y = tvm.nd.array(np.random.uniform(-1, 1, yshape).astype(np.int32), ctx)
        f(x, w, y)
#        print(y)

    verify()

if __name__ == "__main__":
    test_conv2d()

