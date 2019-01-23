import os
import mxnet as mx
import numpy as np
import logging
from mxnet import nd
from mxnet import autograd
from math import floor
from mxnet.test_utils import get_mnist_iterator
import copy

def quantize_to(x, bits=8):
    max_v = nd.max(nd.abs(x))
    if max_v == 0:
        return x.astype(np.int8), 8
    int_len = nd.ceil(nd.log2(max_v)).asscalar()
    sb = bits - int_len
    f = 2 ** sb
    y = nd.floor(x * f)
    y = nd.clip(y, a_min=-2**(bits-1), a_max=2**(bits-1) - 1)
    return y, sb

class CVMDense(mx.operator.CustomOp):

    def __init__(self, num_hidden, act_type=None):
        self.num_hidden = num_hidden
        self.act_type = act_type
        self.prec_bits = 8

    def forward(self, is_train, req, in_data, out_data, aux):
        x_int = in_data[0]
        sbits = in_data[3]

        w = in_data[1]
        b = in_data[2]

        w_int, w_sb = quantize_to(w, self.prec_bits)
        total_sb = w_sb + sbits
        # b_int is in range int32, no need to clip
        b_int = (b / (2 ** total_sb)).floor()

        y = mx.nd.dot(x_int.astype(np.float32), w_int.T.astype(np.float32))
        y = mx.nd.add(y, b_int.astype(np.float32))
        y, y_sb = quantize_to(y, self.prec_bits)
        y_sb += total_sb

        self.assign(out_data[0], req[0], y)
        self.assign(out_data[1], req[1], y_sb)
        #if not is_train:
        #    print (is_train)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dy = out_grad[0] * 2 ** out_data[1]
        x = in_data[0] / 2 ** in_data[3]

        w = in_data[1]
        b = in_data[2]
        y = out_data[0]
        dx = mx.nd.dot(dy, w)
        dw = mx.nd.dot(dy.T, x)
        db = dy.sum(axis=0)
        assert dw.shape == in_grad[1].shape
        assert db.shape == in_grad[2].shape
        self.assign(in_grad[0], req[0], dx)
        self.assign(in_grad[1], req[1], dw)
        self.assign(in_grad[2], req[2], db)

@mx.operator.register("cvm.dense")

class CVMDenseProp(mx.operator.CustomOpProp):

    def __init__(self, num_hidden):
        super(CVMDenseProp, self).__init__(True)
        # we use constant bias here to illustrate how to pass arguments
        # to operators. All arguments are in string format so you need
        # to convert them back to the type you want.
        self.num_hidden = np.long(num_hidden)

    def list_arguments(self):
        return ['data', 'weight', 'bias', 'sbits']
        # return ['data']

     # def list_auxiliary_states(self):
         # return ['data_sb', 'weight_int', 'weight_sb', 'bias_int']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output', 'osbits']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        weight_shape = (self.num_hidden, in_shapes[0][1])
        bias_shape = (self.num_hidden,)
        output_shape = (data_shape[0], self.num_hidden)
        return [data_shape, weight_shape, bias_shape, (1,)], [output_shape, (1,)], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype, np.float32], [dtype, np.float32], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return CVMDense(self.num_hidden)
def test_autograd():
    a = nd.ones((1, 1)) * 0.05
    b = nd.ones((1, 1))
    q_a = (a * 256).floor()
    q_b = (b * 256).floor()
    a.attach_grad(), b.attach_grad()
    c = a * b
    c.attach_grad()
    with autograd.record():
        loss = (1 - c) ** 2 / 2
    q_c = q_a * q_b / 256 / 256
    q_loss = (1 - q_c) ** 2 / 2

    da, db, dc = autograd.grad(loss, [a, b, c], 1 - q_loss, retain_graph=True)
    # print (loss, a, b, c, '|', 'ag', a.grad, 'bg', b.grad, 'cg', c.grad, 'lossg', loss.grad)


if __name__ == "__main__":
    test_autograd()
    # print (quantize_to(mx.nd.array([0.001, 0.001, -0.001, 0]), 16))
