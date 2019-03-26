import os
import mxnet as mx
import numpy as np
import copy
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

        self.real_params = []

    def forward(self, is_train, req, in_data, out_data, aux):
        x_int = in_data[0]
        x_sb = in_data[1]

        w_int = in_data[2]
        w_sb = in_data[3]

        b_int = in_data[4]
        b_sb = in_data[5]

        if is_train and len(self.real_params) == 0:
            print("Init once")
            self.real_params.append(w_int * 2 ** w_sb)
            self.real_params.append(b_int * 2 ** b_sb)

        xw_sb = x_sb + w_sb
        # let bits of B consistent with X*W, clip value with int16 range.
        b_quan = np.clip( (b_int * (2 ** (xw_sb - b_sb))).floor(),
                        a_min=-32768, a_max=32766)

        y = mx.nd.dot(x_int.astype(np.float32), w_int.T.astype(np.float32))
        y = mx.nd.add(y, b_quan.astype(np.float32))
        y, y_sb = quantize_to(y, self.prec_bits)
        y_sb += xw_sb

        self.assign(out_data[0], req[0], y)
        self.assign(out_data[1], req[1], y_sb)

        # w_int, w_sb = quantize_to(w, self.prec_bits)
        # total_sb = w_sb + sbits
        # b_int = (b / (2 ** total_sb)).floor()

        # y = mx.nd.dot(x_int.astype(np.float32), w_int.T.astype(np.float32))
        # y = mx.nd.add(y, b_int.astype(np.float32))
        # y, y_sb = quantize_to(y, self.prec_bits)
        # y_sb += total_sb

        # self.assign(out_data[0], req[0], y)
        # self.assign(out_data[1], req[1], y_sb)
        #if not is_train:
        #    print (is_train)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dy = out_grad[0]

        x = in_data[0]
        # w = in_data[2] * 2 ** in_data[3]
        w = in_data[2]
        b = in_data[4]
        # b = in_data[4] * 2 ** in_data[5]
        y = out_data[0]

        dx = mx.nd.dot(dy, w)
        dw = mx.nd.dot(dy.T, x)
        db = dy.sum(axis=0)

        self.real_params[0] -= dw
        self.real_params[1] -= db
        #  aux[0] += dw
        #  aux[1] += db

        dw_int, dw_sb = quantize_to(self.real_params[0])
        db_int, db_sb = quantize_to(self.real_params[1])
        #  dw_int, dw_sb = quantize_to(aux[0])
        #  db_int, db_sb = quantize_to(aux[1])
        #  print(self.real_params[0].shape, self.real_params[1].shape, dw_int.shape, dw_sb.shape)
        #  print(len(in_grad), in_grad[2].shape, in_grad[3].shape)
        assert dw_int.shape == in_grad[2].shape
        #  assert dw_sb.shape == in_grad[3].shape
        assert db_int.shape == in_grad[4].shape
        #  assert db_sb.shape == in_grad[5].shape

        #  self.assign(in_grad[2], req[2], dw_int - in_data[2])
        #  self.assign(in_grad[3], req[3], dw_sb - in_data[3])
        #  self.assign(in_grad[4], req[4], db_int - in_data[4])
        #  self.assign(in_grad[5], req[5], db_sb - in_data[5])

        self.assign(in_grad[0], req[0], dx)
        self.assign(in_grad[2], req[2], in_data[2] - dw_int)
        self.assign(in_grad[3], req[3], in_data[3] - dw_sb)
        self.assign(in_grad[4], req[4], in_data[4] - db_int)
        self.assign(in_grad[5], req[5], in_data[5] - db_sb)

@mx.operator.register("cvm.dense")

class CVMDenseProp(mx.operator.CustomOpProp):

    def __init__(self, num_hidden, is_train=True):
        super(CVMDenseProp, self).__init__(True)
        # we use constant bias here to illustrate how to pass arguments
        # to operators. All arguments are in string format so you need
        # to convert them back to the type you want.
        self.num_hidden = np.long(num_hidden)
        self.is_train = is_train

    def list_arguments(self):
        return ['data', 'data_bits', 'weight', 'weight_bits', 'bias', 'bias_bits']

    #  def list_auxiliary_states(self):
        #  return ['real_weight', 'real_bias'] if self.is_train else []

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output', 'osbits']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        weight_shape = (self.num_hidden, in_shapes[0][1])
        bias_shape = (self.num_hidden,)
        output_shape = (data_shape[0], self.num_hidden)
        return [data_shape, (1,), weight_shape, (1,), bias_shape, (1,)], [output_shape, (1,)], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, np.float32, dtype, np.float32, dtype, np.float32], [dtype, np.float32], []

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
