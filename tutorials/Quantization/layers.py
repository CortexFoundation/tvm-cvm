# -*- coding:utf-8 -*-
import os
import mxnet as mx
import numpy as np
import logging
from mxnet import nd
from mxnet import autograd
from math import floor
from mxnet.test_utils import get_mnist_iterator
import copy

class IntDense(mx.operator.CustomOp):

    def __init__(self, bit, num_hidden):
        self.num_bit = nd.array([bit])
        self.num_hidden = num_hidden

    def forward(self, is_train, req, in_data, out_data, aux):
        if is_train:
            x = in_data[0]
            w = in_data[1]
            b = in_data[2]
            quan_w = in_data[3]
            quan_b = in_data[4]

            x_shift_bit = in_data[5]  # int_w = in_data[3]
            # int_b = in_data[4]
            #x, x_shift_bit = self.int_quantize(x)
            quan_w, quan_b, w_shift_bit = self.int_quantize_double(w, b)
            y = out_data[0]
            #if x.max() > 127 or x.min() < -128:
             #   print('False')
            y[:] = mx.nd.add(mx.nd.dot(x, quan_w.T), quan_b)
            y, y_shift_bit = self.int_quantize(y)
            self.assign(out_data[0], req[0], mx.nd.array(y))

        else:
            x = in_data[0]
            w = in_data[3]
            b = in_data[4]
            y_shift_bit = in_data[5]
            w, b, w_shift_bit = self.int_quantize_double(w, b)
            # self.assign(in_data[1], 'write', w)
            # self.assign(in_data[2], 'write', b)
            y = out_data[0]

            # int_w = int_w.astype('int8')
            # int_b = int_b.astype('int8')

            #int_x, x_shift_bit = self.int_quantize(x)

            y[:] = mx.nd.add(mx.nd.dot(x, w.T), b)
            y * (2 ** y_shift_bit)
            self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dy = out_grad[0]

        x = in_data[0]
        w = in_data[1]
        b = in_data[2]
        y = out_data[0]


        x.attach_grad(), w.attach_grad(), b.attach_grad()

        with autograd.record():
            y[:] = mx.nd.add(mx.nd.dot(x, w.T), b)

        dx, dw, db = autograd.grad(y, [x, w, b], dy, retain_graph=True)
        y, y_shift_bit = self.int_quantize(y)

        self.assign(in_grad[0], req[0], dx)
        self.assign(in_grad[1], req[0], dw)
        self.assign(in_grad[2], req[0], db)

        self.assign(in_data[3], req[0], w)
        self.assign(in_data[4], req[0], b)
        self.assign(in_data[5], req[0], y_shift_bit)

    def int_quantize(self, x):
        max = nd.max(nd.abs(x))
        if max != 0:
            int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** (frac_len)).astype('float32')
            y = ((x * f)).floor()
            y = nd.clip(y, a_min=-128, a_max=127)
            return y, frac_len
        return x, 0

    def int_quantize_double(self, x, w):
        max1 = nd.max(nd.abs(x))
        max2 = nd.max(nd.abs(w))
        if max1 > max2:
            max = max1
        else:
            max = max2
        if max != 0:
            int_len = (nd.ceil(nd.log2(max))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** (frac_len)).astype('float32')
            int_x = ((x * f)).floor()
            int_w = ((w * f)).floor()
            return int_x, int_w, frac_len
        return x, w, 0

@mx.operator.register("intdense")

class IntDenseProp(mx.operator.CustomOpProp):

    def __init__(self, bit, num_hidden):
        super(IntDenseProp, self).__init__(True)
        # we use constant bias here to illustrate how to pass arguments
        # to operators. All arguments are in string format so you need
        # to convert them back to the type you want.
        self.num_bit = bit
        self.num_hidden = np.long(num_hidden)

    def list_arguments(self):
        return ['data', 'weight', 'bias', 'quan_weight', 'quan_bias', 'y_shift_bit']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        weight_shape = (self.num_hidden, in_shapes[0][1])
        bias_shape = (self.num_hidden,)
        output_shape = (data_shape[0], self.num_hidden)
        # print(data_shape)
        return [data_shape, weight_shape, bias_shape, weight_shape, bias_shape, [1]], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype, dtype, dtype, dtype], [dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return IntDense(self.num_bit, self.num_hidden)

class IntConv(mx.operator.CustomOp):
    def __init__(self, bit, num_filter, kernel, stride=[], pad=[], no_bias=False, workspace=1024, name=None):
        self.num_bit = nd.array([bit])
        self.num_filter = num_filter
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.no_bias = no_bias
        self.workspace = workspace
        self.name = name

    def forward(self, is_train, req, in_data, out_data, aux):
        if is_train:
            x = in_data[0]
            w = in_data[1]
            quan_w = in_data[3]
            quan_b = in_data[4]
            x_shift_bit = in_data[5]
            #if x.max() > 127 or x.min() < -128:
             #   print('False')
            y = out_data[0]
            last_shift_bit = out_data[1]
            if self.no_bias:
                quan_w, w_shift_bit = self.int_quantize(w)
                y[:] = nd.Convolution(data=x, weight=quan_w, kernel=self.kernel, num_filter=self.num_filter,
                                      stride=self.stride, pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
                                      name=self.name)

            else:

                b = in_data[2]

                quan_w, quan_b, w_shift_bit = self.int_quantize_double(w, b)
                y = nd.Convolution(data=quan_x, weight=quan_w, bias=quan_b, kernel=self.kernel,
                                   num_filter=self.num_filter,
                                   stride=self.stride,
                                   pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
                                   name=self.name)

            y, y_shift_bit = self.int_quantize(y)
            self.assign(out_data[0], req[0], mx.nd.array(y))
            self.assign(out_data[1], req[0], mx.nd.array(y_shift_bit))

        else:
            quan_x = in_data[0]
            w = in_data[1]
            quan_w = in_data[3]
            y_shift_bit = in_data[5]
            y = out_data[0]

            # quan_x, x_shift_bit = self.int_quantize(x)

            if self.no_bias:
                quan_w_2, w_shift_bit = self.int_quantize(w)
                # print('sum', (quan_w_2 != quan_w).sum())
                # self.assign(in_data[1], req[0], quan_w)
                y[:] = nd.Convolution(data=quan_x, weight=quan_w_2, kernel=self.kernel, num_filter=self.num_filter,
                                      stride=self.stride, pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
                                      name=self.name)
            # y, y_shift_bit = self.int_quantize(y)

            # print('----------------int conv------------------')
            # print(y)
            else:
                b = in_data[2]
                quan_w, quan_b, w_shift_bit = self.int_quantize_double(w, b)
                y = nd.Convolution(data=quan_x, weight=quan_w, bias=quan_b, kernel=self.kernel,
                                   num_filter=self.num_filter,
                                   stride=self.stride, pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
                                   name=self.name)

            y = (y * (2 ** y_shift_bit)).floor()

            self.assign(out_data[0], req[0], mx.nd.array(y))
            self.assign(out_data[1], req[0], y_shift_bit)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dx = in_grad[0]
        dw = in_grad[1]
        x = in_data[0]
        w = in_data[1]
        # b = in_data[2]
        # w = in_data[3]
        # quan_b = in_data[4]
        x_shift_bit = in_data[5]
        dy = out_grad[0]

        # x, x_shift_bit = self.int_quantize(x)
        quan_w, w_shift_bit = self.int_quantize(w)
        y = out_data[0]
        if self.no_bias:
            x.attach_grad(), quan_w.attach_grad()
            with autograd.record():
                y[:] = nd.Convolution(data=x, weight=quan_w, kernel=self.kernel, num_filter=self.num_filter,
                                      stride=self.stride,
                                      pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
                                      name=self.name)
            dx, dw = autograd.grad(y, [x, quan_w], dy, retain_graph=True)
            y, y_shift_bit = self.int_quantize(y)
            # print(y_shift_bit)
            # y_shift_bit = (x_shift_bit * 0.3 + y_shift_bit * 0.7).floor()
            self.assign(in_grad[0], req[0], dx / (2 ** y_shift_bit))
            self.assign(in_grad[1], req[0], dw / (2 ** w_shift_bit))

            self.assign(in_data[3], req[0], quan_w)
            # self.assign(in_data[3], req[0], quan_b)
            self.assign(in_data[5], req[0], y_shift_bit)

        else:
            b = in_data[2]
            b, b_shift_bit = self.int_quantize(b)
            x.attach_grad(), w.attach_grad(), b.attach_grad()
            with autograd.record():
                y[:] = nd.Convolution(data=x, weight=w, bias=b, kernel=self.kernel, num_filter=self.num_filter,
                                      stride=self.stride,
                                      pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
                                      name=self.name)
            dx, dw, db = autograd.grad(y, [x, w, b], dy, retain_graph=True)
            self.assign(in_grad[0], req[0], dx / (2 ** x_shift_bit))
            self.assign(in_grad[1], req[0], dw / (2 ** w_shift_bit))
            self.assign(in_grad[2], req[0], db / (2 ** b_shift_bit))

            self.assign(in_data[2], req[0], quan_w)
            self.assign(in_data[3], req[0], quan_b)
            self.assign(in_data[4], req[0], x_shift_bit)
            self.assign(in_data[5], req[0], y_shift_bit)

    def int_quantize(self, x):
        max = nd.max(nd.abs(x))
        if max != 0:
            int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** (frac_len)).astype('float32')
            y = ((x * f)).floor()
            y = nd.clip(y, a_min=-128, a_max=127)
            return y, frac_len
        return x, 0

    def int_quantize_double(self, x, w):
        max1 = nd.max(nd.abs(x))
        max2 = nd.max(nd.abs(w))
        if max1 > max2:
            max = max1
        else:
            max = max2
        if max != 0:
            int_len = (nd.ceil(nd.log2(max))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** (frac_len)).astype('float32')
            int_x = ((x * f)).floor()
            int_w = ((w * f)).floor()
            return int_x, int_w, frac_len
        return x, w, 0

@mx.operator.register("intconv")

class IntConvProp(mx.operator.CustomOpProp):
    def __init__(self, bit, num_filter, kernel, stride=[], pad=[], no_bias=False, workspace=1024, name=None):
        super(IntConvProp, self).__init__(True)
        # we use constant bias here to illustrate how to pass arguments
        # to operators. All arguments are in string format so you need
        # to convert them back to the type you want.
        self.num_bit = np.long(bit)
        self.num_filter = np.long(num_filter)
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.no_bias = no_bias
        self.workspace = workspace
        self.name = name

    def list_arguments(self):
        return ['data', 'weight', 'bias', 'quan_weight', 'quan_bias', 'x_shift_bit']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output', 'y_shift_bit']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        weight_shape = [self.num_filter, in_shapes[0][1], int(self.kernel[1]), int(self.kernel[1])]
        bias_shape = [self.num_filter]
        width = self.f(in_shapes[0][2], int(self.kernel[1]), int(self.pad[1]), int(self.stride[1]), 1)
        output_shape = [in_shapes[0][0], self.num_filter, int(width), int(width)]
        # print(weight_shape, bias_shape, width, output_shape)
        return [data_shape, weight_shape, bias_shape, weight_shape, bias_shape, [1]], [output_shape, [1]], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype, dtype, dtype, dtype], [dtype, dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return IntConv(self.num_bit, self.num_filter, self.kernel, self.stride,
                       self.pad, self.no_bias, self.workspace, self.name)

    def f(self, x, k, p, s, d):
        return floor((x + 2 * p - d * (k - 1) - 1) / s) + 1

class IntAdd(mx.operator.CustomOp):
	def __init__(self, name=''):
		self.name = name

	def forward(self, is_train, req, in_data, out_data, aux):
		input_1 = in_data[0]
		input_2 = in_data[1]

		output = out_data[0]

		output = input_1 * (2**-1) + input_2 * (2**-1)
		output = output.round()
		self.assign(out_data[0], req[0], output)

	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		dx1 = in_grad[0]
		dx2 = in_grad[1]

		dy = out_grad[0]

		dx1 = dy / (2**-1)
		dx2 = dy / (2**-1)

		self.assign(in_grad[0], req[0], dx1)
		self.assign(in_grad[1], req[0], dx2)
@mx.operator.register("short_cut")
class IntAddProp(mx.operator.CustomOpProp):
	def __init__(self, name=''):
		super(IntAddProp, self).__init__(True)
		self.name = name
	def list_arguments(self):
		return ['data1', 'data2']

	def list_outputs(self):
		return ['output']

	def infer_shape(self, in_shapes):
		data_shape = in_shapes[0]

		return [data_shape, data_shape], [data_shape], []

	def infer_type(self, in_type):
		dtype = in_type[0]
		return [dtype, dtype], [dtype], []

	def create_operator(self, ctx, in_shapes, in_dtypes):
		return IntAdd( name = self.name)

class IntBatch(mx.operator.CustomOp):
    def __init__(self, bit, eps=0.001, momentum=0.9, fix_gamma=False, name=''):
        self.num_bit = nd.array([bit])
        self.eps = eps
        self.momentum = momentum
        self.fix_gamma = fix_gamma
        self.name = name

    def forward(self, is_train, req, in_data, out_data, aux):
        #global batch
        #batch += 1
        x = in_data[0]
        gamma = in_data[1]
        beta = in_data[2]
        moving_mean = in_data[3]
        moving_var = in_data[4]
        new_gamma = in_data[5]
        new_beta = in_data[6]
        y_shift_bit = in_data[7]
        last_shift_bit = in_data[8]
        #if batch % 200 == 0:
        #    writer.add_histogram('batch1_input', x.asnumpy(), batch, bins='sturges')
        #if x.max() > 127 or x.min() < -128:
            #print(x)
        y = out_data[0]
        if is_train:
            mean = nd.mean(x, axis=(0, 2, 3))
            var = nd.array(np.var(x.asnumpy(), axis=(0, 2, 3)))

            quan_gamma = gamma / (nd.sqrt(var + self.eps))
            quan_beta = beta - mean * gamma / nd.sqrt(var + self.eps)
            # print(quan_gamma)
            quan_gamma = quan_gamma * (2 ** last_shift_bit)
            quan_gamma, quan_beta, gamma_shift_bit = self.int_quantize_double(quan_gamma, quan_beta)

            y = nd.BatchNorm(x, gamma=quan_gamma, beta=quan_beta, moving_mean=nd.zeros(shape=moving_mean.shape),
                             moving_var=nd.ones(shape=moving_var.shape), eps=1e-5,
                             momentum=self.momentum, fix_gamma=True, name=self.name)
            y, y_shift_bit = self.int_quantize(y)
        # print('train gamma', quan_gamma)
        else:
            # quan_gamma, quan_beta, gamma_shift_bit = self.int_quantize_double(quan_gamma, quan_beta)
            y = nd.BatchNorm(x, gamma=nd.ones(shape=moving_var.shape), beta=new_beta,
                             moving_mean=nd.zeros(shape=moving_mean.shape),
                             moving_var=nd.ones(shape=moving_var.shape), eps=1e-5,
                             momentum=self.momentum, fix_gamma=True, name=self.name)
            # y, y_shift_bit = self.int_quantize(y)
            y = y * (2 ** y_shift_bit)
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dx = in_grad[0]
        dgamma = in_grad[1]
        dbeta = in_grad[2]

        x = in_data[0]
        gamma = in_data[1]
        beta = in_data[2]
        mean = in_data[3]
        var = in_data[4]
        new_gamma = in_data[5]
        new_beta = in_data[6]
        y_shift_bit = in_data[7]
        last_shift_bit = in_data[8]

        y = out_data[0]
        dy = out_grad[0]

        mean = nd.mean(x, axis=(0, 2, 3))
        var = nd.array(np.var(x.asnumpy(), axis=(0, 2, 3)))

        quan_gamma = gamma / (nd.sqrt(var + self.eps))
        quan_beta = beta - mean * gamma / nd.sqrt(var + self.eps)

        # quan_gamma = nd.clip(nd.floor(nd.log2(quan_gamma)), a_min=-3, a_max=0)
        # quan_gamma = 2**(quan_gamma)
        quan_gamma = quan_gamma * (2 ** last_shift_bit)
        # quan_beta, beta_shift_bit = self.int_quantize(quan_beta)
        quan_gamma, quan_beta, gamma_shift_bit = self.int_quantize_double(quan_gamma, quan_beta)
        x.attach_grad(), quan_gamma.attach_grad(), quan_beta.attach_grad()
        # print(quan_gamma)

        with autograd.record():
            y = nd.BatchNorm(x, gamma=quan_gamma, beta=quan_beta, moving_mean=nd.zeros(shape=mean.shape),
                             moving_var=nd.ones(shape=var.shape), eps=self.eps,
                             momentum=self.momentum, fix_gamma=False, name=self.name)
            y, y_shift_bit = self.int_quantize(y)
        # print(quan_gamma)

        dx, dgamma, dbeta = autograd.grad(y, [x, quan_gamma, quan_beta], dy, retain_graph=True)

        self.assign(in_grad[0], req[0], dx / 2 ** y_shift_bit)
        self.assign(in_grad[1], req[0], dgamma / 2 ** (gamma_shift_bit + last_shift_bit))
        self.assign(in_grad[2], req[0], dbeta / 2 ** gamma_shift_bit)

        self.assign(in_data[5], req[0], quan_gamma)
        self.assign(in_data[6], req[0], quan_beta)
        self.assign(in_data[7], req[0], y_shift_bit)

    def quantize(self, x):
        max = nd.max(nd.abs(x))
        if max != 0:
            int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** (frac_len)).astype('float32')
            y = ((x * f)).floor() * (1 / f)
            return y
        return x

    def int_quantize(self, x):
        max = nd.max(nd.abs(x))
        if max != 0:
            int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** (frac_len)).astype('float32')
            y = ((x * f)).floor()
            y = nd.clip(y, a_min=-128, a_max=127)

            return y, frac_len
        return x, 0

    def int_quantize_double(self, x, w):
        max1 = nd.max(nd.abs(x))
        max2 = nd.max(nd.abs(w))
        if max1 > max2:
            max = max1
        else:
            max = max2
        if max != 0:
            int_len = (nd.ceil(nd.log2(max))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** (frac_len)).astype('float32')
            int_x = ((x * f)).floor()
            int_w = ((w * f)).floor()
            return int_x, int_w, frac_len
        return x, w, 0

@mx.operator.register("batchnorm_int")

class IntBatchProp(mx.operator.CustomOpProp):
    def __init__(self, bit, eps=0.001, momentum=0.9, fix_gamma=1, name=''):
        super(IntBatchProp, self).__init__(True)
        self.num_bit = bit
        self.eps = eps
        self.momentum = momentum
        self.fix_gamma = fix_gamma
        self.name = name

    def list_arguments(self):
        return ['data', 'gamma', 'beta', 'moving_mean', 'moving_var', 'quan_gamma', 'quan_beta', 'y_shift_bit',
                'last_shift_bit']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        gamma_shape = [in_shapes[0][1]]
        beta_shape = [in_shapes[0][1]]
        moving_mean_shape = [in_shapes[0][1]]
        moving_var_shape = [in_shapes[0][1]]
        output_shape = in_shapes[0]
        return [data_shape, gamma_shape, beta_shape, moving_mean_shape,
                moving_var_shape, gamma_shape, beta_shape, [1], [1]], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype, dtype], [dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return IntBatch(self.num_bit, self.eps, self.momentum, self.fix_gamma, self.name)
