# -*- coding:utf-8 -*-
import os
import mxnet as mx
import numpy as np
import logging
from numpy import *
from mxnet import nd
from mxnet import autograd
from math import floor
from mxnet.test_utils import get_mnist_iterator

class Batch_Quan(mx.operator.CustomOp):
    def __init__(self, bit, eps=0.001, momentum=0.9, fix_gamma=1, name=''):
        self.num_bit = nd.array([bit])
        self.eps = eps
        self.momentum = momentum
        self.fix_gamma = fix_gamma
        self.name = name

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        gamma = in_data[1]
        beta = in_data[2]
        moving_mean = in_data[3]
        moving_var = in_data[4]
        # print(x.sum())
        y = out_data[0]

        if is_train:
            mean = nd.mean(x, axis=(0, 2, 3))
            var = nd.array(np.var(x.asnumpy(), axis=(0, 2, 3)))
            #print(moving_mean ,self.momentum, mean)
            moving_mean = moving_mean * self.momentum + mean * (1 - self.momentum)
            moving_var = moving_var * self.momentum + var * (1 - self.momentum)
            self.assign(in_data[3], req[0], moving_mean)
            self.assign(in_data[4], req[0], moving_var)

        else:
            mean = moving_mean
            var = moving_var

        quan_gamma = self.quantize(gamma / (nd.sqrt(var + self.eps)))
        quan_beta = self.quantize(beta - mean * gamma / nd.sqrt(var + self.eps))

        y = nd.BatchNorm(x, gamma=quan_gamma, beta=quan_beta, moving_mean=nd.zeros(shape=moving_mean.shape),
                         moving_var=nd.ones(shape=moving_var.shape), eps=self.eps,
                         momentum=self.momentum, fix_gamma=self.fix_gamma, name=self.name)

        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dx = in_grad[0]
        dgamma = in_grad[1]
        dbeta = in_grad[2]

        x = in_data[0]
        gamma = in_data[1]
        beta = in_data[2]

        y = out_data[0]
        dy = out_grad[0]

        mean = nd.mean(x, axis=(0, 2, 3))
        var = nd.array(np.var(x.asnumpy(), axis=(0, 2, 3)))

        quan_gamma = gamma
        quan_beta = beta

        x.attach_grad(), gamma.attach_grad(), beta.attach_grad()
        with autograd.record():
            y = nd.BatchNorm(x, gamma=quan_gamma, beta=quan_beta, moving_mean=mean,
                             moving_var=var, eps=self.eps,
                             momentum=self.momentum, fix_gamma=self.fix_gamma, name=self.name)

        dx, dgamma, dbeta = autograd.grad(y, [x, quan_gamma, quan_beta], dy, retain_graph=True)
        self.assign(in_grad[0], req[0], dx)
        self.assign(in_grad[1], req[0], dgamma)
        self.assign(in_grad[2], req[0], dbeta)

    def quantize(self, x):
        max = nd.max(nd.abs(x))
        if max != 0:
            int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** (frac_len)).astype('float32')
            y = ((x * f)).round() * (1 / f)
            return y
        return x


@mx.operator.register("batchnorm_quan")
class Batch_QuanProp(mx.operator.CustomOpProp):
    def __init__(self, bit, eps=0.001, momentum=0.9, fix_gamma=1, name=''):
        super(Batch_QuanProp, self).__init__(True)
        self.num_bit = bit
        self.eps = eps
        self.momentum = momentum
        self.fix_gamma = fix_gamma
        self.name = name

    def list_arguments(self):
        return ['data', 'gamma', 'beta', 'moving_mean', 'moving_var']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        gamma_shape = [in_shapes[0][1]]
        beta_shape = [in_shapes[0][1]]
        moving_mean_shape = [in_shapes[0][1]]
        moving_var_shape = [in_shapes[0][1]]
        output_shape = in_shapes[0]
        return [data_shape, gamma_shape, beta_shape, moving_mean_shape, moving_var_shape], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype, dtype, dtype], [dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Batch_Quan(self.num_bit, self.eps, self.momentum, self.fix_gamma, self.name)


class Conv_Quan(mx.operator.CustomOp):
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
        x = in_data[0]
        w = in_data[1]
        y = out_data[0]
        #print(self.no_bias)

        quan_x = self.quantize(x)
        quan_w = self.quantize(w)
        # print(quan_x.sum(), quan_w)
        if self.no_bias:
            y[:] = nd.Convolution(data=quan_x, weight=quan_w, kernel=self.kernel, num_filter=self.num_filter,
                                  stride=self.stride, pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
                                  name=self.name)

        else:
            b = in_data[2]
            quan_b = self.quantize(b)
            y = nd.Convolution(data=quan_x, weight=quan_w, bias=quan_b, kernel=self.kernel, num_filter=self.num_filter,
                               stride=self.stride,
                               pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
                               name=self.name)
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dx = in_grad[0]
        dw = in_grad[1]
        x = in_data[0]
        w = in_data[1]
        dy = out_grad[0]
        x = self.quantize(x)
        w = self.quantize(w)
        y = out_data[0]
        if self.no_bias :
            x.attach_grad(), w.attach_grad()
            with autograd.record():
                y[:] = nd.Convolution(data=x, weight=w, kernel=self.kernel, num_filter=self.num_filter,
                                      stride=self.stride,
                                      pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
                                      name=self.name)
            dx, dw = autograd.grad(y, [x, w], dy, retain_graph=True)
            self.assign(in_grad[0], req[0], dx)
            self.assign(in_grad[1], req[0], dw)
        else:
            b = in_data[2]
            b = self.quantize(b)
            x.attach_grad(), w.attach_grad(), b.attach_grad()
            with autograd.record():
                y[:] = nd.Convolution(data=x, weight=w, bias=b, kernel=self.kernel, num_filter=self.num_filter,
                                      stride=self.stride,
                                      pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
                                      name=self.name)
            dx, dw, db = autograd.grad(y, [x, w, b], dy, retain_graph=True)
            self.assign(in_grad[0], req[0], dx)
            self.assign(in_grad[1], req[0], dw)
            self.assign(in_grad[2], req[0], db)

    def quantize(self, x):
        max = nd.max(nd.abs(x))
        if max != 0:
            int_len = (nd.ceil(nd.log2(max))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** frac_len).astype('float32')
            y = (x * f).round() * (1 / f)
            return y
        return x


@mx.operator.register("conv_quan")
class Conv_QuanProp(mx.operator.CustomOpProp):
    def __init__(self, bit, num_filter, kernel, stride=[], pad=[], no_bias=False, workspace=1024, name=None):
        super(Conv_QuanProp, self).__init__(True)
        # we use constant bias here to illustrate how to pass arguments
        # to operators. All arguments are in string format so you need
        # to convert them back to the type you want.
        self.num_bit = long(bit)
        self.num_filter = long(num_filter)
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.no_bias = no_bias
        self.workspace = workspace
        self.name = name

    def list_arguments(self):
        return ['data', 'weight', 'bias', 'int_weight', 'int_bias']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        weight_shape = [self.num_filter, in_shapes[0][1], int(self.kernel[1]), int(self.kernel[1])]
        bias_shape = [self.num_filter]
        width = self.f(in_shapes[0][2], int(self.kernel[1]), int(self.pad[1]), int(self.stride[1]), 1)
        output_shape = [in_shapes[0][0], self.num_filter, int(width), int(width)]
        #print(weight_shape, bias_shape, width, output_shape)
        return [data_shape, weight_shape, bias_shape], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype], [dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return Conv_Quan(self.num_bit, self.num_filter, self.kernel, self.stride,
                         self.pad, self.no_bias, self.workspace, self.name)

    def f(self, x, k, p, s, d):
        return floor((x + 2 * p - d * (k - 1) - 1) / s) + 1


class Quantize(mx.operator.CustomOp):

    def __init__(self, bit):
        self.num_bit = nd.array([bit])

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        y = out_data[0]
        in_max = nd.max(nd.abs(x))
        if in_max != 0:
            int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** (frac_len)).astype('float32')
            y = ((x * f)).round() * (1 / f)
        y = x
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dy = out_grad[0]
        self.assign(in_grad[0], req[0], dy)


@mx.operator.register("quantize")
class QuantizeProp(mx.operator.CustomOpProp):
    def __init__(self, num_bit):
        super(QuantizeProp, self).__init__(True)
        # we use constant bias here to illustrate how to pass arguments
        # to operators. All arguments are in string format so you need
        # to convert them back to the type you want.
        self.num_bit = int(num_bit)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        output_shape = (data_shape)
        return [data_shape], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype], [dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return Quantize(self.num_bit)


class QuanDense(mx.operator.CustomOp):

    def __init__(self, bit, num_hidden):
        self.num_bit = nd.array([bit])
        self.num_hidden = num_hidden

    def forward(self, is_train, req, in_data, out_data, aux):
        x = in_data[0]
        w = in_data[1]
        b = in_data[2]
        #w = self.quantize(w)
        #b = self.quantize(b)
        y = out_data[0]

        y[:] = mx.nd.add(mx.nd.dot(x, w.T), b)
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dx = in_grad[0]
        dw = in_grad[1]
        db = in_grad[2]
        dy = out_grad[0]
        x = in_data[0]
        w = in_data[1]
        b = in_data[2]
        y = out_data[0]
        # x.attach_grad(), w.attach_grad(), b.attach_grad()
        # with autograd.record():
        #	y[:] = mx.nd.add(mx.nd.dot(x, w.T), b)
        # dw, dx, db = autograd.grad(y, [x, w, b], dy, retain_graph=True)
        dw[:] = mx.nd.dot(dy.T, x)
        dx[:] = mx.nd.dot(dy, w)
        db[:] = mx.nd.sum(dy, axis=0)
        self.assign(in_grad[0], req[0], dx)
        self.assign(in_grad[1], req[0], dw)
        self.assign(in_grad[2], req[0], db)

    def quantize(self, x):
        max = nd.max(nd.abs(x))
        if max != 0:
            int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
            num_bit = self.num_bit.as_in_context(x.context)
            frac_len = num_bit - int_len
            f = (2 ** (frac_len)).astype('float32')
            y = ((x * f)).round() * (1 / f)
            return y
        return x, 0


@mx.operator.register("quandense")
class QuanDenseProp(mx.operator.CustomOpProp):
    def __init__(self, bit, num_hidden):
        super(QuanDenseProp, self).__init__(True)
        # we use constant bias here to illustrate how to pass arguments
        # to operators. All arguments are in string format so you need
        # to convert them back to the type you want.
        self.num_bit = bit
        self.num_hidden = long(num_hidden)

    def list_arguments(self):
        return ['data', 'weight', 'bias']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        weight_shape = (self.num_hidden, in_shapes[0][1])
        bias_shape = (self.num_hidden,)
        output_shape = (data_shape[0], self.num_hidden)
        return [data_shape, weight_shape, bias_shape], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype], [dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return QuanDense(self.num_bit, self.num_hidden)


class IntDense(mx.operator.CustomOp):

	def __init__(self, bit, num_hidden):
		self.num_bit = nd.array([bit])
		self.num_hidden = num_hidden


	def forward(self, is_train, req, in_data, out_data, aux):
		if is_train:
			x = in_data[0]
			w = in_data[1]
			b = in_data[2]
			int_w = in_data[3]
			int_b = in_data[4]

			x, x_shift_bit = self.int_quantize(x)
			w, b, w_shift_bit = self.int_quantize_double(w, b)
			y = out_data[0]

			y[:] = mx.nd.add(mx.nd.dot(x, w.T), b)

			self.assign(out_data[0], req[0], mx.nd.array(y))
			self.assign(in_data[3], req[0], w)
			self.assign(in_data[4], req[0], b)
		else:
			x = in_data[0]
			int_w = in_data[3]
			int_b = in_data[4]

			y = out_data[0]

			#int_w = int_w.astype('int8')
			#int_b = int_b.astype('int8')

			int_x, x_shift_bit = self.int_quantize(x)
			y[:] = mx.nd.add(mx.nd.dot(int_x, int_w.T), int_b)
			self.assign(out_data[0], req[0], mx.nd.array(y))

	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		dx = in_grad[0]
		dw = in_grad[1]
		db = in_grad[2]
		dy = out_grad[0]

		x = in_data[0]
		w = in_data[1]
		b = in_data[2]
		y = out_data[0]

		int_x, x_shift_bit = self.int_quantize(x)
		int_w, int_b, w_shift_bit = self.int_quantize_double(w, b)

		int_x.attach_grad(), int_w.attach_grad(), int_b.attach_grad()

		with autograd.record():
			y[:] = mx.nd.add(mx.nd.dot(int_x, int_w.T), int_b)
		dx, dw, db = autograd.grad(y, [int_x, int_w, int_b], dy, retain_graph=True)

		#print('dx_origin:', dx)
		#print('dw_origin:', dw)
		#print('db_origin:', db)
		#print(x_shift_bit, w_shift_bit)
		#print('dx:', dx/(2**x_shift_bit), )
		#print('dw:', dw/(2**w_shift_bit), )
		#print('db:', db/(2**w_shift_bit))
		self.assign(in_grad[0], req[0], dx/(2**x_shift_bit))
		self.assign(in_grad[1], req[0], dw/(2**w_shift_bit))
		self.assign(in_grad[2], req[0], db/(2**w_shift_bit))

	def int_quantize(self, x):
		max = nd.max(nd.abs(x))
		if max != 0:
			int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
			num_bit = self.num_bit.as_in_context(x.context)
			frac_len = num_bit - int_len
			f = (2 ** (frac_len)).astype('float32')
			y = ((x * f)).round()
			return y, frac_len
		return x, 0

	def int_inference(self, x):
		max = nd.max(nd.abs(x))
		if max != 0:
			int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
			num_bit = self.num_bit.as_in_context(x.context)
			frac_len = num_bit - int_len
			f = (2 ** (frac_len)).astype('float32')
			y = ((x * f)).round()
			return y.astype('int8'), frac_len
		return x.astype('int8'), 0

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
			int_x = ((x * f)).round()
			int_w = ((w * f)).round()
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
        self.num_hidden = long(num_hidden)


    def list_arguments(self):
        return ['data', 'weight', 'bias', 'int_weight', 'int_bias']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        weight_shape = (self.num_hidden, in_shapes[0][1])
        bias_shape = (self.num_hidden,)
        output_shape = (data_shape[0], self.num_hidden)
        return [data_shape, weight_shape, bias_shape, weight_shape, bias_shape], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype, dtype, dtype], [dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return IntDense(self.num_bit, self.num_hidden)

class IntQuantize(mx.operator.CustomOp):

	def __init__(self, bit):
		self.num_bit = nd.array([bit])
		self.shift_bit = 0

	def forward(self, is_train, req, in_data, out_data, aux):
		x = in_data[0]
		y = out_data[0]

		y, self.shift_bit = self.int_quantize(x)

		self.assign(out_data[0], req[0], mx.nd.array(y))

	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		dy = out_grad[0]
		self.assign(in_grad[0], req[0], dy/(2**self.shift_bit))

	def int_quantize(self, x):
		max = nd.max(nd.abs(x))
		if max != 0:
			int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
			num_bit = self.num_bit.as_in_context(x.context)
			frac_len = num_bit - int_len
			f = (2 ** (frac_len)).astype('float32')
			y = ((x * f)).round()
			return y, frac_len
		return x, 0

@mx.operator.register("intquantize")

class IntQuantizeProp(mx.operator.CustomOpProp):
    def __init__(self, num_bit):
        super(IntQuantizeProp, self).__init__(True)
        # we use constant bias here to illustrate how to pass arguments
        # to operators. All arguments are in string format so you need
        # to convert them back to the type you want.
        self.num_bit = int(num_bit)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        output_shape = (data_shape)
        return [data_shape], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype], [dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return Quantize(self.num_bit)

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
			int_w = in_data[3]
			y = out_data[0]
			# print(self.no_bias)

			quan_x, x_shift_bit = self.int_quantize(x)
			quan_w, w_shift_bit = self.int_quantize(w)

			self.assign(in_data[3], req[0], quan_w)

			# print(quan_x.sum(), quan_w)
			if self.no_bias:
				y[:] = nd.Convolution(data=quan_x, weight=quan_w, kernel=self.kernel, num_filter=self.num_filter,
									  stride=self.stride, pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
									  name=self.name)

			else:
				b = in_data[2]
				int_b = in_data[4]
				quan_b, b_shift_bit = self.int_quantize(b)
				y = nd.Convolution(data=quan_x, weight=quan_w, bias=quan_b, kernel=self.kernel, num_filter=self.num_filter,
								   stride=self.stride,
								   pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
								   name=self.name)
				self.assign(in_data[4], req[0], quan_b)

			self.assign(out_data[0], req[0], mx.nd.array(y))
		else:
			x = in_data[0]
			int_w = in_data[3]
			y = out_data[0]

			quan_x, x_shift_bit = self.int_quantize(x)

			if self.no_bias:
				y[:] = nd.Convolution(data=quan_x, weight=int_w, kernel=self.kernel, num_filter=self.num_filter,
									  stride=self.stride, pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
									  name=self.name)

			else:
				int_b = in_data[4]
				y = nd.Convolution(data=quan_x, weight=int_w, bias=int_b, kernel=self.kernel, num_filter=self.num_filter,
								   stride=self.stride, pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
								   name=self.name)

			self.assign(out_data[0], req[0], mx.nd.array(y))



	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		dx = in_grad[0]
		dw = in_grad[1]
		x = in_data[0]
		w = in_data[1]

		dy = out_grad[0]

		x, x_shift_bit = self.int_quantize(x)
		w, w_shift_bit = self.int_quantize(w)
		y = out_data[0]
		if self.no_bias :
			x.attach_grad(), w.attach_grad()
			with autograd.record():
				y[:] = nd.Convolution(data=x, weight=w, kernel=self.kernel, num_filter=self.num_filter,
									  stride=self.stride,
									  pad=self.pad, no_bias=self.no_bias, workspace=self.workspace,
									  name=self.name)
			dx, dw = autograd.grad(y, [x, w], dy, retain_graph=True)
			self.assign(in_grad[0], req[0], dx/(2**x_shift_bit))
			self.assign(in_grad[1], req[0], dw/(2**w_shift_bit))
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
			self.assign(in_grad[0], req[0], dx/(2**x_shift_bit))
			self.assign(in_grad[1], req[0], dw/(2**w_shift_bit))
			self.assign(in_grad[2], req[0], db/(2**b_shift_bit))

	def int_quantize(self, x):
		max = nd.max(nd.abs(x))
		if max != 0:
			int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
			num_bit = self.num_bit.as_in_context(x.context)
			frac_len = num_bit - int_len
			f = (2 ** (frac_len)).astype('float32')
			y = ((x * f)).round()
			return y, frac_len
		return x, 0

@mx.operator.register("intconv")

class IntConvProp(mx.operator.CustomOpProp):
    def __init__(self, bit, num_filter, kernel, stride=[], pad=[], no_bias=False, workspace=1024, name=None):
        super(IntConvProp, self).__init__(True)
        # we use constant bias here to illustrate how to pass arguments
        # to operators. All arguments are in string format so you need
        # to convert them back to the type you want.
        self.num_bit = long(bit)
        self.num_filter = long(num_filter)
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.no_bias = no_bias
        self.workspace = workspace
        self.name = name

    def list_arguments(self):
        return ['data', 'weight', 'bias', 'int_weight', 'int_bias']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        weight_shape = [self.num_filter, in_shapes[0][1], int(self.kernel[1]), int(self.kernel[1])]
        bias_shape = [self.num_filter]
        width = self.f(in_shapes[0][2], int(self.kernel[1]), int(self.pad[1]), int(self.stride[1]), 1)
        output_shape = [in_shapes[0][0], self.num_filter, int(width), int(width)]
        #print(weight_shape, bias_shape, width, output_shape)
        return [data_shape, weight_shape, bias_shape, weight_shape, bias_shape], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype, dtype, dtype], [dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return IntConv(self.num_bit, self.num_filter, self.kernel, self.stride,
                         self.pad, self.no_bias, self.workspace, self.name)

    def f(self, x, k, p, s, d):
        return floor((x + 2 * p - d * (k - 1) - 1) / s) + 1

class IntBatch(mx.operator.CustomOp):
	def __init__(self, bit, eps=0.001, momentum=0.9, fix_gamma=1, name=''):
		self.num_bit = nd.array([bit])
		self.eps = eps
		self.momentum = momentum
		self.fix_gamma = fix_gamma
		self.name = name

	def forward(self, is_train, req, in_data, out_data, aux):
		x = in_data[0]
		gamma = in_data[1]
		beta = in_data[2]
		moving_mean = in_data[3]
		moving_var = in_data[4]
		# print(x.sum())
		y = out_data[0]

		if is_train:
			mean = nd.mean(x, axis=(0, 2, 3))
			var = nd.array(np.var(x.asnumpy(), axis=(0, 2, 3)))
			#print(moving_mean ,self.momentum, mean)
			moving_mean = moving_mean * self.momentum + mean * (1 - self.momentum)
			moving_var = moving_var * self.momentum + var * (1 - self.momentum)
			self.assign(in_data[3], req[0], moving_mean)
			self.assign(in_data[4], req[0], moving_var)

		else:
			mean = moving_mean
			var = moving_var

		quan_gamma = self.quantize(gamma / (nd.sqrt(var + self.eps)))
		quan_beta = self.quantize(beta - mean * gamma / nd.sqrt(var + self.eps))

		y = nd.BatchNorm(x, gamma=quan_gamma, beta=quan_beta, moving_mean=nd.zeros(shape=moving_mean.shape),
						 moving_var=nd.ones(shape=moving_var.shape), eps=self.eps,
						 momentum=self.momentum, fix_gamma=self.fix_gamma, name=self.name)

		self.assign(out_data[0], req[0], mx.nd.array(y))

	def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
		dx = in_grad[0]
		dgamma = in_grad[1]
		dbeta = in_grad[2]

		x = in_data[0]
		gamma = in_data[1]
		beta = in_data[2]

		y = out_data[0]
		dy = out_grad[0]

		mean = nd.mean(x, axis=(0, 2, 3))
		var = nd.array(np.var(x.asnumpy(), axis=(0, 2, 3)))

		quan_gamma = gamma
		quan_beta = beta

		x.attach_grad(), gamma.attach_grad(), beta.attach_grad()
		with autograd.record():
			y = nd.BatchNorm(x, gamma=quan_gamma, beta=quan_beta, moving_mean=mean,
							 moving_var=var, eps=self.eps,
							 momentum=self.momentum, fix_gamma=self.fix_gamma, name=self.name)

		dx, dgamma, dbeta = autograd.grad(y, [x, quan_gamma, quan_beta], dy, retain_graph=True)
		self.assign(in_grad[0], req[0], dx)
		self.assign(in_grad[1], req[0], dgamma)
		self.assign(in_grad[2], req[0], dbeta)

	def quantize(self, x):
		max = nd.max(nd.abs(x))
		if max != 0:
			int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
			num_bit = self.num_bit.as_in_context(x.context)
			frac_len = num_bit - int_len
			f = (2 ** (frac_len)).astype('float32')
			y = ((x * f)).round() * (1 / f)
			return y
		return x

	def int_quantize(self, x):
		max = nd.max(nd.abs(x))
		if max != 0:
			int_len = (nd.ceil(nd.log2(nd.max(nd.abs(x))))).astype('float32')
			num_bit = self.num_bit.as_in_context(x.context)
			frac_len = num_bit - int_len
			f = (2 ** (frac_len)).astype('float32')
			y = ((x * f)).round()
			return y, frac_len
		return x, 0

@mx.operator.register("batchnorm_quan")

class IntBatchProp(mx.operator.CustomOpProp):
	def __init__(self, bit, eps=0.001, momentum=0.9, fix_gamma=1, name=''):
		super(IntBatchProp, self).__init__(True)
		self.num_bit = bit
		self.eps = eps
		self.momentum = momentum
		self.fix_gamma = fix_gamma
		self.name = name

	def list_arguments(self):
		return ['data', 'gamma', 'beta', 'moving_mean', 'moving_var']

	def list_outputs(self):
		return ['output']

	def infer_shape(self, in_shapes):
		data_shape = in_shapes[0]
		gamma_shape = [in_shapes[0][1]]
		beta_shape = [in_shapes[0][1]]
		moving_mean_shape = [in_shapes[0][1]]
		moving_var_shape = [in_shapes[0][1]]
		output_shape = in_shapes[0]
		return [data_shape, gamma_shape, beta_shape, moving_mean_shape, moving_var_shape], [output_shape], []

	def infer_type(self, in_type):
		dtype = in_type[0]
		return [dtype, dtype, dtype, dtype, dtype], [dtype], []

	def create_operator(self, ctx, in_shapes, in_dtypes):
		return Batch_Quan(self.num_bit, self.eps, self.momentum, self.fix_gamma, self.name)

if __name__ == '__main__':
	batch_size = 400  # 定义每次处理数据的数量为64
	mnist = mx.test_utils.get_mnist()  # 使用内置的api读取mnist数据
	train_iter = mx.io.NDArrayIter(mnist['train_data'],
								   mnist['train_label'], batch_size, shuffle=True)  # 划分训练集和验证集，本代码仅用验证集做inference
	val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)


	data = mx.symbol.Variable('data')
	moving_mean = mx.symbol.Variable('moving_mean', init=mx.init.Zero())
	moving_var = mx.symbol.Variable('moving_var', init=mx.init.One())
	weight_int = mx.symbol.Variable('int_weight', init=mx.init.Zero())
	bias_int = mx.symbol.Variable('int_bias', init=mx.init.Zero())
	##This is the new defined layer
	#conv1 = mx.sym.Convolution(data=data, name='conv1', num_filter=3, kernel=(3,3), stride=(1,1), pad=(1,1), no_bias=False)

	conv1 = mx.symbol.Custom(data, name='conv1', op_type='intconv', bit=8, num_filter=3, kernel=(3, 3), stride=(1, 1),
							pad=(1, 1), no_bias=True)
	#batch1 = mx.symbol.Custom(data=conv1, moving_mean=moving_mean, moving_var=moving_var, name='batch1',
							  #op_type='batchnorm_quan', bit=8)
	flatten = mx.symbol.flatten(conv1, name='flatten')
	#fc1 = mx.symbol.contrib.quantized_fully_connected(data=flatten, name='fc1', num_hidden=128)
	fc1 = mx.symbol.Custom(data=flatten, name='fc1', op_type='intdense', bit=8, num_hidden=128)
	#fc1 = mx.symbol.FullyConnected(data=flatten, name = 'fc1', num_hidden = 64)
	act1 = mx.symbol.Activation(data=fc1, name='relu1', act_type="relu")
	#fc2 = mx.symbol.FullyConnected(data=act1, name='fc2', num_hidden=64)
	fc2 = mx.symbol.Custom(data=act1, name='fc2', op_type='intdense', bit=8, num_hidden=64)
	act2 = mx.symbol.Activation(data=fc2, name='relu2', act_type="relu")
	#fc3 = mx.symbol.FullyConnected(data=act2, name='fc3', num_hidden=10)
	fc3 = mx.symbol.Custom(data=act2, name='fc3', op_type='intdense', bit=8, num_hidden=10)
	mlp = mx.symbol.Softmax(fc3, name='softmax')
	mlp2 = mx.symbol.Custom(mlp, name='quantize', op_type='intquantize', num_bit=8)
	mod = mx.mod.Module(mlp2, context=mx.gpu(0))
	#mod.bind(data_shapes=val_iter.provide_data, label_shapes=val_iter.provide_label)
	#logging.basicConfig(level=logging.DEBUG)
	#mod.init_params(mx.init.)
	mod.fit(train_iter, val_iter, num_epoch=20, optimizer='adam', optimizer_params={'learning_rate':0.0001}, eval_metric='acc', initializer=mx.init.Xavier())

	#mod.fit(train_data=train_iter, eval_data=val_iter, optimizer='sgd',
	#optimizer_params={'learning_rate':0.01, 'momentum': 0.9},
	#num_epoch=10, eval_metric='acc',
	#batch_end_callback=mx.callback.Speedometer(100, 100))
	score = mod.score(val_iter, ['acc'])  # 进行inference
	mod.save_checkpoint('int_mnist', 100)
	print("Accuracy score is ", (score[0]))  # 输出预测精度
