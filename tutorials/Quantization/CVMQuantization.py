import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import autograd
from math import floor

def quantize_vector(x, bits=8):
    """Quantize vertor with precision 'bits'

    Parameters
    ----------
    x: NDArray
        shape is (1, n)
    bits: int
        vector after quantize preserve bits' precision

    Returns
    -------
    y, sb: vector after quantization should be left_shift 'sb' bit
        to backward original value.
    """
    max_v = nd.max(nd.abs(x))
    if max_v == 0:
        return x.astype(np.int8), 8
    int_len = nd.ceil(nd.log2(max_v)).asscalar()
    sb = bits - int_len
    f = 2 ** sb
    y = nd.floor(x * f)
    y = nd.clip(y, a_min=-2**(bits-1), a_max=2**(bits-1) - 1)
    return y, sb


class CVMConv(mx.operator.CustomOp):
    def __init__(self, num_filter, kernel, stride=[], pad=[], no_bias=False, workspace=1024, name=None):
        self.num_filter = num_filter
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.no_bias = no_bias
        self.workspace = workspace
        self.name = name

        self.prec_bits = 8

    def forward(self, is_train, req, in_data, out_data, aux):
        x_int = in_data[0]
        sbits = in_data[3]

        w = in_data[1]
        b = in_data[2]

        w_int, w_sb = quantize_vector(w, self.prec_bits)
        total_sb = w_sb + sbits

        if self.no_bias:
            y = nd.Convolution(
                    data=x_int,
                    weight=w_int,
                    kernel=self.kernel,
                    num_filter=self.num_filter,
                    stride=self.stride,
                    pad=self.pad,
                    no_bias=self.no_bias,
                    workspace=self.workspace,
                    name=self.name)
        else:
            b_int = (b / (2 ** total_sb)).floor()
            y = nd.Convolution(
                    data=x_int,
                    weight=w_int,
                    bias=b_int,
                    kernel=self.kernel,
                    num_filter=self.num_filter,
                    stride=self.stride,
                    pad=self.pad,
                    no_bias=self.no_bias,
                    workspace=self.workspace,
                    name=self.name)

        y, y_sb = quantize_vector(y, self.prec_bits)
        y_sb += total_sb

        self.assign(out_data[0], req[0], y)
        self.assign(out_data[1], req[1], y_sb)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        dy = out_grad[0] * 2 ** out_data[1]

        x = in_data[0] / 2 ** in_data[3]
        w = in_data[1]
        b = in_data[2]

        x.attach_grad(), w.attach_grad()
        if self.no_bias:
            with autograd.record():
                y = nd.Convolution(
                        data=x,
                        weight=w,
                        # bias=b_int,
                        kernel=self.kernel,
                        num_filter=self.num_filter,
                        stride=self.stride,
                        pad=self.pad,
                        no_bias=self.no_bias,
                        workspace=self.workspace,
                        name=self.name)
            dx, dw = autograd.grad(y, [x, w], dy, retain_graph=True)
        else:
            with autograd.record():
                y = nd.Convolution(
                        data=x,
                        weight=w,
                        bias=b_int,
                        kernel=self.kernel,
                        num_filter=self.num_filter,
                        stride=self.stride,
                        pad=self.pad,
                        no_bias=self.no_bias,
                        workspace=self.workspace,
                        name=self.name)
            dx, dw, db = autograd.grad(y, [x, w, b], dy, retain_graph=True)

        self.assign(in_grad[0], req[0], dx)
        self.assign(in_grad[1], req[1], dw)
        if not self.no_bias:
            self.assign(in_grad[2], req[2], db)

@mx.operator.register("cvm.conv")
class CVMConvProp(mx.operator.CustomOpProp):
    def __init__(self, num_filter, kernel, stride=[], pad=[], no_bias=False, workspace=1024, name=None):
        super(CVMConvProp, self).__init__(True)
        # we use constant bias here to illustrate how to pass arguments
        # to operators. All arguments are in string format so you need
        # to convert them back to the type you want.
        self.num_filter = np.long(num_filter)
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.no_bias = no_bias
        self.workspace = workspace
        self.name = name

    def list_arguments(self):
        return ['data', 'weight', 'bias', 'sbits']

    def list_outputs(self):
        #  this can be omitted if you only have 1 output.
        return ['output', 'y_sb']

    def infer_shape(self, in_shapes):
        data_shape = in_shapes[0]
        weight_shape = [self.num_filter, in_shapes[0][1], int(self.kernel[1]), int(self.kernel[1])]
        bias_shape = [self.num_filter]
        width = self.f(in_shapes[0][2], int(self.kernel[1]), int(self.pad[1]), int(self.stride[1]), 1)
        output_shape = [in_shapes[0][0], self.num_filter, int(width), int(width)]
        # print(weight_shape, bias_shape, width, output_shape)
        return [data_shape, weight_shape, bias_shape, (1,)], [output_shape, (1,)], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype, dtype, dtype, np.float32], [dtype, dtype], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        #  create and return the CustomOp class.
        return CVMConv(self.num_filter, self.kernel, self.stride,
                       self.pad, self.no_bias, self.workspace, self.name)

    def f(self, x, k, p, s, d):
        return floor((x + 2 * p - d * (k - 1) - 1) / s) + 1

