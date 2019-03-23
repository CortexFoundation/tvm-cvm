"""External function interface to CVM library."""
# pylint: disable-msg=C0103
from .. import api as _api
from .. import intrin as _intrin

def conv2d_w_shape(in_channel,
                   out_channel,
                   filter_h,
                   filter_w):
    """Get weight shape for a 2D convolution

    Parameters
    ----------
    in_channel: int
        input channel
    out_channel: int
        output channel
    filter_h: int
        filter height
    filter_w: int
        filter width

    Returns
    -------
    wshape: list
        weight shape
    """
    return [out_channel, in_channel, filter_h, filter_w]


def conv2d_output_shape(x_shape,
                        w_shape,
                        pad_h = 0,
                        pad_w = 0,
                        stride_h = 1,
                        stride_w = 1,
                        dilation_h = 0,
                        dilation_w = 0):

    """Get output shape of 2D convolution

    Paramters
    ---------
    tensor_format: int
        0: CUDNN_TENSOR_NCHW
        1: CUDNN_TENSOR_NHWC
        2: CUDNN_TENSOR_NCHW_VECT_C
    pad_h: int
        height pad
    pad_w: int
        weight pad
    stride_h: int
        height stride
    stride_w: int
        width stride
    dilation_h: int
        height dilation
    dilation_w: int
        width dilation
    x_shape: list
        input shape
    w_shape: list
        weight shape

    Returns
    -------
    oshape: list
        output shape
    """
    assert isinstance(x_shape, list)
    assert isinstance(w_shape, list)
    assert len(x_shape) == 4
    assert len(w_shape) == 4
    return list([x_shape[0].value,
                 w_shape[0].value,
                 x_shape[2].value // stride_h,
                 x_shape[3].value // stride_w])


def conv2d_forward(x,
                   w,
                   stride_h=1,
                   stride_w=1,
                   pad_h=0,
                   pad_w=0,
                   dilation_h=1,
                   dilation_w=1,
                   conv_mode=1,
                   tensor_format=0):

    """Create an extern op that compute 2D convolution with CVM

    Parameters
    ----------
    x: Tensor
        input feature map
    w: Tensor
        convolution weight
    stride_h: int
        height stride
    stride_w: int
        width stride
    pad_h: int
        height pad
    pad_w: int
        weight pad
    dilation_h: int
        height dilation
    dilation_w: int
        width dilation
    conv_mode: int
        0: CUDNN_CONVOLUTION
        1: CUDNN_CROSS_CORRELATION
    tensor_format: int
        0: CVM_TENSOR_NCHW
        1: CVM_TENSOR_NHWC
        2: CVM_TENSOR_NCHW_VECT_C
    algo: int
        Forward algorithm, get index from ```algo_to_index``` function
        if algo == -1, the best algo will be chosen by CUDNN

    Returns
    -------
    y: Tensor
        The result tensor
    """

    oshape = conv2d_output_shape(list(x.shape),
                                 list(w.shape),
                                 pad_h,
                                 pad_w,
                                 stride_h,
                                 stride_w,
                                 dilation_h,
                                 dilation_w)

    return _api.extern(
        oshape, [x, w],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.cvm.conv2d.forward",
            conv_mode,
            tensor_format,
            0,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dilation_h,
            dilation_w,
            ins[0],
            ins[1],
            outs[0]), name="y", dtype='int32')




def dense(data,
          weight,
          bias=None):

    """Create an extern op that compute 2D convolution with CVM

    Parameters
    ----------
    data: Tensor
        input data int8
    weight: Tensor
        input weight int8
    bias: Tensor
        input bias int32
    Returns
    -------
    y: Tensor
        The result tensor int32
    """

    oshape = list([data.shape[0], weight.shape[1]])
    if bias is not None:
        print("bias is not none")
        return _api.extern(
            oshape, [data, weight, bias],
            lambda ins, outs: _intrin.call_packed(
                "tvm.contrib.cvm.dense.forward",
                ins[0],
                ins[1],
                ins[2],
                outs[0]), name="y", dtype='int32')
    else:
        print("bias is none")
        return _api.extern(
                oshape, [data, weight],
                lambda ins, outs: _intrin.call_packed(
                    "tvm.contrib.cvm.dense.forward",
                    ins[0],
                    ins[1],
                    outs[0]), name="y", dtype='int32')
