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
                        stride_h = 1,
                        stride_w = 1):

    """Get output shape of 2D convolution

    Paramters
    ---------
    stride_h: int
        height stride
    stride_w: int
        width stride
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
    return list([x_shape[0].value, w_shape[0].value,
                 x_shape[2].value // stride_h, x_shape[3].value // stride_w])


def conv2d_forward(x, w, strides = [1, 1]):

    """Create an extern op that compute 2D convolution with CVM

    Parameters
    ----------
    x: Tensor
        input feature map
    w: Tensor
        convolution weight
    strides: (int, int) width and height stride

    Returns
    -------
    y: Tensor
        The result tensor
    """

    [stride_h, stride_w] = strides 
    oshape = conv2d_output_shape(list(x.shape), list(w.shape), stride_h, stride_w)

    return _api.extern(
        oshape, [x, w],
        lambda ins, outs: _intrin.call_packed(
            "tvm.contrib.cvm.conv2d.forward",
            stride_h,
            stride_w,
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
    print(data.shape, weight.shape, oshape)
    if bias is not None:
        return _api.extern(
            oshape, [data, weight, bias],
            lambda ins, outs: _intrin.call_packed(
                "tvm.contrib.cvm.dense.forward",
                ins[0],
                ins[1],
                ins[2],
                outs[0]), name="y", dtype='int32')
    else:
        return _api.extern(
                oshape, [data, weight],
                lambda ins, outs: _intrin.call_packed(
                    "tvm.contrib.cvm.dense.forward",
                    ins[0],
                    ins[1],
                    outs[0]), name="y", dtype='int32')
