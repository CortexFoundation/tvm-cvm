"""TVM operator fully connected compute."""
from __future__ import absolute_import
import tvm
from tvm.contrib import cvm
from .. import tag
from .. import math

def dense_default(data, weight, bias=None):
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1

    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = tvm.reduce_axis((0, in_dim), name='k')
    if data.dtype == 'int8' and weight.dtype == 'int8':
        matmul = tvm.compute((batch, out_dim), \
                             lambda i, j: tvm.sum(data[i, k].astype('int32') * weight[j, k].astype('int32'), axis=k), \
                             tag='dense')
        if bias is not None:
            matmul = tvm.compute((batch, out_dim), \
                                 lambda i, j: matmul[i, j].astype('int64') + bias[j].astype('int64'), \
                                 tag=tag.BROADCAST)
    else:
        matmul = tvm.compute((batch, out_dim), \
                             lambda i, j: tvm.sum(data[i, k] * weight[j, k], axis=k), \
                             tag='dense')
        if bias is not None:
            matmul = tvm.compute((batch, out_dim), \
                                 lambda i, j: matmul[i, j] + bias[j], \
                                 tag=tag.BROADCAST)


    return matmul


@tvm.target.override_native_generic_func("dense")
def dense(data, weight, bias=None):
    """Applies a linear transformation: :math:`Y = XW^T + b`.

    Parameters
    ----------
    data : tvm.Tensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    target = tvm.target.current_target()
    if "cvm" in target.libs:
        # check type here.
        return cvm.dense(data, weight)
    return dense_default(data, weight, bias)
    #return cpp.nn.dense(data, weight, bias)
