import logging
import math

import mxnet as mx
from mxnet import ndarray as nd

def get_zero_symmetric(threshold):
    min_range, max_range = threshold

    if max_range == min_range:
        return 0
    else:
        return (max_range + min_range) / 2

def get_threshold_requant_parameter(threshold, sbit=None, target_bit=8,
        is_symmetric=True, is_sign=True):
    min_range, max_range = threshold
    real_bit = target_bit - (1 if is_sign else 0)

    offset = None
    if is_symmetric:
        alpha = max(abs(min_range), abs(max_range))
    else:
        if max_range == min_range:
            alpha = abs(max_range)
        else:
            alpha = (max_range - min_range) / 2
        offset = (alpha - max_range)

    if alpha != 0 and sbit is None:
        alpha_bit = math.ceil(math.log2(alpha))
        sbit = alpha_bit - real_bit

    if offset:
        offset = int(offset / (2**sbit) + 0.5)

    return (sbit, offset)

def nd_quant(data, shift_bits=None, target_bit=8,
        logger=logging):
    real_bits = target_bit - 1 # real bits is decreased for int type
    if shift_bits is None:
        assert isinstance(data, nd.NDArray)
        shift_bits, _ = _nd_quant_params(data, real_bits)

    out = (data / (2 ** (shift_bits))).round()

    clip_range = 2 ** real_bits - 1
    if logger and out.abs().max() > clip_range:
        logger.warn("quant out of range int%d with data=<%s,%s,%s>, sb=%s",
                target_bit,
                out.asnumpy().flatten()[0],
                out.max().asnumpy(),
                out.min().asnumpy(),
                shift_bits.asnumpy())
    elif logger:
        logger.debug("quant into int%d with data=<%s,%s,%s>, sb=%s",
                target_bit,
                out.asnumpy().flatten()[0],
                out.max().asnumpy(),
                out.min().asnumpy(),
                shift_bits.asnumpy())

    out = out.clip(a_min=-clip_range, a_max=clip_range)
    return out, shift_bits


def sym_quant(sym, params, graph, shift_bits, target_bit=8):
    scale_name = sym.attr('name') + '_requant_scale'
    assert scale_name not in graph
    scale_sym = mx.sym.var(scale_name, shape=(1,))
    graph[scale_name] = scale_sym

    n1, n2 = "const_var_1", 'const_var_2'
    sym_1 = graph[n1] if n1 in graph else mx.sym.var(n1, shape=(1,))
    sym_2 = graph[n2] if n2 in graph else mx.sym.var(n2, shape=(1,))
    graph[n1], graph[n2] = sym_1, sym_2

    assert shift_bits.shape == (1,)
    if shift_bits < 0:
        scale = 2 ** (-shift_bits)
        out = mx.sym.broadcast_mul(sym, scale_sym)
    elif shift_bits == 0:
        out, scale = sym, nd.zeros((1,))
    else:
        scale = 2 ** (shift_bits - 1)
        out = mx.sym.broadcast_div(sym, scale_sym)
        out = mx.sym.floor(out)
        out = mx.sym.broadcast_add(out, sym_1)
        out = mx.sym.broadcast_div(out, sym_2)
        out = mx.sym.floor(out)

    params[n1] = nd.array([1])
    params[n2] = nd.array([2])
    params[scale_name] = scale

    clip_range = 2 ** (target_bit - 1) -1
    out = mx.sym.clip(out, a_min=-clip_range, a_max=clip_range)
    return out, params


def _nd_quant_params(data, real_bits, use_asymmetric=True,
        eliminate_outlier=False):
    if eliminate_outlier:
        mean = data.mean()
        var = ((data - mean) * (data - mean)).mean()
        std = var.sqrt()
        norm_data = ((data - mean) / std).clip(a_min=-4, a_max=4) * std + data.mean()
    else:
        norm_data = data

    if use_asymmetric:
        alpha = norm_data.abs().max()
        offset = nd.zeros((1,))
    else:
        min_v = norm_data.min()
        max_v = norm_data.max()
        alpha = (max_v - min_v) / 2
        offset = (alpha - max_v).floor()

    assert any(alpha != 0)

    bits = alpha.log2().ceil()
    shift_bits = bits - real_bits

    return shift_bits, offset
