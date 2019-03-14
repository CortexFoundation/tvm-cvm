import logging
from enum import Enum
from mxnet import symbol
from mxnet import ndarray as nd

class CalibMode(Enum):
    NONE = 0
    NAIVE = 1
    CALIBRATION = 2


class QuantFlag():
    def __init__(self, is_fuse_bn=True, calib_mode=CalibMode.NONE,
            allowed_layers=[], disabled_layers=[],
            log_level=logging.INFO,
            use_asymmetric=True, eliminate_outlier=True):
        self.is_fuse_bn = is_fuse_bn
        assert isinstance(calib_mode, CalibMode)
        self.calib_mode = calib_mode
        self.log_level = log_level
        self.allowed_layers = allowed_layers
        self.disabled_layers = disabled_layers

DEFAULT_TARGET_BITS = 15
BIAS_TARGET_BITS= (DEFAULT_TARGET_BITS+1)*4-1

def quant_helper(data, **kwargs):
    if isinstance(data, nd.NDArray):
        return nd_quant(data, **kwargs)

    assert isinstance(data, symbol.Symbol)
    return symbol_quant(data, **kwargs)

def nd_quant(data, shift_bits=None, offset=None, target_bits=DEFAULT_TARGET_BITS,
        logger=None, msg="", **kwargs):
    assert isinstance(data, nd.NDArray)

    if shift_bits is None:
        shift_bits, offset = calib_quant_params(data, target_bits, **kwargs)

    out = (data / 2 ** shift_bits).floor()

    clip_range = 2 ** target_bits - 1
    if logger and out.abs().max() > clip_range:
        logger.error("quant %s out of range int%d with data=<%s,%s,%s>, sb=%s",
                msg,
                target_bits+1,
                out.asnumpy().flatten()[0],
                # out.asnumpy().flatten()[0:49].max(),
                out.max().asnumpy(),
                out.min().asnumpy(),
                shift_bits.asnumpy())
    elif logger:
        logger.debug("quant %s into int%d with data=<%s,%s,%s>, sb=%s",
                msg,
                target_bits+1,
                out.asnumpy().flatten()[0],
                # out.asnumpy().flatten()[0:49].max(),
                out.max().asnumpy(),
                out.min().asnumpy(),
                shift_bits.asnumpy())

    out = out.clip(a_min=-clip_range, a_max=clip_range)
    return out, shift_bits

def symbol_quant(data, shift_bits, target_bits=DEFAULT_TARGET_BITS,
        logger=None, **kwargs):
    assert isinstance(data, symbol.Symbol)

    F = symbol
    power = F.pow(2, shift_bits)
    out = F.floor(data / power)

    clip_range = 2 ** target_bits - 1
    out = F.clip(out, a_min=-clip_range, a_max=clip_range)

    return out, shift_bits

def calib_quant_params(data, target_bits, use_asymmetric=True,
        eliminate_outlier=True):
    """ Used in calibration pass
    """
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
    shift_bits = bits - target_bits

    return shift_bits, offset
