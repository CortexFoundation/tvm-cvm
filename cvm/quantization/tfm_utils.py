from mxnet import ndarray as nd
import mxnet as mx

def get_bit(opt):
    if isinstance(opt, nd.NDArray):
        opt = opt.abs().max().asscalar()
    return math.ceil(math.log2(math.fabs(opt)+1)) + 1

def get_bit_cnt(cnt):
    # get_bit_cnt (mrt) should be consistent with 
    # GetReduceSumBit (cvm-runtime)
    assert isinstance(cnt, int) and cnt > 0, \
        "Error in get_bit_cnt, provided cnt: %s"%cnt
    prec = 0
    while cnt != 0:
        prec += 1
        cnt  >>= 1
    return prec

def get_range(prec):
    return (2 ** (prec - 1)) - 1

def scale(threshold, precision):
    assert threshold >= 0
    if threshold == 0:
        return 1
    alpha = (2 ** (precision - 1)) - 1
    return alpha / threshold
