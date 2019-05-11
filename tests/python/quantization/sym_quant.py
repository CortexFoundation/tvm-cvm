import mxnet as mx
from mxnet import ndarray as nd

max_bit = 32 # INT32
default_target_bit = 8 # INT8
bias_target_bit = default_target_bit * 4 - 1
disable_requant_ops = [
    'Activation', 'relu',
    'Pooling',
    'slice', 'slice_like', 'slice_axis',
    'clip', 'negative',
    'repeat', 'tile',
    'Reshape', 'transpose', 'Flatten',
]

def sym_simulate(symbol, params, inputs_ext, data):
    return symbol, params
