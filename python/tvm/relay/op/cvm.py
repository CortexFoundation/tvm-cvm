"""CVM operations."""
from __future__ import absolute_import as _abs
from . import _make

def cvm_clip(a, precision):
    return _make.cvm_clip(a, precision)

def cvm_left_shift(a, precision, shift_bit):
    return _make.cvm_left_shift(a, precision, shift_bit)

def cvm_right_shift(a, precision, shift_bit):
    return _make.cvm_right_shift(a, precision, shift_bit)

def cvm_lut(data, table, in_dim):
    return _make.cvm_lut(data, table, in_dim)
