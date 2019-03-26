from __future__ import absolute_import

def verify_dense(data, weight, bias=None):
    if data.dtype == 'float32':
        return True
    if data.dtype == 'int8' and weight.dtype == 'int8':
        if bias is None:
            return (data.shape[1] < (2**16))
        else:
            if bias.dtype == 'int32':
                return (data.shape[1] < ((2**16)-1))
            else:
                return False
    return False
