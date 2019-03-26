from __future__ import absolute_import, print_function

import tvm
import numpy as np

def run_multi(in_shape1, in_shape2, name1, name2, name3):
    # check shape
    if (in_shape1[1] != in_shape2[0]):
        return
    # make param
    n = tvm.var('n')
    m = tvm.var('m')
    v = tvm.var('v')
    num_n = in_shape1[0]
    num_k = in_shape1[1]
    num_m = in_shape2[0]
    out_shape = (num_n, num_m)

    # graph
    A = tvm.placeholder(in_shape1, dtype='int32', name=name1)
    B = tvm.placeholder(in_shape2, dtype='int32', name=name2)
    k = tvm.reduce_axis((0, num_k), name='k')
    C = tvm.compute(out_shape, lambda i,j : tvm.sum(A[i,k] * B[k,j], axis=k), name=name3)
    #A.dtype = 'int32'
    #B.dtype = 'int32'
    #C.dtype = 'int32'

   # schedule
    s = tvm.create_schedule(C.op)
    print(tvm.lower(s, [A, B, C], simple_mode=True))


if __name__ == '__main__':
    run_multi([255, 255], [255, 255], 'A', 'B', 'C')
