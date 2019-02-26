from __future__ import absolute_import, print_function
import tvm
import topi
import numpy as np

if __name__ == '__main__':
    x, y = 100, 10
    a = tvm.placeholder((x, y, y), name='a')
    b = tvm.placeholder((y, y), name='b')
    c = a+b
    d = a*b

    e = topi.elemwise_sum([c, d])
    f = e/2.0
    g = topi.sum(f)
    with tvm.target.cuda():
        sg = topi.generic.schedule_reduce(g)
        print(tvm.lower(sg, [a,b], simple_mode=True))
