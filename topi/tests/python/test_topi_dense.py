"""Test code for dense operator"""
import numpy as np
import tvm
import topi
import topi.testing
from topi.util import get_const_tuple
from tvm.contrib.pickle_memoize import memoize

from common import get_all_backend

def verify_dense(batch, in_dim, out_dim, use_bias=True, itype='int8', otype='int32'):
    A = tvm.placeholder((batch, in_dim), name='A', dtype = itype)
    B = tvm.placeholder((out_dim, in_dim), name='B', dtype = itype)
    C = tvm.placeholder((out_dim,), name='C', dtype = otype)
    dtype = A.dtype

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_dense")
    def get_ref_data():
        a_np = np.random.uniform(size=(batch, in_dim)).astype(itype)
        b_np = np.random.uniform(size=(out_dim, in_dim)).astype(itype)
        c_np = np.random.uniform(size=(out_dim,)).astype(otype)
        if use_bias:
            d_np = np.maximum(np.dot(a_np, b_np.T).astype(otype) + c_np, 0)
        else:
            d_np = np.maximum(np.dot(a_np, b_np.T).astype(otype), 0)
        return (a_np, b_np, c_np, d_np)
    # get the test data
    a_np, b_np, c_np, d_np = get_ref_data()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            D = topi.nn.dense(A, B, C if use_bias else None)
            D = topi.nn.relu(D)
            s = topi.generic.schedule_dense([D])
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(c_np, ctx)
        d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=otype), ctx)
        f = tvm.build(s, [A, B, C, D], device, name="dense")
        f(a, b, c, d)
        tvm.testing.assert_allclose(d.asnumpy(), d_np, rtol=1e-5)

    for device in get_all_backend():
        check_device(device)

def test_dense():
    print("test1")
    verify_dense(1, 1024, 1000, use_bias=True, otype='int64')
    print("test2")
    verify_dense(1, 1024, 1000, use_bias=False)
    print("test3")
    verify_dense(2, 1024, 1000, use_bias=True, otype='int64')
    print("test4 -- float")
    verify_dense(1, 1024, 1000, use_bias=False, itype='float32', otype='float32')

if __name__ == "__main__":
    test_dense()
