from common import get_all_backend
import numpy as np
import tvm
import topi


def check_shift(lhs_np, rhs_np):
    dtype = 'int32'
    A = tvm.placeholder(shape=(1,), name="A", dtype=dtype)
    B = tvm.placeholder(shape=(1,), name="B", dtype=dtype)
    C = topi.cvm_left_shift(A,B)
    D = topi.cvm_right_shift(A,B)

    device='llvm'
    ctx = tvm.context(device, 0)
    with tvm.target.create(device):
        s = topi.generic.schedule_broadcast(C)
        s2 = topi.generic.schedule_broadcast(D)
    left_shift = tvm.build(s, [A,B,C], device, name="left_shift")
    right_shift = tvm.build(s2, [A,B,D], device, name="right_shift")

    lhs = tvm.nd.array(lhs_np.astype(A.dtype), ctx)
    rhs = tvm.nd.array(rhs_np.astype(B.dtype), ctx)
    #lhs = tvm.nd.array(np.random.randint(low=-128, high=127, size=(1,)).astype(A.dtype), ctx)
    #rhs = tvm.nd.array(np.random.randint(low=0, high=8, size=(1,)).astype(B.dtype), ctx)
    left_out = tvm.nd.array(np.empty(lhs.shape).astype(C.dtype), ctx)
    right_out = tvm.nd.array(np.empty(lhs.shape).astype(C.dtype), ctx)
    left_shift(lhs, rhs, left_out)
    print(lhs)
    print(rhs)
    print("left_shift_out:")
    print(left_out)
    right_shift(lhs, rhs, right_out)
    print("right_shift_out:")
    print(right_out)


for i in range(0,10):
    lhs_np = np.random.randint(low=-128, high=127, size=(1,))
    rhs_np = np.random.randint(low=1, high=8, size=(1,))
    check_shift(lhs_np, rhs_np)
    print("")
