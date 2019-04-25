import mxnet as mx
from mxnet import ndarray as nd

import numpy as np

def vars_increment(vars_, shape):
    if vars_ is None:
        return [0 for _ in range(len(shape))]
    for i in range(0, len(shape)):
        idx = len(shape) - 1 - i
        if (vars_[idx] + 1 == shape[idx]):
            vars_[idx] = 0
        else:
            vars_[idx] += 1
            break
    return vars_
def input_idx_from_broadcast(ovars, shape):
    ivars = [None] * len(shape)
    for i, var in enumerate(reversed(ovars)):
        if i >= len(shape):
            break
        idx = len(shape) - 1 - i
        if var < shape[idx]:
            ivars[idx] = var
        elif shape[idx] == 1:
            ivars[idx] = 0
        else:
            assert False
    return ivars
def broadcast_shape(shape1, shape2):
    s1_size, s2_size = len(shape1), len(shape2)
    min_size = min(s1_size, s2_size)
    max_size = max(s1_size, s2_size)
    expand_shape = [None] * max_size
    for i in range(1, min_size+1):
        if (shape1[s1_size - i] == shape2[s2_size - i]):
            expand_shape[max_size-i] = shape1[s1_size-i]
        elif (shape1[s1_size - i] == 1):
            expand_shape[max_size-i] = shape2[s2_size-i]
        elif (shape2[s2_size - i] == 1):
            expand_shape[max_size-i] = shape1[s1_size-i]
        else:
            assert False

    shape = shape1 if s1_size > s2_size else shape2
    for i in range(min_size+1, max_size+1):
        expand_shape[max_size-i] = shape[max_size-i]
    return expand_shape

class BroadcastShift(mx.operator.CustomOp):
    def __init__(self, precision, **kwargs):
        super(BroadcastShift, self).__init__(**kwargs)
        clip = 2 ** (int(precision) - 1) -1
        self.min = int(-clip)
        self.max = int(clip)

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X, SB = in_data[0], in_data[1]
        Y = out_data[0]
        bh = self.broadcast_shape(X.shape, SB.shape)
        assert Y.shape == bh
        ovars = None
        for i in range(np.product(bh)):
            ovars = vars_increment(ovars, bh)
            xvars = input_idx_from_broadcast(ovars, X.shape)
            sbvars = input_idx_from_broadcast(ovars, SB.shape)
            x, sb = X[xvars].asscalar(), SB[sbvars].asscalar()
            if sb > 0:
                if sb > 1:
                    x = int(x) >> (sb - 1)
                x += 1
                x = x >> 1
            elif sb < 0:
                x = x << (-sb)
            Y[ovars] = x
        a_min, a_max = self.min, self.max
        Y = Y.clip(a_min=a_min, a_max=a_max)
        self.assign(out_data[0], req[0], Y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False

if __name__ == '__main__':
    shape1, shape2 = (32, 1, 3, 3), (1, 3)
    es = broadcast_shape(shape1, shape2)
    print (es)
    ovars = [0 for _ in range(len(es))]
    ovars = (16, 0, 2, 2)
    ivars1 = input_idx_from_broadcast(ovars, shape1)
    ivars2 = input_idx_from_broadcast(ovars, shape2)
    print (ivars1, ivars2)

    ovars = None
    for i in range(np.product(es)):
        ovars = vars_increment(ovars, es)
        print (ovars,
                input_idx_from_broadcast(ovars, shape1),
                input_idx_from_broadcast(ovars, shape2))

class Clip(mx.operator.CustomOp):
    def __init__(self, precision, **kwargs):
        super(Clip, self).__init__(**kwargs)
        clip = 2 ** (int(precision) - 1) - 1
        self.min = int(-clip)
        self.max = int(clip)

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X = in_data[0]
        a_min, a_max = self.min, self.max
        out = X.clip(a_min=a_min, a_max=a_max)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False

class LeftShift(mx.operator.CustomOp):
    def __init__(self, precision, shift_bit, **kwargs):
        super(LeftShift, self).__init__(**kwargs)
        clip = 2 ** (int(precision) - 1) - 1
        self.min = int(-clip)
        self.max = int(clip)
        self.sb = int(shift_bit)
        assert self.sb > 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X = in_data[0]
        a_min, a_max = self.min, self.max
        out = X * (2 ** (self.sb))
        out = out.clip(a_min=a_min, a_max=a_max)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False

class RightShift(mx.operator.CustomOp):
    def __init__(self, precision, shift_bit, **kwargs):
        super(RightShift, self).__init__(**kwargs)
        clip = 2 ** (int(precision) - 1) - 1
        self.min = int(-clip)
        self.max = int(clip)
        self.sb = int(shift_bit)
        assert self.sb > 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X = in_data[0]
        a_min, a_max = self.min, self.max
        out = X / (2 ** (self.sb-1))
        out = out.floor()
        out = out + 1
        out = out / 2
        out = out.floor()
        out = out.clip(a_min=a_min, a_max=a_max)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False

@mx.operator.register("cvm_clip")
class ClipProp(mx.operator.CustomOpProp):
    def __init__(self, precision=8, shift_bit=0):
        self.precision= precision
        self.shift_bit = shift_bit
        super(ClipProp, self).__init__(need_top_grad=False)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        X_shape = in_shape[0]
        out_shape = in_shape[0]
        return [X_shape], [out_shape], []
    def infer_type(self, in_type):
        X_type = in_type[0]
        return [X_type], [X_type], []
    def create_operator(self, ctx, shapes, dtypes):
        return Clip(self.precision)

@mx.operator.register("cvm_left_shift")
class LeftShiftProp(mx.operator.CustomOpProp):
    def __init__(self, precision=8, shift_bit=0):
        self.precision= precision
        self.shift_bit = shift_bit
        super(LeftShiftProp, self).__init__(need_top_grad=False)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        X_shape = in_shape[0]
        out_shape = in_shape[0]
        return [X_shape], [out_shape], []
    def infer_type(self, in_type):
        X_type = in_type[0]
        return [X_type], [X_type], []
    def create_operator(self, ctx, shapes, dtypes):
        return LeftShift(self.precision, self.shift_bit)

@mx.operator.register("cvm_right_shift")
class RightShiftProp(mx.operator.CustomOpProp):
    def __init__(self, precision=8, shift_bit=0):
        self.precision= precision
        self.shift_bit = shift_bit
        super(RightShiftProp, self).__init__(need_top_grad=False)
    def list_arguments(self):
        return ['data']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        X_shape = in_shape[0]
        out_shape = in_shape[0]
        return [X_shape], [out_shape], []
    def infer_type(self, in_type):
        X_type = in_type[0]
        return [X_type], [X_type], []
    def create_operator(self, ctx, shapes, dtypes):
        return RightShift(self.precision, self.shift_bit)



