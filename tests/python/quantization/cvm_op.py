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
        X, SB, TB = in_data[0], in_data[1], in_data[2]
        Y = out_data[0]
        bh = broadcast_shape(X.shape, SB.shape)
        assert list(Y.shape) == bh, "%s vs. %s in: %s %s" % (Y.shape, bh, X.shape, SB.shape)
        ovars = None
        for i in range(np.product(bh)):
            ovars = vars_increment(ovars, bh)
            xvars = tuple(input_idx_from_broadcast(ovars, X.shape))
            sbvars = tuple(input_idx_from_broadcast(ovars, SB.shape))
            x, sb = X[xvars].asscalar(), SB[sbvars].asscalar()
            x, sb = int(round(x)), int(round(sb))
            tb = int(round(TB[sbvars].asscalar()))
            if sb > 0:
                if sb > 1:
                    x = x >> (sb - 1)
                x += 1
                x = x >> 1
            elif sb < 0:
                x = x << (-sb)
            clip = 2 ** (tb - 1) - 1
            Y[ovars] = max(min(x, clip), -clip)
        #  a_min, a_max = self.min, self.max
        #  Y = Y.clip(a_min=a_min, a_max=a_max)
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

cvm_log = open('/tmp/inception_v3/out/cvm_op1.txt', "w+")
class Clip(mx.operator.CustomOp):
    def __init__(self, precision, cvm_name, **kwargs):
        super(Clip, self).__init__(**kwargs)
        clip = 2 ** (int(precision) - 1) - 1
        self.min = int(-clip)
        self.max = int(clip)
        self.cvm_name = cvm_name

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X = in_data[0]
        a_min, a_max = self.min, self.max
        out = X.round()
        out = out.clip(a_min=a_min, a_max=a_max)
        # X_max, X_min = X.max().asscalar(), X.min().asscalar()
        # omax, omin = out.max().asscalar(), out.min().asscalar()
        # cvm_log.write("%s:\n %d %d %d %d %s \n" % (self.cvm_name,
        #             round(X_max), round(X_min),
        #             round(omax), round(omin),
        #             " ".join(str(int(round(d))) for d in X.asnumpy().flatten()[:10])))
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
        out = X.round()
        out = out * (2 ** (self.sb))
        out = out.clip(a_min=a_min, a_max=a_max)
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False

class RightShift(mx.operator.CustomOp):
    def __init__(self, precision, shift_bit, cvm_name, **kwargs):
        super(RightShift, self).__init__(**kwargs)
        clip = 2 ** (int(precision) - 1) - 1
        self.min = int(-clip)
        self.max = int(clip)
        self.sb = int(shift_bit)
        assert self.sb > 0
        self.cvm_name = cvm_name

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X = in_data[0]
        a_min, a_max = self.min, self.max
        out = X.round()
        if self.sb > 1:
            out = out / (2 ** (self.sb-1))
            out = out.floor()
        out = out + 1
        out = out / 2
        out = out.floor()
        out = out.clip(a_min=a_min, a_max=a_max)
        # X_max, X_min = X.max().asscalar(), X.min().asscalar()
        # omax, omin = out.max().asscalar(), out.min().asscalar()
        # cvm_log.write("%s:\n %d %d %d %d %s \n" % (self.cvm_name,
        #             round(X_max), round(X_min),
        #             round(omax), round(omin),
        #             " ".join(str(int(round(d))) for d in X.asnumpy().flatten()[:10])))
        self.assign(out_data[0], req[0], out)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False

class LUT(mx.operator.CustomOp):
    def __init__(self, precision=8, **kwargs):
        super(LUT, self).__init__(**kwargs)

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        X, T = in_data[0], in_data[1]
        Y = out_data[0]
        tsize = T.shape[0]
        xvars = vars_increment(None, X.shape)
        while xvars != list(X.shape):
            idx = int(round(X[tuple(xvars)].asscalar()))
            Y[xvars] = T[idx]
            xvars = vars_increment(xvars, X.shape)
        self.assign(out_data[0], req[0], Y)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        assert False

class Annotate(mx.operator.CustomOp):
    def __init__(self, in_prec, out_prec):
        super(Annotate, self).__init__()
        self.in_prec = int(in_prec)
        self.out_prec = int(out_prec)

    def forward(self, is_train, req, in_data, out_data, aux):
        assert is_train == False
        self.assign(out_data[0], req[0], in_data[0])

@mx.operator.register("cvm_clip")
class ClipProp(mx.operator.CustomOpProp):
    def __init__(self, precision=8, shift_bit=0, cvm_name='clip'):
        self.precision= precision
        self.shift_bit = shift_bit
        self.cvm_name = cvm_name
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
        return Clip(self.precision, self.cvm_name)

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
    def __init__(self, precision=8, shift_bit=0, cvm_name='right_shift'):
        self.precision= precision
        self.shift_bit = shift_bit
        self.cvm_name = cvm_name
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
        return RightShift(self.precision, self.shift_bit, self.cvm_name)

@mx.operator.register("cvm_broadcast_shift")
class BroadcastShiftProp(mx.operator.CustomOpProp):
    def __init__(self, precision=8):
        self.precision= precision
        super(BroadcastShiftProp, self).__init__(need_top_grad=False)
    def list_arguments(self):
        return ['data', 'shift_bits', 'target_bits']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        X_shape = in_shape[0]
        B_shape = in_shape[1]
        P_shape = in_shape[1]
        out_shape = broadcast_shape(X_shape, B_shape)
        return [X_shape, B_shape, P_shape], [out_shape], []
    def infer_type(self, in_type):
        X_type = in_type[0]
        B_type = P_type = X_type
        return [X_type, B_type, P_type], [X_type], []
    def create_operator(self, ctx, shapes, dtypes):
        return BroadcastShift(self.precision)

@mx.operator.register("cvm_lut")
class LUTProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(LUTProp, self).__init__(need_top_grad=False)
    def list_arguments(self):
        return ['data', 'table']
    def list_outputs(self):
        return ['output']
    def infer_shape(self, in_shape):
        X_shape = in_shape[0]
        B_shape = in_shape[1]
        out_shape = in_shape[0]
        return [X_shape, B_shape], [out_shape], []
    def infer_type(self, in_type):
        X_type = in_type[0]
        B_type = X_type
        return [X_type, B_type], [X_type], []
    def create_operator(self, ctx, shapes, dtypes):
        return LUT()

@mx.operator.register("cvm_annotate")
class AnnotateProp(mx.operator.CustomOpProp):
    def __init__(self, in_prec=8, out_prec=8):
        self.in_prec = in_prec
        self.out_prec = out_prec
        super(AnnotateProp, self).__init__(need_top_grad=False)
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
        return Annotate(self.in_prec, self.out_prec)









