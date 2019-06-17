from mxnet import nd
import mxnet as mx
import numpy as np
import topi.testing
import tvm
import os
import math

import ops_generator as opg
from ops_generator import std_int_constraint, iter_constraint, \
    list_constraint, gen_non_constraint, range_constraint, \
    rand_constraint, shape_constraint
from ops_generator import IntIter, NoneIter, ConstantIter, ConcatIter, \
    VectorIter, PermutationIter, ShapeIter, AllOverIter, BoolIter, \
    RepeatIter, RandomBoolIter, RandomVectorIter, RandomIter
import utils

INT32 = "int32"

# ====== transform ======

def verify_expand_dims():
    pass

def verify_transpose():
    dshp = (1, 2, 3)
    data = opg.ConstantIter(iter_constraint(6), shape=dshp)
    axes = opg.PermutationIter(list_constraint([0, 1, 2]),
                               list_constraint([-1, 0, 1]),
                               list_constraint([-3, -2, -1]),
                               list_constraint([-3, 1, 2]))
    axes = opg.ConcatIter(axes,
                opg.NoneIter(),
                opg.RepeatIter(IntIter(std_int_constraint(2)), 3),
                opg.ConstantIter(list_constraint([-4, 0, 1])),
                opg.ConstantIter(list_constraint([-1, 0, 4])),
                opg.ConstantIter(list_constraint([0, 1])),
                opg.ConstantIter(list_constraint([1])),
                name="axes")
    op_units = opg.OpUnitIter([data, axes], attr_index=1)
    def transpose(data, axes):
        data_npy = np.array(data, dtype=INT32)
        return [np.transpose(data_npy, axes)]
    op_units.eval_data("transpose", transpose, is_dump=True)

def verify_reshape():
    pass

def verify_squeeze():
    pass

def verify_concatenate():
    data = ConcatIter(
            ConstantIter(iter_constraint(6), shape=(1,2,3)),
            ConstantIter(iter_constraint(3)))
    axis = IntIter(range_constraint(-4, 4), name="axis")
    def concatenate(*inputs):
        data_npys = [np.array(d) for d in inputs[:-1]]
        out_npy = np.concatenate(data_npys, axis=inputs[-1])
        return [out_npy]
    def cstr_func(*inputs):
        axis = inputs[-1]
        shp = np.array(inputs[0]).shape
        ndim = len(shp)
        if (-ndim-1 <= axis) and (axis <= ndim):
            return True
        return False

    is_dump = True
    op_units = opg.OpUnitIter([data, axis], 1, [cstr_func])
    op_units.eval_data("concatenate", concatenate, is_dump)

    op_units = opg.OpUnitIter([data, data, axis], 2, [cstr_func])
    op_units.eval_data("concatenate", concatenate, is_dump)

    op_units = opg.OpUnitIter([data, data, data, axis], 3, [cstr_func])
    op_units.eval_data("concatenate", concatenate, is_dump)

def verify_take():
    dshp = (1, 2, 3)
    data = opg.ConstantIter(opg.iter_constraint(6), shape=dshp)
    iattr = opg.RandomVectorIter(-1, 7, 5)
    iattr2 = opg.VectorIter(iattr, 2)
    indices = opg.ConcatIter(iattr, iattr2)
    axis = opg.IntIter(opg.range_constraint(-4, 4), opg.gen_non_constraint(),
            name="axis")
    print (len(data), len(indices), len(axis))
    def take_func(data, indices, axis):
        if axis == None:
            return True
        dim = dshp[axis % 3]
        np_idx = np.array(indices).flatten()
        if (np_idx > dim).any():
            return True
        return False
    op_units = opg.OpUnitIter([data, indices, axis], 2, [take_func])
    def take(data, indices, axis):
        data_npy = np.array(data, dtype=INT32)
        if axis is None:
            return [np.take(data_npy, indices, mode="clip")]
        else:
            return [np.take(data_npy, indices, axis=axis, mode="clip")]
    op_units.eval_data("take", take, is_dump=True)

def verify_strided_slice():
    dshp = (2, 2)
    data = opg.ConstantIter(opg.iter_constraint(4), shape=dshp)
    attr = IntIter(range_constraint(-3, 3))
    begin = ConcatIter(
            VectorIter(attr, 2),
            VectorIter(attr, 1),
            VectorIter(attr, 0),
            name="begin")
    end = ConcatIter(
            VectorIter(attr, 2),
            VectorIter(attr, 1),
            VectorIter(attr, 0),
            name="end")
    sattr = IntIter(list_constraint([-2, -1, 1, 2]))
    strides = ConcatIter(
            VectorIter(sattr, 2),
            VectorIter(sattr, 1),
            [[0], [1, 0], [0, 1]],
            NoneIter(),
            name="stride")
    def cstr_func(*inputs):
        data, begin, end, strides = inputs
        if strides is not None and strides[0] != 1:
            return False
        satisfied = True
        for i in range(len(data)):
            dim = len(data[i])
            b = begin[i]%dim if (begin is not None) and (i < len(begin)) else 0
            e = end[i]%dim if (end is not None) and (i < len(end)) else dim
            s = strides[i] if (strides is not None) and (i < len(strides)) else 1
            if (s < 0) and (b <= e):
                satisfied = False
                break
            elif (s > 0) and (e <= b):
                satisfied = False
                break
        return satisfied
    def strided_slice(data, begin, end, strides):
        dshp = len(data)
        for i in range(len(begin), dshp):
            begin.append(0)
        for i in range(len(end), dshp):
            end.append(len(data[i]))
        if strides is None:
            strides = []
        for i in range(len(strides), dshp):
            strides.append(1)
        for i in range(dshp):
            dim = len(data[i])
            b, e, s = begin[i], end[i], strides[i]
            begin_range = -1 if s < 0 else 0
            end_range = dim-1 if s < 0 else dim
            b = b+dim if b < 0 else b
            e = e+dim if e < 0 else e
            b = min(max(b, begin_range), end_range)
            e = min(max(e, begin_range), end_range)
            if ((s < 0 and (b <= e)) or \
                (s > 0 and (e <= b))):
                raise ValueError("begin=%d;%d, end=%d;%d, stride=%d" \
                        % (begin[i], b, end[i], e, s))

        data_npy = np.array(data)
        out_npy = topi.testing.strided_slice_python(data_npy, begin, end, strides)
        return [out_npy]

    op_units = opg.OpUnitIter([data, begin, end, strides], 1, [cstr_func])
    op_units.eval_data("strided_slice", strided_slice, is_dump=True)

    dshp = (10, 10)
    data = ConstantIter(iter_constraint(100), shape=dshp)
    begin = ConcatIter([[0, 9]], name="begin")
    end = ConcatIter([[9, -20]], name="end")
    strides = ConcatIter(
                [[1, -1], [2, -2], [3, -3], [4, -4], [5, -5], [6, -6],
                 [9, -9], [10, -10], [20, -20]],
                name="stride")
    op_units = opg.OpUnitIter([data, begin, end, strides], 1)
    op_units.eval_data("strided_slice", strided_slice, is_dump=True)

def verify_repeat():
    dshp = (1, 2, 3, 4)
    data = opg.ConstantIter(opg.iter_constraint(24), shape=dshp)
    repeats = opg.IntIter(iter_constraint(3), name="repeats")
    axis = opg.IntIter(range_constraint(-5, 5), name="axis")
    def repeat(data, repeats, axis):
        if repeats < 1:
            raise ValueError("repeats invalid: %s"%repeats)
        data_npy = np.array(data)
        out_npy = np.repeat(data_npy, repeats, axis)
        return [out_npy]

    op_units = opg.OpUnitIter([data, repeats, axis], 1)
    op_units.eval_data("repeat", repeat, is_dump=True)

def verify_tile():
    dshp = (1, 2, 3)
    data = opg.ConstantIter(opg.iter_constraint(6), shape=dshp)
    iattr = IntIter(range_constraint(1, 3))
    reps = ConcatIter(
            opg.AllOverIter(iattr, shape_constraint(4)),
            opg.VectorIter(iattr, 0),
            [[2, 1, 0], [-1]],
            name="reps")
    def tile(data, reps):
        invalid = (len(reps) == 0)
        for rep in reps:
            if rep <= 0:
                invalid = True
                break
        if invalid:
            raise ValueError("reps invalid %s" % reps)
        data_npy = np.array(data)
        out_npy = np.tile(data_npy, reps)
        return [out_npy]

    op_units = opg.OpUnitIter([data, reps], 1)
    op_units.eval_data("tile", tile, is_dump=True)

def verify_slice_like():
    dshp = (1, 2, 3)
    data = opg.ConstantIter(opg.iter_constraint(6), shape=dshp)
    sshp = (1, 2, 2, 2)
    sdata = ConstantIter(iter_constraint(8), shape=sshp)
    iattr = IntIter(range_constraint(-4, 4))
    axis = ConcatIter(
            AllOverIter(iattr, shape_constraint(4)),
            [[-5], [0, 1, 4]],
            name="axis")
    def slice_like(data, shape, axis):
        data_nd = nd.array(data)
        shape_nd = nd.array(shape)
        out = nd.slice_like(data_nd, shape_nd, axis)
        return [out]

    op_units = opg.OpUnitIter([data, sdata, axis], 2)
    op_units.eval_data("slice_like", slice_like, is_dump=True)

    sdata = ConcatIter(
            ConstantIter(iter_constraint(1), shape=(1,1,1)),
            ConstantIter(iter_constraint(2), shape=(1,1,2)),
            ConstantIter(iter_constraint(2), shape=(1,2,1)),
            ConstantIter(iter_constraint(2), shape=(2,1,1)),
            ConstantIter(iter_constraint(2), shape=(1,1,1,2)),
            ConstantIter(iter_constraint(2), shape=(1,2)),
            [[]],
            )
    axis = VectorIter(iattr, 0, name="axis")
    op_units = opg.OpUnitIter([data, sdata, axis], 2)
    op_units.eval_data("slice_like", slice_like, is_dump=True)

# ====== reduce ======

def verify_reduce(op_name):
    dshp = (1, 2, 3)
    data = opg.ConstantIter(opg.iter_constraint(6), shape=dshp)

    iattr = IntIter(range_constraint(-3, 3))
    axis = ConcatIter(
            AllOverIter(iattr, shape_constraint(3)),
            [[0, 1, 2, 2], [1, 2, 3], [-1, -4, 1], []],
            name="axis")
    keepdims = opg.BoolIter(name="keepdims")
    exclude = opg.BoolIter(name="exclude")
    def _reduce(data, axis, keepdims, exclude):
        data_nd = nd.array(data)
        out_nd = getattr(nd, op_name)(data_nd, axis, keepdims, exclude)
        return [out_nd]

    op_units = opg.OpUnitIter([data, axis, keepdims, exclude], 1)
    op_units.eval_data(op_name, _reduce, is_dump=True)

def verify_max():
    verify_reduce("max")

def verify_sum():
    verify_reduce("sum")

# ====== nn ======
def verify_conv2d():
    batch = IntIter(list_constraint([1, 4, 8, 16]))
    channel = IntIter(list_constraint([1, 3, 4]))
    h = IntIter(range_constraint(1, 9, 3))
    w = IntIter(range_constraint(1, 9, 3))
    dshp = opg.ExtendIter(batch, channel, h, w)
    datas = []
    for i in range(len(dshp)):
        size = np.product(dshp[i])
        arr1 = ConstantIter(rand_constraint(-127, 127, size), shape=dshp[i])
        arr2 = ConstantIter(rand_constraint(0, 127, size), shape=dshp[i])
        datas.extend([arr1, arr2])
    data = ConcatIter(*datas)
    print (len(data))

    num_filter = IntIter(list_constraint([1, 32, 64]), name="num_filter")
    kernel = VectorIter(
            IntIter(list_constraint([1, 2, 3])),
            size=2,
            name="kernel_size")
    strides = RandomVectorIter(1, 4, 2, 1, name="strides")
    padding = RandomVectorIter(0, 3, 2, 1, name="padding")
    dilation = RepeatIter(
            IntIter(list_constraint([1, 2])),
            size=2,
            name="dilation")
    groups = IntIter(list_constraint([1, 2]), name="groups")
    use_bias = RandomBoolIter(name="use_bias")
    op_units = opg.OpUnitIter(
            [data, num_filter, kernel, strides,
            padding, dilation, groups, use_bias], 1)
    for i in range(len(op_units)):
        data, num_filter, kernel, strides, padding, \
            dilation, groups, use_bias = op_units[i]
        a_np = np.array(data)
        batch, ic, h, w = a_np.shape
        groups = 1 if groups == 1 else ic
        wshp = (num_filter, ic // groups, *kernel)
        wsize = np.product(wshp)
        weight = ConstantIter(rand_constraint(-127, 127, wsize), shape=wshp)
        w_np = np.array(weight[0])
        bshp = (num_filter,)
        bias = ConstantIter(rand_constraint(-127, 127, num_filter), shape=bshp)
        b_np = np.array(bias[0])
        attr = {
            'channels': num_filter,
            'kernel_size': kernel,
            'strides': strides,
            'padding': padding,
            'dilation': dilation,
            'groups': groups,
            'layout': "NCHW",
            'kernel_layout': "OIHW",
            'use_bias': use_bias,
        }
        ins = [a_np, w_np, b_np] if use_bias else [a_np, w_np]
        outs, err = None, None
        try:
            b_nd = nd.array(b_np) if use_bias else None
            _ = nd.Convolution(nd.array(a_np), nd.array(w_np), b_nd, kernel,
                    strides, dilation, padding, num_filter, groups,
                    no_bias=(not use_bias))

            dw_np = topi.testing.dilate_python(w_np, (1, 1, *dilation))
            c_np = topi.testing.conv2d_nchw_python(a_np, dw_np, strides, padding)
            if use_bias:
                c_np += b_np.reshape(num_filter, 1, 1)
            outs = [c_np]
        except Exception as e:
            err = "Error:\n" + str(e)
        print (a_np.shape, wshp, bshp, attr,
                outs[0].shape if outs else None,
                err.replace("\n", "") if err else None)
        opg.dump("conv2d", attr, ins, outs, err)

    # attr = {'num_filter': 64, 'kernel': [3, 3], 'stride': [2, 1], 'pad': [1, 3], 'dilate': [2, 2], 'num_group': 3, 'no_bias': True}
    # dshp = (8, 3, 4, 4)
    # data = mx.sym.var("data", shape=dshp)
    # weight = mx.sym.var("weight")
    # bias = mx.sym.var("bias") 
    # out = mx.sym.Convolution(data, weight, None, **attr)
    # print (out.infer_shape())

def verify_dense():
    batch = IntIter(list_constraint([1, 4, 8, 16]))
    channel = IntIter(list_constraint([1, 3, 32, 64]))
    dshp = opg.ExtendIter(batch, channel)
    datas = []
    for i in range(len(dshp)):
        size = np.product(dshp[i])
        arr1 = ConstantIter(rand_constraint(-127, 127, size), shape=dshp[i])
        arr2 = ConstantIter(rand_constraint(0, 127, size), shape=dshp[i])
        arr3 = ConstantIter(rand_constraint(1, 127, size), shape=dshp[i])
        datas.extend([arr1, arr2, arr3])
    data = ConcatIter(*datas,
            ConstantIter(rand_constraint(-127, 127, 112), shape=(4, 4, 7)))
    print (len(data))

    units = IntIter(list_constraint([1, 32, 64]), name="units")
    use_bias = BoolIter(name="use_bias")
    op_units = opg.OpUnitIter([data, units, use_bias], 1)
    for i in range(len(op_units)):
        data, units, use_bias = op_units[i]
        data_nd = nd.array(data)
        batch, ic = data_nd.shape[:2]
        wshp = (units, ic)
        wsize = np.product(wshp)
        weight = ConstantIter(rand_constraint(-127, 127, wsize), shape=wshp)
        weight_nd = nd.array(weight[0])
        bshp = (units,)
        bias = ConstantIter(rand_constraint(-127, 127, units), shape=bshp)
        bias_nd = nd.array(bias[0])
        attr = {
            'units': units,
            'use_bias': use_bias,
        }
        ins = [data_nd, weight_nd, bias_nd] if use_bias else [data_nd, weight_nd]
        outs, err = None, None
        try:
            if use_bias:
                out = nd.FullyConnected(data_nd, weight_nd, bias_nd,
                        units, (not use_bias))
            else:
                out = nd.FullyConnected(data_nd, weight_nd, None,
                        units, (not use_bias))
            outs = [out]
        except Exception as e:
            err = "Error:\n" + str(e)
        print (data_nd.shape, wshp, bshp, attr,
                outs[0].shape if outs else None,
                err.replace("\n", "") if err else None)
        opg.dump("dense", attr, ins, outs, err)

def verify_max_pool2d():
    batch = IntIter(list_constraint([1, 4]))
    channel = IntIter(list_constraint([1, 4]))
    h = IntIter(range_constraint(1, 9, 3))
    w = IntIter(range_constraint(1, 9, 3))
    dshp = opg.ExtendIter(batch, channel, h, w)
    datas = []
    for i in range(len(dshp)):
        size = np.product(dshp[i])
        arr1 = ConstantIter(rand_constraint(-127, 127, size), shape=dshp[i])
        arr2 = ConstantIter(rand_constraint(0, 127, size), shape=dshp[i])
        datas.extend([arr1, arr2, arr3, arr4])
    data = ConcatIter(*datas)
    print (len(data))

    iattr = IntIter(range_constraint(1, 4))
    pool_size = ConcatIter(
            VectorIter(iattr, 2),
            name="pool_size")
    strides = RandomVectorIter(1, 5, 2, 4, name="strides")
    iattr = IntIter(iter_constraint(2))
    padding = ConcatIter(
            VectorIter(iattr, 1),
            VectorIter(iattr, 2),
            name="padding")
    ceil_mode = BoolIter(name="ceil_mode")
    def max_pool2d(data, pool_size, strides, padding, ceil_mode):
        if ceil_mode:
            raise ValueError("ceil_mode must be false")
        data_nd = nd.array(data)
        pad = padding
        if len(padding) == 1:
            pad = (padding[0], padding[0])
        out = nd.Pooling(data_nd, pool_size, pool_type="max",
                global_pool=False, stride=strides, pad=pad)

        data_npy = np.array(data)
        ashape = data_npy.shape
        n, ic, ih, iw = data_npy.shape
        sh, sw = strides
        kh, kw = pool_size
        bshape = [n, ic, ih, iw]
        if len(padding) == 1:
            pt = pl = pb = pr = padding[0]
        else:
            pt = pb = padding[0]
            pl = pr = padding[1]
        pad_np = np.zeros(shape=(n, ic, ih+pt+pb, iw+pl+pr)).astype(INT32)
        no_zero = (range(n), range(ic), (range(pt, ih+pt)), (range(pl, iw+pl)))
        pad_np[np.ix_(*no_zero)] = data_npy
        if ceil_mode:
            bshape[2] = int(math.ceil(float(ashape[2] - kh + pt + pb) / sh) + 1)
            bshape[3] = int(math.ceil(float(ashape[3] - kw + pl + pr) / sw) + 1)
        else:
            bshape[2] = int(math.floor(float(ashape[2] - kh + pt + pb) / sh) + 1)
            bshape[3] = int(math.floor(float(ashape[3] - kw + pl + pr) / sw) + 1)
        _, oc, oh, ow = bshape
        b_np = np.zeros(shape=(n, oc, oh, ow)).astype(INT32)
        for i in range(oh):
            for j in range(ow):
                b_np[:,:,i,j] = np.max(pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2,3))
        return [b_np]

    op_units = opg.OpUnitIter(
            [data, pool_size, strides, padding, ceil_mode], 1)
    op_units.eval_data("max_pool2d", max_pool2d, is_dump=True)

def verify_upsampling():
    batch = IntIter(list_constraint([1, 4, 8, 16]))
    channel = IntIter(list_constraint([1, 3, 4]))
    h = IntIter(range_constraint(1, 9, 3))
    w = IntIter(range_constraint(1, 9, 3))
    dshp = opg.ExtendIter(batch, channel, h, w)
    datas = []
    for i in range(len(dshp)):
        size = np.product(dshp[i])
        arr1 = ConstantIter(rand_constraint(-127, 127, size), shape=dshp[i])
        arr2 = ConstantIter(rand_constraint(0, 127, size), shape=dshp[i])
        datas.extend([arr1, arr2])
    data = ConcatIter(*datas)

    scale = IntIter(iter_constraint(3), name="scale")
    def upsampling(data, scale):
        if scale == 0:
            raise ValueError("scale must > 0 vs. " + str(scale))
        a_np = np.array(data)
        b_np = topi.testing.upsampling_python(a_np, scale, "NCHW")
        return [b_np]

    op_units = opg.OpUnitIter([data, scale], 1)
    op_units.eval_data("upsampling", upsampling, is_dump=True)

# ====== broadcast ======
# def verify_broadcast(op_name="broadcast_add"):
#     iattr = IntIter(shape_constraint(3))
#     dattr = RandomIter(-127, 127)
#     datas = []
#     for i in range(1, 5):
#         dshp = VectorIter(iattr, i)
#         for j in range(len(dshp)):
#             data = ShapeIter(dattr, shape=dshp[j])
#             datas.append(data[0])
#             print (dshp[j], data[0])
# 
#     for shp in [(4096, 256), (50, 7, 37, 42)]:
#         data = ShapeIter(dattr, shape=shp)
#         datas.append(data[0])
#         print (np.array(data[0]).shape)
# 
#     data = ConcatIter(*datas)
#     def broadcast(data):
#         data_npy = np.array(data)
#         dshp = data_npy.shape

# ====== elemwise ======







# ====== vision ======
def verify_get_valid_counts():
    dshp = ConcatIter(
            PermutationIter(list_constraint([4, 5, 6])),
            [[1, 2, 3], [3, 1, 2], [2, 3, 1]])
    data_arr = []
    for i in range(len(dshp)):
        shp = dshp[i]
        size = np.product(shp)
        arr = ConstantIter(rand_constraint(0, 10, size), shape=shp)
        data_arr.append(arr)
    data = ConcatIter(*data_arr)
    score = IntIter(list_constraint([-1, 0, 1, 5, 7]),
                name="score_threshold")
    def get_valid_counts(data, score_threshold):
        np_data = np.array(data)
        dshp = np_data.shape
        if len(dshp) != 3 or (dshp[2] <= 2):
            raise ValueError("data shape error: " + str(dshp))
        batch_size, num_anchor, elem_length = dshp
        np_out1 = np.zeros(shape=(batch_size,))
        np_out2 = np.zeros(shape=dshp, dtype=INT32)
        for i in range(batch_size):
            np_out1[i] = 0
            inter_idx = 0
            for j in range(num_anchor):
                score = np_data[i, j, 1]
                if score > score_threshold:
                    for k in range(elem_length):
                        np_out2[i, inter_idx, k] = np_data[i, j, k]
                    np_out1[i] += 1
                    inter_idx += 1
                if j >= np_out1[i]:
                    for k in range(elem_length):
                        np_out2[i, j, k] = -1.0
        return [np_out1, np_out2]

    op_units = opg.OpUnitIter([data, score], 1)
    op_units.eval_data("get_valid_counts", get_valid_counts, is_dump=True)

# def verify_non_max_suppression():
#     return
#     attr = {
#         'iou_threshold': 0,
#         'force_suppress': False,
#         'top_k': -1,
#         'id_index': 0,
#         'score_index': 1,
#         'coord_start': 2,
#         'max_output_size': -1,
#         'return_indices': False,
#         'invalid_to_bottom': True,
#     }
#     elem = RandomVectorIter(-10, 10, 6)
#     dshp = (10, 20, 6)
#     data = ShapeIter(rand_constraint(-10, 10, 1200), shape=dshp)
# 
#     def non_max_suppression(data, valid_count):
#         ctx = tvm.context("llvm")
#         device = "llvm"
#         tvm_data = tvm.nd.array(data, ctx=ctx)
#         tvm_valid_count = tvm.nd.array(valid_count, ctx=ctx)
#         with tvm.target.create(device):
#             out = topi.vision.non_max_suppression(data, valid_count,
#                     -1, nms_threshold, force_suppress, nms_topk, return_indices=False)
#             indices_out = non_max_suppression(data, valid_count, -1, nms_threshold, force_suppress, nms_topk)

# ====== test ======
def test_load(op_name, hsh, datadir="/data/ops_generator"):
    hsh_dir = "%s/%s/%s" % (datadir, op_name, hsh)
    files = os.listdir(hsh_dir)

    ins, outs = [], []
    attr, err = None, None
    for fname in files:
        fpath = hsh_dir + "/" + fname
        if fname == 'attr.txt':
            attr_str = open(fpath, "r").read()
            attr = eval(attr_str)
            print (fname, attr)
        elif fname.startswith('in_'):
            npy_str = open(fpath, "r").read()
            npy = opg.txt_npy(npy_str)
            ins.append(npy)
            print(fname, npy.shape)
        elif fname.startswith('out_'):
            npy_str = open(fpath, "r").read()
            npy = opg.txt_npy(npy_str)
            outs.append(npy)
            print(fname, npy.shape)
        else:
            assert fname == 'err.txt'
            err = open(fpath, "r").read()
            print (fname, err)

    a_np = ins[0]
    w_np = ins[1]
    dilation = (1, 1, *eval(attr['dilation']))
    stride = eval(attr['strides'])
    padding = eval(attr['padding'])
    print (a_np.shape, w_np.shape, dilation, stride, padding)
    #  attr = {
        #  'no_bias': (not eval(attr['use_bias'])),
        #  'kernel': attr['kernel_size'],
        #  'num_filter': attr['channels'],
        #  'stride': attr['strides'],
        #  'pad': attr['padding'],
        #  'dilate': attr['dilation'],
        #  'num_group': attr['groups'],
    #  }
    dw_np = topi.testing.dilate_python(w_np, dilation)
    c_np = topi.testing.conv2d_nchw_python(a_np, dw_np, stride, padding)
    print (c_np.flatten(), outs[0].flatten())



if __name__ == "__main__":
    utils.log_init()
    # opg.clean_dir()
    # verify_transpose()
    # verify_concatenate()
    # verify_take()
    # verify_strided_slice()
    # verify_repeat()
    # verify_tile()
    # verify_slice_like()

    # verify_max()
    # verify_sum()

    #  verify_conv2d()
    # verify_dense()

    # verify_max_pool2d()
    # verify_upsampling()

    # verify_broadcast()

    # test_load("conv2d", "ffd9ad6afc62dd7541778a81d6529c9a2735fc0a")

    # verify_get_valid_counts()
    # verify_non_max_suppression()
