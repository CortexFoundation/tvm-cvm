import unittest
import mxnet as mx

import nnvm

from _base import *

class TestNull(TfmTest):
    op = mx.sym.var("data", shape=[1, 3, 2, 1])

    def test_validate(self):
        self._assert_equal(self.op, self.op, "validate")

    def test_rewrite(self):
        self._assert_equal(self.op, self.op, "rewrite")

    def test_fuse_transpose(self):
        self._assert_equal(self.op, self.op, "fuse_transpose")

    def test_compile(self):
        des = nnvm.sym.Variable('data', __shape__=[1, 3, 2, 1])
        self._assert_equal(self.op, des, "compile")

class TestTranspose(TfmTest):
    op = mx.sym.transpose(
            mx.sym.transpose(TestNull.op, axes=[0, 2, 3, 1]),
            axes=[3, 2, 1, 0])

    def test_validate(self):
        self._assert_equal(self.op, self.op, "validate")

        op1 = mx.sym.transpose(TestNull.op, axes=[])
        des = mx.sym.transpose(TestNull.op, axes=[3, 2, 1, 0])
        self._assert_equal(op1, des, "validate")

        op2 = mx.sym.transpose(TestNull.op)
        self._assert_equal(op2, des, "validate")

    def test_rewrite(self):
        self._assert_equal(self.op, self.op, "rewrite")

    def test_fuse_transpose(self):
        des = mx.sym.transpose(TestNull.op, axes=[1, 3, 2, 0])
        self._assert_equal(self.op, des, "fuse_transpose")


class TestElemwiseAdd(TfmTest):
    def test_fuse_transpose(self):
        x = mx.sym.var("x", shape=(1,2,3,4))
        op1 = mx.sym.transpose(x, axes=[1, 3, 2, 0])
        op2 = mx.sym.transpose(x, axes=[1, 3, 2, 0])
        op = mx.symbol.elemwise_add(op1, op2)

        des = mx.sym.transpose(
                mx.sym.elemwise_add(x, x),
                axes=[1, 3, 2, 0])

        self._assert_equal(op, des, "fuse_transpose")


class TestBroadcastAdd(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2,3))
        y = mx.sym.var('y', shape=(2,1))
        ans = mx.sym.broadcast_add(x, y)

        x = nnvm.sym.Variable('x', __shape__=(2, 3))
        y = nnvm.sym.Variable('y', __shape__=(2, 1))
        des = nnvm.sym.broadcast_add(x, y)
        self._assert_equal(ans, des, "compile")


class TestBroadcastDiv(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2,3))
        y = mx.sym.var('y', shape=(2,1))
        ans = mx.sym.broadcast_div(x, y)

        x = nnvm.sym.Variable('x', __shape__=(2, 3))
        y = nnvm.sym.Variable('y', __shape__=(2, 1))
        des = nnvm.sym.broadcast_div(x, y)
        self._assert_equal(ans, des, "compile")


class TestBroadcastMul(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2,3))
        y = mx.sym.var('y', shape=(2,1))
        ans = mx.sym.broadcast_mul(x, y)

        x = nnvm.sym.Variable('x', __shape__=(2, 3))
        y = nnvm.sym.Variable('y', __shape__=(2, 1))
        des = nnvm.sym.broadcast_mul(x, y)
        self._assert_equal(ans, des, "compile")


class TestBroadcastGreater(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2,3))
        y = mx.sym.var('y', shape=(2,1))
        ans = mx.sym.broadcast_greater(x, y)

        x = nnvm.sym.Variable('x', __shape__=(2, 3))
        y = nnvm.sym.Variable('y', __shape__=(2, 1))
        des = nnvm.sym.broadcast_greater(x, y)
        self._assert_equal(ans, des, "compile")


class TestBroadcastSub(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(2,3))
        y = mx.sym.var('y', shape=(2,1))
        ans = mx.sym.broadcast_sub(x, y)

        x = nnvm.sym.Variable('x', __shape__=(2, 3))
        y = nnvm.sym.Variable('y', __shape__=(2, 1))
        des = nnvm.sym.broadcast_sub(x, y)
        self._assert_equal(ans, des, "compile")

class TestBroadcastTo(TfmTest):
    def test_compile(self):
        x = mx.sym.var('x', shape=(1,3))
        ans = mx.sym.broadcast_to(x, shape=(2,3))

        x = nnvm.sym.Variable('x', __shape__=(1,3))
        des = nnvm.sym.broadcast_to(x, shape=(2,3))
        self._assert_equal(ans, des, 'compile')


class TestConcat(TfmTest):
    def test_fuse_transpose(self):
        x = mx.sym.var('x', shape=(1, 3, 244, 244))
        op1 = mx.sym.transpose(x, axes=[3, 2, 1, 0])
        op2 = mx.sym.transpose(x, axes=[])
        op = mx.sym.Concat(op1, op2, dim=1)

        des = mx.sym.transpose(mx.sym.concat(x, x, dim=2), axes=[3,2,1,0])

        self._assert_equal(op, des,
                ["validate", "fuse_transpose"])

class TestSlice(TfmTest):
    def test_compile(self):
        data = mx.sym.var('data', shape=(20,))
        op = mx.sym.slice(data, begin=(0,), end=(6,))

        datan = nnvm.sym.Variable('data', __shape__=(20,))
        des = nnvm.sym.strided_slice(datan, begin=(0,), end=(6,))

        self._assert_equal(op, des, 'compile')

class TestReshape(TfmTest):
    def test_compile(self):
        data = mx.sym.var('data', shape=(24,))
        op = mx.sym.reshape(data, shape=(1, 2, 3, 4))

        datan = nnvm.sym.Variable('data', __shape__=(24,))
        des = nnvm.sym.reshape(datan, shape=(1, 2, 3, 4))

        self._assert_equal(op, des, 'compile')


if __name__ == "__main__":
    import sys
    unittest.main(argv=sys.argv, verbosity=5)
